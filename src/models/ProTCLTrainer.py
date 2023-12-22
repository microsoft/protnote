import logging
from src.utils.data import load_gz_json, log_gpu_memory_usage
from src.utils.evaluation import EvalMetrics,metric_collection_to_dict_float,save_evaluation_results
from src.utils.losses import BatchWeightedBCE, FocalLoss, RGDBCE, WeightedBCE,SupCon, CBLoss
from torchmetrics import MetricCollection, Metric
from src.utils.proteinfer import normalize_confidences
import numpy as np
import torch
import wandb
import os
import json
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import BatchEncoding
from src.utils.models import generate_label_embeddings_from_text, biogpt_train_last_n_layers, save_checkpoint, load_model
from torcheval.metrics import MultilabelAUPRC, BinaryAUPRC
from torch.utils.tensorboard import SummaryWriter

class ProTCLTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        config: dict,
        vocabularies: dict,
        logger: logging.Logger,
        timestamp: str,
        run_name: str,
        loss_fn: torch.nn.Module,
        use_wandb: bool = False,
        is_master: bool = True,
        starting_epoch: int = 1
    ):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (str): The device to use for training (e.g., 'cpu' or 'cuda').
            logger (logging.Logger): The logger to use for logging training progress.
            timestamp (str): The timestamp to use for naming log files and checkpoints.
            run_name (str): The name of the current training run.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. Defaults to False.
            bce_pos_weight (torch.Tensor, optional): The positive weight for binary cross-entropy loss. Defaults to None.
            is_master (bool, optional): Whether the current process is the master process. Defaults to True.
            starting_epoch (int, optional): The starting epoch number. Defaults to 1. Used for resuming training.
        """

        self.model = model
        self.is_master = is_master
        self.device = device
        self.run_name = run_name
        self.logger = logger
        self.timestamp = timestamp
        self.use_wandb = use_wandb
        self.num_epochs = config["params"]["NUM_EPOCHS"]
        self.train_sequence_encoder = config["params"]["TRAIN_SEQUENCE_ENCODER"]
        self.label_encoder_num_trainable_layers = config["params"]["LABEL_ENCODER_NUM_TRAINABLE_LAYERS"]
        self.train_projection_head = config["params"]["TRAIN_PROJECTION_HEAD"]

        self.normalize_probabilities = config["params"]["NORMALIZE_PROBABILITIES"]
        self.EPOCHS_PER_VALIDATION = config["params"]["EPOCHS_PER_VALIDATION"]
        self.gradient_accumulation_steps = config["params"]["GRADIENT_ACCUMULATION_STEPS"]
        self.clip_value = config["params"]["CLIP_VALUE"]
        self.vocabularies = vocabularies
        self.label_normalizer = load_gz_json(
            config["paths"]["PARENTHOOD_LIB_PATH"]
        )
        self.output_model_dir = config["paths"]["OUTPUT_MODEL_DIR"]
        self.lora_params = {'rank':config["params"]["LORA_RANK"],
                            'in_features':config["params"]["LABEL_EMBEDDING_DIM"],
                            'out_features':config["params"]["LABEL_EMBEDDING_DIM"],
                            'device':self.device
                            } if config["params"]["LORA"] else None
        
        self._set_optimizer(opt_name = config["params"]["OPTIMIZER"],
                            lr = config["params"]["LEARNING_RATE"])
        
        self.loss_fn = loss_fn
        self.model_path = self._get_saved_model_path()
        self.best_val_metric = 0.0
        self.scaler = GradScaler()
        self.starting_epoch = starting_epoch
        self.epoch = starting_epoch
        self.config = config
        self.tb = SummaryWriter(f"runs/{self.run_name}_{self.timestamp}") if self.is_master else None

    def _get_saved_model_path(self):
        # Save model to OUTPUT_MODEL_DIR. Create path if it doesn't exist.
        if not os.path.exists(self.output_model_dir) and self.is_master:
            os.makedirs(self.output_model_dir)

        model_name = (
            self.run_name if self.run_name else "best_ProTCL.pt"
        )
        model_path = os.path.join(
            self.output_model_dir, f"{self.timestamp}_{model_name}.pt"
        )
        return model_path

    def _to_device(self, *args):
        processed_args = []
        for item in args:
            if isinstance(item, torch.Tensor):
                processed_args.append(item.to(self.device))
            elif isinstance(item, BatchEncoding) or isinstance(item, dict):
                processed_dict = {k: v.to(self.device) if isinstance(
                    v, torch.Tensor) else v for k, v in item.items()}
                processed_args.append(processed_dict)
            else:
                processed_args.append(item)
        return processed_args

    def _set_optimizer(self, opt_name, lr):
        trainable_params = []
        trainable_params_names = []

        # Use to unfreeze last n layers. 0 means entire model frozen.
        biogpt_train_last_n_layers(self.model.module.label_encoder,
                                   self.label_encoder_num_trainable_layers,
                                   lora_params=self.lora_params
                                   )
        
        for name, param in self.model.module.named_parameters():
            if name.startswith('sequence_encoder') and (not self.train_sequence_encoder):
                param.requires_grad = False

            if (name.startswith('W_p.weight') or name.startswith('W_l.weight')) and (not self.train_projection_head):
                param.requires_grad = False

            if name.startswith('output_layer') and (not self.train_projection_head):
                param.requires_grad = False

            if param.requires_grad:
                trainable_params.append(param)
                trainable_params_names.append(name)

        self.trainable_params_names = trainable_params_names

        if opt_name == 'Adam':
            opt = torch.optim.Adam
        elif opt_name == 'SGD':
            opt = torch.optim.SGD
        else:
            raise ValueError("Unsupported optimizer name")

        self.optimizer = opt(
            trainable_params, lr=lr
        )

    def evaluation_step(self, batch) -> tuple:
        """Perform a single evaluation step.

        :param batch: _description_
        :type batch: _type_
        :return: batch loss, logits and labels
        :rtype: tuple
        """

        # Unpack the validation or testing batch
        sequence_onehots, sequence_embeddings, sequence_lengths, sequence_ids, label_multihots, tokenized_labels, label_embeddings = (
            batch["sequence_onehots"],
            batch["sequence_embeddings"],
            batch["sequence_lengths"],
            batch["sequence_ids"],
            batch["label_multihots"],
            batch["tokenized_labels"],
            batch["label_embeddings"]
        )

        # Move all unpacked batch elements to GPU, if available
        sequence_onehots, sequence_embeddings, sequence_lengths, label_multihots, tokenized_labels, label_embeddings = self._to_device(
            sequence_onehots, sequence_embeddings, sequence_lengths, label_multihots, tokenized_labels, label_embeddings)

        # Forward pass
        inputs = {
            "sequence_onehots": sequence_onehots,
            "sequence_embeddings": sequence_embeddings,
            "sequence_lengths": sequence_lengths,
            "tokenized_labels": tokenized_labels,
            "label_embeddings": label_embeddings
        }
        with autocast():
            logits = self.model(**inputs)
            # Compute validation loss for the batch
            loss = self.loss_fn(logits, label_multihots.float())

        return loss.item(), logits, label_multihots, sequence_ids

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 eval_metrics: MetricCollection,
                 val_optimization_metric_name: str
                 ):

        self.logger.info("Running validation...")

        prefix = 'validation'

        val_metrics = self.evaluate(data_loader=val_loader,
                                       eval_metrics=eval_metrics,
                                       metrics_prefix=prefix)
        val_optimization_metric_name = f'{prefix}_{val_optimization_metric_name}'

        
        self.logger.info("+-------------------------------- Validation Results --------------------------------+")
        # Print memory consumption
        if self.is_master:
            log_gpu_memory_usage(self.logger, 0)
        self.logger.info(
            f"Validation metrics:\n{json.dumps(val_metrics, indent=4)}")

        if self.use_wandb and self.is_master:
            try:
                if self.use_wandb and self.is_master:
                    wandb.log(val_metrics,
                              step=self.training_step
                              )

            except Exception as e:
                self.logger.warning(
                    f"Failed to log validation metrics to wandb: {e}")

        # Save the model if it has the best validation loss so far (only on master node)
        if self.is_master and val_metrics[val_optimization_metric_name] > self.best_val_metric:
            self.logger.info(
                f"New best {val_optimization_metric_name}: {val_metrics[val_optimization_metric_name]}. Saving model..."
            )
            self.best_val_metric = val_metrics[val_optimization_metric_name]

            save_checkpoint(
                model=self.model.module,
                optimizer=self.optimizer,
                epoch=self.epoch,
                best_val_metric=self.best_val_metric,
                model_path=self.model_path
            )
            self.logger.info(f"Saved model to {self.model_path}.")

            if self.use_wandb:
                wandb.save(f"{self.timestamp}_best_ProTCL.pt")
        
        self.logger.info("+------------------------------------------------------------------------------------+") 

        return val_metrics

    def find_optimal_threshold(
        self, data_loader: torch.utils.data.DataLoader, optimization_metric_name: str
    ) -> tuple[float, float]:
        """Find the optimal threshold for the given data loader.

        :param data_loader: _description_
        :type data_loader: torch.utils.data.DataLoader
        :param average: _description_
        :type average: Literal[&#39;micro&#39;, &#39;macro&#39;, &#39;weighted&#39;]
        :param optimization_metric_name: _description_
        :type optimization_metric_name: str
        :return: _description_
        :rtype: tuple[float, float]
        """

        self.logger.info("Finding optimal threshold...")
        self.model.eval()

        best_th = 0.0
        best_score = 0.0

        with torch.no_grad():
            all_probabilities = []
            all_label_multihots = []
            for batch in data_loader:
                _, logits, label_multihots, _ = self.evaluation_step(
                    batch=batch)

                # Apply sigmoid to get the probabilities for multi-label classification
                probabilities = torch.sigmoid(logits)

                if self.normalize_probabilities:
                    probabilities = self._normalize_probabilities(probabilities)

                all_probabilities.append(probabilities)
                all_label_multihots.append(label_multihots)

            all_probabilities = torch.cat(all_probabilities)
            all_label_multihots = torch.cat(all_label_multihots)

        for th in np.arange(0.1, 1, 0.01):
            optimization_metric = EvalMetrics(device=self.device)\
                .get_metric_by_name(name=optimization_metric_name,
                                    threshold=th,
                                    num_labels=label_multihots.shape[-1])

            optimization_metric(all_probabilities, all_label_multihots)
            score = optimization_metric.compute().item()
            if score > best_score:
                best_score = score
                best_th = th
            self.logger.info("TH: {:.3f}, F1: {:.3f}".format(th, score))

        best_score = best_score
        self.logger.info(
            f"Best validation score: {best_score}, Best val threshold: {best_th}"
        )
        self.model.train()
        return best_th, best_score

    def _normalize_probabilities(self,probabilities):
        # TODO: Using original normalize_confidences implemented with numpy,
                    # but this is slow. Should be able to do this with torch tensors.
        return torch.tensor(
                    normalize_confidences(
                        predictions=probabilities.detach().cpu().numpy(),
                        label_vocab=self.vocabularies["GO_label_vocab"],
                        applicable_label_dict=self.label_normalizer,
                    ),
                    device=self.device,
                )

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        eval_metrics: MetricCollection = None,
        save_results: bool = False,
        metrics_prefix = None
    ) -> tuple[dict, dict]:
        """Evaluate the model on the given data loader.
        :param data_loader: pytorch data loader
        :type data_loader: torch.utils.data.DataLoader
        :param eval_metrics: an eval metrics class to calculate metrics like F1 score, defaults to None
        :type eval_metrics: EvalMetrics, optional
        :return: dictionary with evaluation metrics. Always return avg_loss and if eval_metrics is not None, it will return the metrics from eval_metrics.compute()
        :rtype: dict
        """
        self.model.eval()

        # Compute all label embeddings upfront, since we're not training
        if data_loader.dataset.label_embedding_matrix is None:
            logging.info(
                "Computing label embeddings for evaluation...")
            with torch.no_grad():
                label_embedding_matrix = generate_label_embeddings_from_text(
                    data_loader.dataset.label_text_list,
                    data_loader.dataset.label_tokenizer,
                    self.model.module.label_encoder,
                    self.config["params"]["LABEL_BATCH_SIZE_LIMIT_NO_GRAD"],
                ).cpu()
            data_loader.dataset.set_label_embedding_matrix(
                label_embedding_matrix)
            logging.info("Done computing label embeddings.")

        total_loss = 0
        test_results = defaultdict(list)

        if eval_metrics is not None:
            eval_metrics.reset()

        mAP_micro = BinaryAUPRC(device='cpu')
        mAP_macro = MultilabelAUPRC(device='cpu',num_labels=len(self.vocabularies["GO_label_vocab"]))

        with torch.no_grad():
            
            for batch_idx, batch in enumerate(data_loader):
                loss, logits, labels, sequence_ids = self.evaluation_step(batch=batch)
                if eval_metrics is not None:
                    # Apply sigmoid to get the probabilities for multi-label classification
                    probabilities = torch.sigmoid(logits)

                    if self.normalize_probabilities:
                        probabilities = self._normalize_probabilities()

                    # Update eval metrics
                    eval_metrics(probabilities, labels)

                    mAP_micro.update(probabilities.cpu().flatten(), labels.cpu().flatten())
                    mAP_macro.update(probabilities.cpu(), labels.cpu())

                    # No need to save results everytime. Only need it for final evaluation.
                    if save_results:
                        test_results["sequence_ids"].append(sequence_ids)
                        test_results["logits"].append(logits.cpu())
                        test_results["labels"].append(labels.cpu())

                # Print progress every 10%
                if batch_idx % (len(data_loader) // 10) == 0:
                    self.logger.info(f"Epoch {self.epoch}: Processed {batch_idx} out of {len(data_loader)} batches ({batch_idx / len(data_loader) * 100:.2f}%).")  


                # Accumulate loss
                total_loss += loss

            if save_results:
                for key in test_results.keys():
                    if key == "sequence_ids":
                        test_results[key] = (
                            np.array(
                                [j for i in test_results["sequence_ids"] for j in i])
                        )
                    else:
                        test_results[key] = (
                            torch.cat(test_results[key]).numpy()
                        )
                
                self.logger.info("Saving validation results...")
                if self.is_master:
                    save_evaluation_results(results=test_results,
                                            label_vocabulary=self.vocabularies["GO_label_vocab"],
                                            run_name=self.run_name,
                                            output_dir=self.config["paths"]["RESULTS_DIR"],
                                            data_split_name=metrics_prefix
                                            )
                

            # Compute average validation loss
            avg_loss = total_loss / len(data_loader)

            final_metrics = eval_metrics.compute() if eval_metrics is not None else {}
            final_metrics.update({"loss": avg_loss,
                                  "map_micro":mAP_micro.compute(),
                                  "map_macro":mAP_macro.compute()
                                  })

            final_metrics = metric_collection_to_dict_float(
                final_metrics,
                prefix=metrics_prefix)           

        self.model.train()

        return final_metrics

    def train_one_epoch(self,
                        train_loader: torch.utils.data.DataLoader,
                        eval_metrics: MetricCollection
        ):
        
        avg_loss = 0
        avg_probs = 0
        avg_gt = 0
        eval_metrics.reset()
        
        ####### TRAINING LOOP #######
        for batch_idx, batch in enumerate(train_loader):
            self.training_step += 1

            # Unpack the training batch
            sequence_onehots, sequence_embeddings, sequence_lengths, label_multihots, tokenized_labels, label_embeddings = (
                batch["sequence_onehots"],
                batch["sequence_embeddings"],
                batch["sequence_lengths"],
                batch["label_multihots"],
                batch["tokenized_labels"],
                batch["label_embeddings"]
            )

            # Move all unpacked batch elements to GPU, if available
            sequence_onehots, sequence_embeddings, sequence_lengths, label_multihots, tokenized_labels, label_embeddings = self._to_device(
                sequence_onehots, sequence_embeddings, sequence_lengths, label_multihots, tokenized_labels, label_embeddings)

            # Forward pass
            inputs = {
                "sequence_onehots": sequence_onehots,
                "sequence_embeddings": sequence_embeddings,
                "sequence_lengths": sequence_lengths,
                "tokenized_labels": tokenized_labels,
                "label_embeddings": label_embeddings
            }

            with autocast():
                logits = self.model(**inputs)

                # Compute loss, normalized by the number of gradient accumulation steps
                loss = self.loss_fn(logits, label_multihots.float()) / \
                    self.gradient_accumulation_steps
        
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()

            # Gradient accumulation every GRADIENT_ACCUMULATION_STEPS
            if (self.training_step % self.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):     
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                
                # Apply gradient clipping
                if self.clip_value is not None:
                    clip_grad_norm_(self.model.module.parameters(),
                                    max_norm=self.clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Log at this point to TB to have weights and gradients after a full epoch
                if (batch_idx + 1 == len(train_loader)) & self.is_master:
                    for name, weight in self.model.module.named_parameters():
                        if weight.requires_grad:
                            self.tb.add_histogram(name,weight, self.epoch)
                            self.tb.add_histogram(f'{name}.grad',weight.grad, self.epoch)

                self.optimizer.zero_grad()
            
            avg_loss+=loss.item()
            avg_probs += torch.mean(torch.sigmoid(logits).detach())
            avg_gt += torch.mean(label_multihots.float().detach())

            eval_metrics(logits, label_multihots)
            
            if self.use_wandb:
                wandb.log({"per_batch_train_loss": loss.item()},
                          step=self.training_step
                          )

            # Print memory consumption after first batch (to get the max memory consumption during training)
            if batch_idx == 1 and self.is_master:
                self.logger.info("+----------------- Train GPU Memory Usage -----------------+")
                log_gpu_memory_usage(self.logger, 0)
                self.logger.info("+----------------------------------------------------------+")
                
            # Print progress every 10%
            if batch_idx % (len(train_loader) // 10) == 0:
                self.logger.info(f"Epoch {self.epoch}: Processed {batch_idx} out of {len(train_loader)} batches ({batch_idx / len(train_loader) * 100:.2f}%).")  

        avg_loss = avg_loss/len(train_loader) if len(train_loader)> 0 else avg_loss
        avg_probs_gt_ration = avg_probs/avg_gt

        train_metrics = eval_metrics.compute() if eval_metrics is not None else {}
        train_metrics.update({"loss": avg_loss,
                              "avg_probabilities_ground_truth_ratio":avg_probs_gt_ration,
                                })
        train_metrics = metric_collection_to_dict_float(train_metrics,prefix='train')
        
        if self.use_wandb:
            wandb.log(train_metrics,
                      step=self.training_step
                      )

        
        return train_metrics
        
        
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_eval_metrics: MetricCollection,
        val_eval_metrics: MetricCollection,
        val_optimization_metric_name: str
    ):
        """Train model
        :param train_loader: _description_
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: _description_
        :type val_loader: torch.utils.data.DataLoader
        """
        self.model.train()

        # Watch the model
        if self.use_wandb:
            wandb.watch(self.model)

        # Compute total number of training steps
        self.training_step = 0
        num_training_steps = len(train_loader) * self.num_epochs
        
        self.logger.info(f"{'='*100}")
        self.logger.info(
            f"Starting training. Total number of training steps: {num_training_steps}")
        self.logger.info(f"{'='*100}")

        for epoch in range(self.starting_epoch, self.starting_epoch + self.num_epochs):
            self.logger.info(
                f"Starting epoch {epoch}/{self.starting_epoch + self.num_epochs - 1}...")
            self.epoch = epoch

            # Set distributed loader epoch to shuffle data
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_one_epoch(train_loader=train_loader,
                                                 eval_metrics=train_eval_metrics)
                

            if (epoch % self.EPOCHS_PER_VALIDATION == 0):
                ####### VALIDATION LOOP #######
                torch.cuda.empty_cache()

                # Run validation
                self.validate(val_loader=val_loader,
                                            eval_metrics=val_eval_metrics,
                                            val_optimization_metric_name=val_optimization_metric_name
                                            )

                if self.label_encoder_num_trainable_layers>0:
                    # Clear the label embedding matrix
                    val_loader.dataset.set_label_embedding_matrix(None)

                self.logger.info(
                    f"Epoch {epoch}/{self.starting_epoch + self.num_epochs - 1}, Batch {self.training_step}, Training Loss: {train_metrics['train_loss']}"
                )

        if self.is_master:
            self.logger.info("Restoring model to best validation map_micro...")
            load_model(trainer=self,
                            checkpoint_path=self.model_path)

            # Broadcast model state to other processes
            for param in self.model.module.parameters():
                # src=0 means source is the master process
                torch.distributed.broadcast(param.data, src=0)
        else:
            # For non-master processes, just receive the broadcasted data
            for param in self.model.module.parameters():
                torch.distributed.broadcast(param.data, src=0)
        
        self.tb.close()