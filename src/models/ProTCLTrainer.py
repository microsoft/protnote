import logging
from src.utils.data import load_gz_json
from src.utils.evaluation import EvalMetrics
from src.utils.losses import WeightedBCE
from src.utils.proteinfer import normalize_confidences
import numpy as np
import torch
import wandb
import os
import json
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict


class ProTCLTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        config: dict,
        logger: logging.Logger,
        timestamp: str,
        run_name: str,
        use_wandb: bool = False,
        pos_weight: torch.Tensor = None,

    ):
        """_summary_
        :param model: pytorch model
        :type model: torch.nn.Module
        :param device: decide for training on cpu or gpu
        :type device: str
        :param config: Training configuration
        :type config: dict
        :param logger: logger
        :type logger: logging.Logger
        :param timestamp: run timestamp
        :type timestamp: str
        :param use_wandb: whether to use weights and biases, defaults to False
        :type use_wandb: bool, optional
        :param run_name: name of the run
        :type run_name: str
        """

        self.model = model
        self.device = device
        self.run_name = run_name
        self.logger = logger
        self.timestamp = timestamp
        self.use_wandb = use_wandb
        self.num_labels = config["params"]["NUM_LABELS"]
        self.num_epochs = config["params"]["NUM_EPOCHS"]
        self.train_sequence_encoder = config["params"]["TRAIN_SEQUENCE_ENCODER"]
        self.train_label_encoder = config["params"]["TRAIN_LABEL_ENCODER"]
        self.train_projection_head = config["params"]["TRAIN_PROJECTION_HEAD"]

        self.normalize_probabilities = config["params"]["NORMALIZE_PROBABILITIES"]
        self.validation_frequency = config["params"]["VALIDATION_FREQUENCY"]
        self.gradient_accumulation_steps = config["params"]["GRADIENT_ACCUMULATION_STEPS"]
        self.label_vocabulary = config["paths"]["GO_LABEL_VOCAB_PATH"]
        self.label_normalizer = load_gz_json(
            config["paths"]["PARENTHOOD_LIB_PATH"]
        )
        self.output_model_dir = config["paths"]["OUTPUT_MODEL_DIR"]
        self.scaler = GradScaler()
        self.cached_label_embeddings = None
        self.label_sample_size = config["params"]["LABEL_SAMPLE_SIZE"]
        self._set_optimizer(config)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        print(pos_weight)

    #TODO: Eventually use factory method to get loss_fn based on config
    def _get_loss_fn(self, config):
        pass

    def _set_optimizer(self, config):

        trainable_params = []
        trainable_params_names = []

        for name, param in self.model.named_parameters():

            if (name.startswith('sequence_encoder') & (not self.train_sequence_encoder)):
                param.requires_grad = False

            if (name.startswith('label_encoder') & (not self.train_label_encoder)):
                param.requires_grad = False

            if (name.startswith('W_p.weight') | name.startswith('W_l.weight')) & (not self.train_projection_head):
                param.requires_grad = False

            if param.requires_grad:
                trainable_params.append(param)
                trainable_params_names.append(name)

        self.trainable_params_names = trainable_params_names
        
        self.optimizer = torch.optim.Adam(
            trainable_params, lr=config["params"]["LEARNING_RATE"]
        )

    def evaluation_step(self, batch) -> tuple:
        """Perform a single evaluation step.

        :param batch: _description_
        :type batch: _type_
        :return: batch loss, logits and labels
        :rtype: tuple
        """

        # Unpack the validation batch
        sequence_ids, sequence_onehots, label_multihots, sequence_lengths = batch

        # Move sequences and labels to GPU, if available
        labels = label_multihots.to(self.device)
        sequences = sequence_onehots.to(self.device)
        sequence_lengths = sequence_lengths.to(self.device)

        # Consider all labels for the validation loop
        considered_labels = torch.arange(labels.size(1)).to(self.device)

        # Forward pass
        with autocast():
            logits = self.model(
                sequences, considered_labels, sequence_lengths
            )

        # Compute validation loss for the batch
        loss =  self.loss_fn(logits,labels.float())

        return loss.item(), logits, labels, sequence_ids

    def validate(self, val_loader: torch.utils.data.DataLoader, best_val_loss: float):

        self.logger.info("Running validation...")

        eval_metrics = EvalMetrics(
            num_labels=self.num_labels, threshold=0.85, device=self.device
        ).get_metric_collection(type="all")

        val_metrics, _ = self.evaluate(data_loader=val_loader,eval_metrics=eval_metrics)

        self.logger.info(val_metrics)

        if self.use_wandb:
            wandb.log({"validation_loss": val_metrics["avg_loss"],"validation_map_micro": val_metrics["map_micro"]})

        # Save the model if it has the best validation loss so far
        if val_metrics["avg_loss"] < best_val_loss:
            self.logger.info(
                f"New best validation loss: {val_metrics['avg_loss']}. Saving model..."
            )
            best_val_loss = val_metrics["avg_loss"]

            # Save model to OUTPUT_MODEL_DIR. Create path if it doesn't exist.
            if not os.path.exists(self.output_model_dir):
                os.makedirs(self.output_model_dir)

            model_name = (
                self.run_name if self.run_name else "best_ProTCL.pt"
            )
            model_path = os.path.join(
                self.output_model_dir, f"{self.timestamp}_{model_name}.pt"
            )
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Saved model to {model_path}.")

            if self.use_wandb:
                wandb.save(f"{self.timestamp}_best_ProTCL.pt")

        return val_metrics, best_val_loss

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

        with torch.no_grad(), autocast():
            all_probabilities = []
            all_labels = []
            for batch in data_loader:
                _, logits, labels, _ = self.evaluation_step(
                    batch=batch)

                # Apply sigmoid to get the probabilities for multi-label classification
                probabilities = torch.sigmoid(logits)

                if self.normalize_probabilities:
                    # TODO: Using original normalize_confidences implemented with numpy,
                    # but this is slow. Should be able to do this with torch tensors.
                    probabilities = torch.tensor(
                        normalize_confidences(
                            predictions=probabilities.detach().cpu().numpy(),
                            label_vocab=self.label_vocabulary,
                            applicable_label_dict=self.label_normalizer,
                        ),
                        device=self.device,
                    )

                all_probabilities.append(probabilities)
                all_labels.append(labels)

            all_probabilities = torch.cat(all_probabilities)
            all_labels = torch.cat(all_labels)

        for th in np.arange(0.1, 1, 0.01):
            eval_metrics = EvalMetrics(
                num_labels=self.num_labels, threshold=th, device=self.device
            ).get_metric_collection(type="all")

            optimization_metric = getattr(
                eval_metrics, optimization_metric_name)

            optimization_metric(all_probabilities, all_labels)
            score = optimization_metric.compute()
            if score > best_score:
                best_score = score
                best_th = th
            print("TH:", th, "F1:", score)

        best_score = best_score.item()
        self.logger.info(
            f"Best validation score: {best_score}, Best val threshold: {best_th}"
        )
        self.model.train()
        return best_th, best_score

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        eval_metrics: EvalMetrics = None
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
        total_loss = 0
        test_results = defaultdict(list)
        with torch.no_grad(), autocast():
            for batch in data_loader:
                loss, logits, labels, sequence_ids = self.evaluation_step(
                    batch=batch)
                if eval_metrics is not None:
                    # Apply sigmoid to get the probabilities for multi-label classification
                    probabilities = torch.sigmoid(logits)

                    if self.normalize_probabilities:
                        # TODO: Using original normalize_confidences implemented with numpy,
                        # but this is slow. Should be able to do this with torch tensors.
                        probabilities = torch.tensor(
                            normalize_confidences(
                                predictions=probabilities.detach().cpu().numpy(),
                                label_vocab=self.label_vocabulary,
                                applicable_label_dict=self.label_normalizer,
                            ),
                            device=self.device,
                        )
                    # Update eval metrics
                    eval_metrics(probabilities, labels)

                    test_results["sequence_ids"].append(
                        # could use .repeat_interleave(num_labels) to make long format
                        sequence_ids
                    )
                    test_results["probabilities"].append(probabilities)
                    test_results["labels"].append(labels)

                # Accumulate loss
                total_loss += loss

            for key in test_results.keys():
                test_results[key] = (
                    torch.cat(test_results[key]).detach().cpu().numpy()
                )

            # Compute average validation loss
            avg_loss = total_loss / len(data_loader)
            final_metrics = eval_metrics.compute() if eval_metrics is not None else {}
            final_metrics.update({"avg_loss": avg_loss})

            final_metrics = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in final_metrics.items()
            }

            for k, v in final_metrics.items():
                if isinstance(v, torch.Tensor):
                    final_metrics[k] = v.item()
                
                #Cast numpy floats to float32. Needed to store as parquet
                # because pyarrow doesn't support float16 from mixed precision
                if np.issubdtype(type(v), np.floating):
                    final_metrics[k] = v.astype("float32")
        
        self.model.train()
        return final_metrics, test_results

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ):
        """Train model
        :param train_loader: _description_
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: _description_
        :type val_loader: torch.utils.data.DataLoader
        """
        # Log that training is starting
        self.logger.info("Starting training...")

        self.model.train()
        # Watch the model
        if self.use_wandb:
            wandb.watch(self.model)

        # Keep track of batches for logging
        batch_count = 1

        # Initialize the best validation loss to a high value
        best_val_loss = float("inf")

        num_training_steps = len(train_loader) * self.num_epochs

        self.logger.info(
            f"Total number of training steps: {num_training_steps}")

        for epoch in range(self.num_epochs):
            ####### TRAINING LOOP #######
            for train_batch in train_loader:
                # Unpack batch
                # TODO: Modify datasets and data loaders to no longer return sequence_ids
                (
                    _,
                    sequence_onehots,
                    label_multihots,
                    sequence_lengths,
                ) = train_batch

                # Move sequences, labels and sequence_lengths to GPU, if available
                labels = label_multihots.to(self.device)
                sequence_lengths = sequence_lengths.to(self.device)
                sequences = sequence_onehots.to(self.device)

                # Randomly sample n labels from the batch
                random_indices = torch.randperm(labels.size(1))[
                    :self.label_sample_size].to(self.device)


                # Compute target
                target = labels.index_select(1, random_indices)

                # Forward pass
                with autocast():
                    logits = self.model(
                        sequences, random_indices, sequence_lengths
                    )


                # log average probabilities to W&B
                if self.use_wandb:
                    with torch.no_grad():
                        wandb.log({"avg_probabilities": torch.mean(torch.sigmoid(logits))})


                # Compute loss
                loss = self.loss_fn(logits,target.float()) / self.gradient_accumulation_steps

                # Log metrics to W&B
                if self.use_wandb:
                    wandb.log({"train_loss": loss.item()})

                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()

                # Gradient accumulation every GRADIENT_ACCUMULATION_STEPS
                if ((batch_count) % self.gradient_accumulation_steps == 0):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Run validation and log progress every n batches
                if batch_count % self.validation_frequency == 0:
                    ####### VALIDATION LOOP #######
                    # Force model to recompute all label embeddings
                    if self.train_label_encoder:
                        self.model.clear_label_embeddings_cache()

                    # Run validation
                    val_metrics, best_val_loss = self.validate(val_loader=val_loader,
                                                               best_val_loss=best_val_loss)

                    self.logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs}, Batch {batch_count}, Training Loss: {loss.item()}, Validation Loss: {val_metrics['avg_loss']}"
                    )

                    self.logger.info(
                        f"Validation metrics:\n{json.dumps(val_metrics, indent=4)}")
                # Log training progress percentage every 5%
                if num_training_steps > 20 and batch_count % int(num_training_steps/20) == 0:
                    self.logger.info(
                        f"Training progress: {round(100*batch_count/num_training_steps,2)}%")

                # Increment batch count
                batch_count += 1
