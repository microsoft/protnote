import logging
from src.utils.data import load_gz_json
from src.utils.losses import contrastive_loss
from src.utils.evaluation import EvalMetrics
from src.utils.proteinfer import normalize_confidences
from typing import Literal
import numpy as np
import torch
import wandb
import os
from torch.cuda.amp import GradScaler, autocast


class ProTCLTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: str,
                 config: dict,
                 logger: logging.Logger,
                 timestamp: str,
                 run_name: str,
                 use_wandb: bool = False
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
        self.num_labels = config['params']['NUM_LABELS']
        self.num_epochs = config['params']['NUM_EPOCHS']
        self.train_sequence_encoder = config['params']['TRAIN_SEQUENCE_ENCODER']
        self.train_label_encoder = config['params']['TRAIN_LABEL_ENCODER']
        self.embedding_update_interval = config['params']['EMBEDDING_UPDATE_INTERVAL']
        self.use_batch_labels_only = config['params']['USE_BATCH_LABELS_ONLY']
        self.normalize_probabilities = config['params']['NORMALIZE_PROBABILITIES']
        self.validation_frequency = config['params']['VALIDATION_FREQUENCY']
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config['params']['LEARNING_RATE'])
        self.label_vocabulary = config['relative_paths']['GO_LABEL_VOCAB_PATH']
        self.label_normalizer = load_gz_json(
            config['relative_paths']['PARENTHOOD_LIB_PATH'])
        self.output_model_dir = config['relative_paths']['OUTPUT_MODEL_DIR']
        self.scaler = GradScaler()
        self.cached_label_embeddings = None

    def evaluation_step(self, batch, testing=False) -> tuple:
        """Perform a single evaluation step.

        :param batch: _description_
        :type batch: _type_
        :return: batch loss, logits and labels
        :rtype: tuple
        """

        # If testing, always use all the labels
        _use_batch_labels_only = self.use_batch_labels_only and not testing

        # Unpack the validation batch
        sequence_ids, sequence_onehots, label_multihots, sequence_lengths = batch

        # Move sequences and labels to GPU, if available
        labels = label_multihots.to(self.device)
        sequences = sequence_onehots.to(
            self.device) if self.train_sequence_encoder else sequence_ids.to(self.device)

        considered_labels = labels if _use_batch_labels_only else torch.ones_like(
            labels)

        # Forward pass
        P_e, L_e = self.model(sequences, considered_labels, is_training=False)

        # Compute validation loss for the batch
        loss = contrastive_loss(P_e,
                                L_e,
                                self.model.t,
                                labels[:, torch.any(
                                    labels, dim=0)] if _use_batch_labels_only else labels
                                )
        # Compute temperature normalized cosine similarities
        logits = torch.mm(P_e,
                          L_e.t()) / self.model.t

        return loss.item(), logits, labels

    def find_optimal_threshold(self,
                               data_loader: torch.utils.data.DataLoader,
                               average: Literal['micro', 'macro', 'weighted'],
                               optimization_metric_name: str
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

        # Reset eval metrics just in case
        with torch.no_grad():
            all_probabilities = []
            all_labels = []
            for batch in data_loader:
                _, logits, labels = self.evaluation_step(
                    batch=batch, testing=True)

                # Apply sigmoid to get the probabilities for multi-label classification
                probabilities = torch.sigmoid(logits)

                if self.normalize_probabilities:

                    # TODO: Using original normalize_confidences implemented with numpy,
                    # but this is slow. Should be able to do this with torch tensors.
                    probabilities = torch.tensor(normalize_confidences(predictions=probabilities.detach().cpu().numpy(),
                                                                       label_vocab=self.label_vocabulary,
                                                                       applicable_label_dict=self.label_normalizer),
                                                 device=self.device)

                # # Only append if the probability tensor is the same size as the existing tensors in all_probabilities (to avoid partial batch issues)
                # if len(all_probabilities) == 0 or probabilities.shape == all_probabilities[0].shape:
                all_probabilities.append(probabilities)
                all_labels.append(labels)

            all_probabilities = torch.cat(all_probabilities)
            all_labels = torch.cat(all_labels)

        for th in np.arange(0.1, 1, 0.01):
            eval_metrics = EvalMetrics(
                num_labels=self.num_labels, average=average, threshold=th, device=self.device)
            optimization_metric = getattr(
                eval_metrics, optimization_metric_name)

            optimization_metric(all_probabilities, all_labels)
            score = optimization_metric.compute()
            if score > best_score:
                best_score = score
                best_th = th
            print('TH:', th, 'F1:', score)

        best_score = best_score.item()
        self.logger.info(
            f'Best validation score: {best_score}, Best val threshold: {best_th}')

        return best_th, best_score

    def evaluate(self,
                 data_loader: torch.utils.data.DataLoader,
                 eval_metrics: EvalMetrics = None,
                 testing=False) -> dict:
        """Evaluate the model on the given data loader.

        :param data_loader: pytorch data loader
        :type data_loader: torch.utils.data.DataLoader
        :param eval_metrics: an eval metrics class to calculate metrics like F1 score, defaults to None
        :type eval_metrics: EvalMetrics, optional
        :return: dictionary with evaluation metrics. Always return avg_loss and if eval_metrics is not None, it will return the metrics from eval_metrics.compute()
        :rtype: dict
        """

        # If testing, always use all the labels
        _use_batch_labels_only = self.use_batch_labels_only and not testing

        if _use_batch_labels_only & (eval_metrics is not None):
            raise ValueError(
                "Cannot use batch labels only and eval metrics within the training or validation loop at the same time.")

        self.model.eval()
        total_loss = 0

        with torch.no_grad(), autocast():
            for batch in data_loader:
                loss, logits, labels = self.evaluation_step(
                    batch=batch, testing=testing)
                if eval_metrics is not None:
                    # Apply sigmoid to get the probabilities for multi-label classification
                    probabilities = torch.sigmoid(logits)

                    if self.normalize_probabilities:
                        # TODO: Using original normalize_confidences implemented with numpy,
                        # but this is slow. Should be able to do this with torch tensors.
                        probabilities = torch.tensor(normalize_confidences(predictions=probabilities.detach().cpu().numpy(),
                                                                           label_vocab=self.label_vocabulary,
                                                                           applicable_label_dict=self.label_normalizer),
                                                     device=self.device)
                    # Update eval metrics
                    eval_metrics(probabilities, labels)

                # Accumulate loss
                total_loss += loss

            # Compute average validation loss
            avg_loss = total_loss / len(data_loader)
            final_metrics = eval_metrics.compute() if eval_metrics is not None else {}
            final_metrics.update({"avg_loss": avg_loss})

        return final_metrics

    def train(self,
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
        batch_count = 0

        # Initialize the best validation loss to a high value
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            ####### TRAINING LOOP #######
            for train_batch in train_loader:
                # Unpack batch
                sequence_ids, sequence_onehots, label_multihots, sequence_lengths = train_batch

                # Move sequences and labels to GPU, if available
                labels = label_multihots.to(self.device)
                sequences = sequence_onehots.to(
                    self.device) if self.train_sequence_encoder else sequence_ids.to(self.device)

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass (project sequences and relevant labels to latent space
                # Labels can be only the labels that are present in the batch (if self.USE_BATCH_LABELS is True) or all the labels (if False)
                considered_labels = labels if self.use_batch_labels_only else torch.ones_like(
                    labels)
                with autocast():
                    P_e, L_e = self.model(sequences, considered_labels)

                # Compute target (note that the target shape varies depending on whether we are using batch labels only or not)
                target = labels[:, torch.any(
                    labels, dim=0)] if self.use_batch_labels_only else labels

                # Assert that first dimension of L_e is the same as the second dimension of target, otherwise we have a mismatch
                assert L_e.shape[0] == target.shape[1], \
                    f"Expected the first dimension of L_e shape ({L_e.shape[1]}) to be the same as " \
                    f"target shape second dimension ({target.shape[1]}), but they are different."

                # Compute loss
                loss = contrastive_loss(P_e, L_e, self.model.t, target)

                # Log metrics to W&B
                if self.use_wandb:
                    wandb.log({"training_loss": loss.item()})

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Increment batch count
                batch_count += 1

                # Force the model to re-calculate the embeddings every n batches
                if batch_count % self.embedding_update_interval == 0 and self.train_label_encoder:
                    self.model.clear_label_embeddings_cache()

                # Run validation and log progress every n batches
                if batch_count % self.validation_frequency == 0:

                    ####### VALIDATION LOOP #######
                    self.logger.info("Running validation...")

                    val_metrics = self.evaluate(
                        data_loader=val_loader, testing=True)

                    self.logger.info(val_metrics)
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs}, Batch {batch_count}, Training Loss: {loss.item()}, Validation Loss: {val_metrics['avg_loss']}")

                    if self.use_wandb:
                        wandb.log({"validation_loss": val_metrics['avg_loss']})

                    # Save the model if it has the best validation loss so far
                    if val_metrics['avg_loss'] < best_val_loss:
                        self.logger.info(
                            f"New best validation loss: {val_metrics['avg_loss']}. Saving model...")
                        best_val_loss = val_metrics['avg_loss']

                        # Save model to OUTPUT_MODEL_DIR. Create path if it doesn't exist.
                        if not os.path.exists(self.output_model_dir):
                            os.makedirs(self.output_model_dir)

                        model_name = self.run_name if self.run_name else "best_ProTCL.pt"
                        model_path = os.path.join(
                            self.output_model_dir, f"{self.timestamp}_{model_name}.pt")
                        torch.save(self.model.state_dict(), model_path)
                        self.logger.info(f"Saved model to {model_path}.")

                        if self.use_wandb:
                            wandb.save(f"{self.timestamp}_best_ProTCL.pt")
