"""
Training module for building damage analysis model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

from models import BuildingDamageAnalysisModel, create_model
from data_loader import create_data_loaders
from config import TRAINING_CONFIG, MODELS_DIR, DAMAGE_CATEGORIES

logger = logging.getLogger(__name__)


class DamageLoss(nn.Module):
    """Custom loss function for damage analysis"""

    def __init__(
        self,
        severity_weight: float = 1.0,
        damage_type_weight: float = 1.0,
        area_weight: float = 1.0,
    ):
        super().__init__()

        self.severity_weight = severity_weight
        self.damage_type_weight = damage_type_weight
        self.area_weight = area_weight

        # Loss functions
        self.severity_loss = nn.CrossEntropyLoss()
        self.damage_type_loss = nn.BCEWithLogitsLoss()
        self.area_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary with individual and total losses
        """
        # Severity loss (classification)
        severity_loss = self.severity_loss(predictions["severity"], targets["severity"])

        # Damage type loss (multi-label)
        damage_type_loss = self.damage_type_loss(
            predictions["damage_types"], targets["damage_types"]
        )

        # Area loss (multi-label)
        area_loss = self.area_loss(
            predictions["affected_areas"], targets["affected_areas"]
        )

        # Total weighted loss
        total_loss = (
            self.severity_weight * severity_loss
            + self.damage_type_weight * damage_type_loss
            + self.area_weight * area_loss
        )

        return {
            "total_loss": total_loss,
            "severity_loss": severity_loss,
            "damage_type_loss": damage_type_loss,
            "area_loss": area_loss,
        }


class DamageTrainer:
    """Trainer class for building damage analysis model"""

    def __init__(
        self,
        model: BuildingDamageAnalysisModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        config: Dict[str, Any] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        if config is None:
            config = TRAINING_CONFIG
        self.config = config

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Initialize loss function
        self.criterion = DamageLoss()

        # Mixed precision training
        self.scaler = GradScaler()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        # Create save directory
        self.save_dir = (
            MODELS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized. Save directory: {self.save_dir}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different learning rates for different components"""

        # Separate parameters for different components
        vision_params = list(self.model.vision_encoder.parameters())
        text_params = list(self.model.text_encoder.parameters())
        other_params = [
            p
            for name, p in self.model.named_parameters()
            if not any(comp in name for comp in ["vision_encoder", "text_encoder"])
        ]

        # Different learning rates
        base_lr = self.config["learning_rate"]
        param_groups = [
            {
                "params": vision_params,
                "lr": base_lr * 0.1,
            },  # Lower LR for pretrained vision
            {
                "params": text_params,
                "lr": base_lr * 0.1,
            },  # Lower LR for pretrained text
            {"params": other_params, "lr": base_lr},  # Full LR for new components
        ]

        optimizer = optim.AdamW(param_groups, weight_decay=self.config["weight_decay"])

        return optimizer

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""

        total_steps = len(self.train_loader) * self.config["num_epochs"]
        warmup_steps = self.config["warmup_steps"]

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.1, (total_steps - step) / (total_steps - warmup_steps))

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler

    def _prepare_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare target tensors from batch"""

        batch_size = len(batch["damage_info"])
        device = self.device

        # Initialize target tensors
        severity_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        damage_type_targets = torch.zeros(
            batch_size,
            len(DAMAGE_CATEGORIES["damage_types"]),
            dtype=torch.float,
            device=device,
        )
        area_targets = torch.zeros(
            batch_size,
            len(DAMAGE_CATEGORIES["affected_areas"]),
            dtype=torch.float,
            device=device,
        )

        # Fill targets from damage_info
        for i, damage_info in enumerate(batch["damage_info"]):
            # Severity (1-5 scale to 0-4 for CrossEntropyLoss)
            severity_targets[i] = damage_info["severity_level"] - 1

            # Damage types (multi-label)
            for damage_type in damage_info["damage_types"]:
                if damage_type in DAMAGE_CATEGORIES["damage_types"]:
                    idx = DAMAGE_CATEGORIES["damage_types"].index(damage_type)
                    damage_type_targets[i, idx] = 1.0

            # Affected areas (multi-label)
            for area in damage_info["affected_areas"]:
                if area in DAMAGE_CATEGORIES["affected_areas"]:
                    idx = DAMAGE_CATEGORIES["affected_areas"].index(area)
                    area_targets[i, idx] = 1.0

        return {
            "severity": severity_targets,
            "damage_types": damage_type_targets,
            "affected_areas": area_targets,
        }

    def _calculate_accuracy(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate accuracy metrics"""

        with torch.no_grad():
            # Severity accuracy
            severity_pred = torch.argmax(predictions["severity"], dim=1)
            severity_acc = (severity_pred == targets["severity"]).float().mean().item()

            # Damage type accuracy (multi-label)
            damage_type_pred = (
                torch.sigmoid(predictions["damage_types"]) > 0.5
            ).float()
            damage_type_acc = (
                (
                    (damage_type_pred == targets["damage_types"]).sum(dim=1)
                    == targets["damage_types"].size(1)
                )
                .float()
                .mean()
                .item()
            )

            # Area accuracy (multi-label)
            area_pred = (torch.sigmoid(predictions["affected_areas"]) > 0.5).float()
            area_acc = (
                (
                    (area_pred == targets["affected_areas"]).sum(dim=1)
                    == targets["affected_areas"].size(1)
                )
                .float()
                .mean()
                .item()
            )

            # Overall accuracy (average)
            overall_acc = (severity_acc + damage_type_acc + area_acc) / 3

        return {
            "overall": overall_acc,
            "severity": severity_acc,
            "damage_type": damage_type_acc,
            "area": area_acc,
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""

        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch["image"].to(self.device)
            texts = batch["text"]

            # Prepare targets
            targets = self._prepare_targets(batch)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                predictions = self.model(images, texts)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict["total_loss"]

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["max_grad_norm"]
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Calculate accuracy
            accuracy_dict = self._calculate_accuracy(predictions, targets)

            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy_dict["overall"]
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{accuracy_dict['overall']:.4f}",
                    "LR": f"{self.scheduler.get_last_lr()[0]:.6f}",
                }
            )

            # Save checkpoint periodically
            if (batch_idx + 1) % self.config["save_steps"] == 0:
                self._save_checkpoint(
                    f"checkpoint_epoch_{self.current_epoch}_step_{batch_idx}"
                )

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def validate(self) -> Dict[str, float]:
        """Validate the model"""

        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                images = batch["image"].to(self.device)
                texts = batch["text"]

                # Prepare targets
                targets = self._prepare_targets(batch)

                # Forward pass
                predictions = self.model(images, texts)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict["total_loss"]

                # Calculate accuracy
                accuracy_dict = self._calculate_accuracy(predictions, targets)

                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy_dict["overall"]
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def train(self) -> Dict[str, List[float]]:
        """Complete training loop"""

        logger.info(f"Starting training for {self.config['num_epochs']} epochs")

        for epoch in range(self.config["num_epochs"]):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update history
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["train_accuracy"].append(train_metrics["accuracy"])
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_best_model()
                logger.info(
                    f"New best model saved with validation loss: {self.best_val_loss:.4f}"
                )

            # Save checkpoint
            if (epoch + 1) % self.config["save_steps"] == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}")

        # Save final model and training history
        self._save_final_model()
        self._save_training_history()
        self._plot_training_curves()

        logger.info("Training completed!")
        return self.training_history

    def _save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "config": self.config,
        }

        checkpoint_path = self.save_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _save_best_model(self):
        """Save the best model"""
        self.model.save_model(self.save_dir / "best_model.pt")

    def _save_final_model(self):
        """Save the final model"""
        self.model.save_model(self.save_dir / "final_model.pt")

    def _save_training_history(self):
        """Save training history as JSON"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")

    def _plot_training_curves(self):
        """Plot and save training curves"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(self.training_history["train_loss"]) + 1)

        # Loss curves
        ax1.plot(epochs, self.training_history["train_loss"], "b-", label="Train Loss")
        ax1.plot(epochs, self.training_history["val_loss"], "r-", label="Val Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(
            epochs,
            self.training_history["train_accuracy"],
            "b-",
            label="Train Accuracy",
        )
        ax2.plot(
            epochs, self.training_history["val_accuracy"], "r-", label="Val Accuracy"
        )
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = self.save_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Training curves saved: {plot_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def train_model(device: str = None, config: Dict[str, Any] = None) -> DamageTrainer:
    """Main training function"""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if config is None:
        config = TRAINING_CONFIG

    logger.info(f"Training on device: {device}")

    # Create model
    model = create_model(device)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(batch_size=config["batch_size"])

    # Create trainer
    trainer = DamageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )

    # Start training
    training_history = trainer.train()

    return trainer


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Train the model
    trainer = train_model()
    print("Training completed!")
