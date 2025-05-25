#!/usr/bin/env python3
"""
Simple training script for building damage analysis model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

# Import our modules
from data_loader import BuildingDamageDataset, create_data_loaders, custom_collate_fn
from models import BuildingDamageAnalysisModel, create_model
from config import TRAINING_CONFIG, MODELS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_train():
    """Simple training function"""

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODELS_DIR / f"training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create data loaders with smaller batch size for CPU
        batch_size = 4 if device == "cpu" else TRAINING_CONFIG["batch_size"]
        logger.info(f"Using batch size: {batch_size}")

        train_loader, val_loader = create_data_loaders(batch_size=batch_size)
        logger.info(
            f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}"
        )

        # Create model
        logger.info("Creating model...")
        model = create_model(device)

        # Setup optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )

        # Setup loss functions
        severity_criterion = nn.CrossEntropyLoss()
        damage_type_criterion = nn.BCEWithLogitsLoss()
        area_criterion = nn.BCEWithLogitsLoss()

        # Training parameters
        num_epochs = 3  # Reduced for quick training
        best_loss = float("inf")

        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move data to device
                    images = batch["image"].to(device)
                    texts = batch["text"]

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images, texts)

                    # Calculate losses (simplified)
                    # For now, just use a simple MSE loss on the combined features
                    # This is a placeholder - in real training you'd use proper labels
                    target = torch.randn_like(outputs["combined_features"])
                    loss = nn.MSELoss()(outputs["combined_features"], target)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), TRAINING_CONFIG["max_grad_norm"]
                    )

                    optimizer.step()

                    train_loss += loss.item()
                    train_batches += 1

                    # Update progress bar
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                    # Break after a few batches for quick demo
                    if batch_idx >= 5:
                        break

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue

            avg_train_loss = train_loss / max(train_batches, 1)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")

            # Validation phase (simplified)
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    try:
                        images = batch["image"].to(device)
                        texts = batch["text"]

                        outputs = model(images, texts)

                        # Simple validation loss
                        target = torch.randn_like(outputs["combined_features"])
                        loss = nn.MSELoss()(outputs["combined_features"], target)

                        val_loss += loss.item()
                        val_batches += 1

                        # Break after a few batches
                        if batch_idx >= 2:
                            break

                    except Exception as e:
                        logger.error(f"Error in validation batch {batch_idx}: {e}")
                        continue

            avg_val_loss = val_loss / max(val_batches, 1)
            logger.info(f"Average validation loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_path = output_dir / "best_model.pt"
                model.save_model(best_model_path)
                logger.info(f"Saved best model with validation loss: {best_loss:.4f}")

        # Save final model
        final_model_path = output_dir / "final_model.pt"
        model.save_model(final_model_path)

        # Save training info
        training_info = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "device": device,
            "best_loss": best_loss,
            "final_loss": avg_val_loss,
            "model_path": str(best_model_path),
            "timestamp": timestamp,
        }

        with open(output_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        logger.info(f"Training completed! Models saved to {output_dir}")
        logger.info(f"Best validation loss: {best_loss:.4f}")

        return output_dir

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    simple_train()
