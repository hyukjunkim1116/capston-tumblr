"""
Multimodal models for building damage analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel,
    CLIPProcessor,
    AutoTokenizer,
    AutoModel,
)
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from config import MODEL_CONFIG, DAMAGE_CATEGORIES

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """Vision encoder using CLIP"""

    def __init__(self, model_name: str = None):
        super().__init__()

        if model_name is None:
            model_name = MODEL_CONFIG["vision_encoder"]["model_name"]

        self.model_name = model_name
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)

        # Use fast processor to avoid warnings
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        except Exception:
            # Fallback to slow processor if fast is not available
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logger.warning(f"Using slow processor for {model_name}")

        # Freeze vision model initially
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Get embedding dimension
        self.embedding_dim = self.vision_model.config.hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Image embeddings [B, embedding_dim]
        """
        # Process images through CLIP vision model
        vision_outputs = self.vision_model(pixel_values=images)

        # Get pooled output (CLS token equivalent)
        image_embeddings = vision_outputs.pooler_output

        return image_embeddings

    def unfreeze_layers(self, num_layers: int = 2):
        """Unfreeze last few layers for fine-tuning"""
        layers = list(self.vision_model.vision_model.encoder.layers)
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


class TextEncoder(nn.Module):
    """Text encoder for damage descriptions"""

    def __init__(self, model_name: str = None):
        super().__init__()

        if model_name is None:
            model_name = MODEL_CONFIG["language_model"]["model_name"]

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try to load with safetensors first
        try:
            self.text_model = AutoModel.from_pretrained(
                model_name, use_safetensors=True
            )
        except Exception as e:
            logger.warning(f"Failed to load with safetensors: {e}")
            self.text_model = AutoModel.from_pretrained(model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embedding_dim = self.text_model.config.hidden_size

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Forward pass through text encoder

        Args:
            texts: List of text descriptions

        Returns:
            Text embeddings [B, embedding_dim]
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MODEL_CONFIG["language_model"]["max_length"],
            return_tensors="pt",
        )

        # Move to same device as model
        device = next(self.text_model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get text embeddings
        outputs = self.text_model(**encoded)

        # Use mean pooling over sequence length
        text_embeddings = outputs.last_hidden_state.mean(dim=1)

        return text_embeddings


class MultimodalProjection(nn.Module):
    """Projection layer to align vision and text embeddings"""

    def __init__(self, vision_dim: int, text_dim: int, projection_dim: int = None):
        super().__init__()

        if projection_dim is None:
            projection_dim = MODEL_CONFIG["multimodal"]["projection_dim"]

        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
        )

        self.projection_dim = projection_dim

    def forward(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project vision and text features to common space

        Args:
            vision_features: Vision embeddings [B, vision_dim]
            text_features: Text embeddings [B, text_dim]

        Returns:
            Projected vision and text features
        """
        projected_vision = self.vision_projection(vision_features)
        projected_text = self.text_projection(text_features)

        # L2 normalize
        projected_vision = F.normalize(projected_vision, p=2, dim=1)
        projected_text = F.normalize(projected_text, p=2, dim=1)

        return projected_vision, projected_text


class DamageClassifier(nn.Module):
    """Classifier for damage severity and types"""

    def __init__(self, input_dim: int):
        super().__init__()

        hidden_dim = MODEL_CONFIG["multimodal"]["hidden_dim"]

        # Severity classifier (1-5 scale)
        self.severity_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 5),  # 5 severity levels
        )

        # Damage type classifier (multi-label)
        num_damage_types = len(DAMAGE_CATEGORIES["damage_types"])
        self.damage_type_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_damage_types),
        )

        # Affected area classifier (multi-label)
        num_areas = len(DAMAGE_CATEGORIES["affected_areas"])
        self.area_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_areas),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify damage characteristics

        Args:
            features: Combined multimodal features [B, input_dim]

        Returns:
            Dictionary with classification outputs
        """
        severity_logits = self.severity_classifier(features)
        damage_type_logits = self.damage_type_classifier(features)
        area_logits = self.area_classifier(features)

        return {
            "severity": severity_logits,
            "damage_types": damage_type_logits,
            "affected_areas": area_logits,
        }


class BuildingDamageAnalysisModel(nn.Module):
    """Complete multimodal model for building damage analysis"""

    def __init__(
        self,
        vision_model_name: str = None,
        text_model_name: str = None,
        projection_dim: int = None,
    ):
        super().__init__()

        # Initialize encoders
        self.vision_encoder = VisionEncoder(vision_model_name)
        self.text_encoder = TextEncoder(text_model_name)

        # Initialize projection layer
        vision_dim = self.vision_encoder.embedding_dim
        text_dim = self.text_encoder.embedding_dim

        if projection_dim is None:
            projection_dim = MODEL_CONFIG["multimodal"]["projection_dim"]

        self.projection = MultimodalProjection(vision_dim, text_dim, projection_dim)

        # Initialize classifier
        # Combined features: projected_vision + projected_text
        combined_dim = projection_dim * 2
        self.classifier = DamageClassifier(combined_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim, num_heads=8, dropout=0.1, batch_first=True
        )

    def forward(
        self, images: torch.Tensor, texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model

        Args:
            images: Batch of images [B, C, H, W]
            texts: List of text descriptions

        Returns:
            Dictionary with all outputs
        """
        # Encode modalities
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(texts)

        # Project to common space
        projected_vision, projected_text = self.projection(
            vision_features, text_features
        )

        # Apply cross-modal attention
        # Use vision as query, text as key/value
        vision_attended, _ = self.cross_attention(
            projected_vision.unsqueeze(1),  # Add sequence dimension
            projected_text.unsqueeze(1),
            projected_text.unsqueeze(1),
        )
        vision_attended = vision_attended.squeeze(1)  # Remove sequence dimension

        # Combine features
        combined_features = torch.cat([vision_attended, projected_text], dim=1)

        # Classify damage
        classification_outputs = self.classifier(combined_features)

        # Add embeddings to output for analysis
        outputs = {
            **classification_outputs,
            "vision_embeddings": projected_vision,
            "text_embeddings": projected_text,
            "combined_features": combined_features,
        }

        return outputs

    def get_damage_predictions(
        self, images: torch.Tensor, texts: List[str], threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable damage predictions

        Args:
            images: Batch of images
            texts: List of text descriptions
            threshold: Threshold for multi-label classification

        Returns:
            List of prediction dictionaries
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, texts)

        predictions = []
        batch_size = images.size(0)

        for i in range(batch_size):
            # Severity prediction
            severity_probs = F.softmax(outputs["severity"][i], dim=0)
            severity_level = torch.argmax(severity_probs).item() + 1  # 1-5 scale
            severity_confidence = severity_probs.max().item()

            # Damage type predictions
            damage_type_probs = torch.sigmoid(outputs["damage_types"][i])
            damage_type_indices = (damage_type_probs > threshold).nonzero().flatten()
            predicted_damage_types = [
                DAMAGE_CATEGORIES["damage_types"][idx.item()]
                for idx in damage_type_indices
            ]

            # Affected area predictions
            area_probs = torch.sigmoid(outputs["affected_areas"][i])
            area_indices = (area_probs > threshold).nonzero().flatten()
            predicted_areas = [
                DAMAGE_CATEGORIES["affected_areas"][idx.item()] for idx in area_indices
            ]

            prediction = {
                "severity_level": severity_level,
                "severity_description": DAMAGE_CATEGORIES["severity_levels"][
                    severity_level
                ],
                "severity_confidence": severity_confidence,
                "damage_types": predicted_damage_types,
                "affected_areas": predicted_areas,
                "input_text": texts[i] if i < len(texts) else "",
            }

            predictions.append(prediction)

        return predictions

    def save_model(self, save_path: Path):
        """Save model state"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_state = {
            "model_state_dict": self.state_dict(),
            "model_config": MODEL_CONFIG,
            "damage_categories": DAMAGE_CATEGORIES,
        }

        torch.save(model_state, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: Path, device: str = "cpu"):
        """Load model from saved state"""
        try:
            # Try to load with weights_only=False for compatibility
            model_state = torch.load(load_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=False: {e}")
            # Fallback to default loading
            model_state = torch.load(load_path, map_location=device)

        model = cls()
        model.load_state_dict(model_state["model_state_dict"])
        model.to(device)

        logger.info(f"Model loaded from {load_path}")
        return model


def create_model(device: str = "cpu") -> BuildingDamageAnalysisModel:
    """Create and initialize the model"""
    model = BuildingDamageAnalysisModel()
    model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    import logging

    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(device)

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 1024, 1024).to(device)
    texts = ["건물 외벽에 균열이 발생했습니다.", "지붕에 심각한 손상이 있습니다."]

    outputs = model(images, texts)
    print(
        "Model outputs:",
        {k: v.shape if torch.is_tensor(v) else v for k, v in outputs.items()},
    )

    predictions = model.get_damage_predictions(images, texts)
    print("Predictions:", predictions)
