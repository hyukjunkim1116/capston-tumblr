"""
Data loader for building damage analysis training data
"""

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re
import torch.nn as nn

from config import (
    LEARNING_TEXTS_PATH,
    LEARNING_PICTURES_PATH,
    DATA_CONFIG,
    TRAINING_CONFIG,
    DAMAGE_CATEGORIES,
)

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """Custom collate function to handle variable length texts"""
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    damage_infos = [item["damage_info"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "image": images,
        "text": texts,
        "damage_info": damage_infos,
        "image_id": image_ids,
    }


class BuildingDamageDataset(Dataset):
    """Dataset class for building damage analysis"""

    def __init__(
        self,
        texts_path: Path = LEARNING_TEXTS_PATH,
        images_path: Path = LEARNING_PICTURES_PATH,
        transform: Optional[A.Compose] = None,
        is_training: bool = True,
    ):
        self.texts_path = texts_path
        self.images_path = images_path
        self.transform = transform
        self.is_training = is_training

        # Load and process data
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} samples")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and match text and image data"""
        try:
            # Load Excel file
            df = pd.read_excel(self.texts_path)
            logger.info(f"Loaded Excel with columns: {df.columns.tolist()}")

            data_samples = []

            # Process each row in the Excel file
            for idx, row in df.iterrows():
                # Extract image ID from the first column (assuming format like "1.*")
                first_col = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""
                image_id = self._extract_image_id(first_col)

                if image_id is None:
                    continue

                # Find corresponding image file
                image_path = self._find_image_file(image_id)
                if image_path is None:
                    logger.warning(f"No image found for ID: {image_id}")
                    continue

                # Extract text content (assuming it's in subsequent columns)
                text_content = self._extract_text_content(row)

                # Parse damage information from text
                damage_info = self._parse_damage_info(text_content)

                sample = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "text_content": text_content,
                    "damage_info": damage_info,
                }

                data_samples.append(sample)

            return data_samples

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []

    def _extract_image_id(self, text: str) -> Optional[str]:
        """Extract image ID from text (e.g., '1.*' -> '1')"""
        # Look for patterns like "1.*", "2.*", etc.
        match = re.search(r"^(\d+)\.", text)
        if match:
            return match.group(1)

        # Also try to extract just numbers
        match = re.search(r"^(\d+)", text)
        if match:
            return match.group(1)

        return None

    def _find_image_file(self, image_id: str) -> Optional[Path]:
        """Find image file with given ID"""
        for ext in DATA_CONFIG["image_extensions"]:
            image_path = self.images_path / f"{image_id}{ext}"
            if image_path.exists():
                return image_path
        return None

    def _extract_text_content(self, row: pd.Series) -> str:
        """Extract meaningful text content from row"""
        text_parts = []

        # Skip the first column (image ID) and collect non-null text
        for col_idx in range(1, len(row)):
            value = row.iloc[col_idx]
            if pd.notna(value) and str(value).strip():
                text_parts.append(str(value).strip())

        return " ".join(text_parts)

    def _parse_damage_info(self, text: str) -> Dict[str, Any]:
        """Parse damage information from text content"""
        damage_info = {
            "severity_level": 1,  # Default to minor damage
            "damage_types": [],
            "affected_areas": [],
            "confidence_level": 0.5,
            "detailed_findings": text,
        }

        text_lower = text.lower()

        # Detect severity level based on keywords
        severity_keywords = {
            5: ["완전", "파괴", "붕괴", "전면", "complete", "destruction", "collapse"],
            4: ["매우", "심각", "위험", "critical", "severe", "dangerous"],
            3: ["심각", "큰", "major", "serious", "significant"],
            2: ["보통", "중간", "moderate", "medium"],
            1: ["경미", "작은", "minor", "small", "slight"],
        }

        for level, keywords in severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                damage_info["severity_level"] = level
                break

        # Detect damage types
        damage_type_keywords = {
            "균열 (Cracks)": ["균열", "갈라짐", "crack", "split"],
            "수해 (Water damage)": ["물", "침수", "습기", "water", "flood", "moisture"],
            "화재 손상 (Fire damage)": ["화재", "불", "탄", "fire", "burn", "smoke"],
            "지붕 손상 (Roof damage)": ["지붕", "기와", "roof", "tile"],
            "창문/문 손상 (Window/Door damage)": ["창문", "문", "window", "door"],
            "기초 침하 (Foundation settlement)": [
                "기초",
                "침하",
                "foundation",
                "settlement",
            ],
            "구조적 변형 (Structural deformation)": [
                "구조",
                "변형",
                "structural",
                "deformation",
            ],
            "외벽 손상 (Facade damage)": ["외벽", "벽", "facade", "wall"],
            "전기/기계 시설 손상 (Electrical/Mechanical damage)": [
                "전기",
                "기계",
                "electrical",
                "mechanical",
            ],
        }

        for damage_type, keywords in damage_type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                damage_info["damage_types"].append(damage_type)

        # Detect affected areas
        area_keywords = {
            "외벽 (Exterior walls)": ["외벽", "벽", "wall", "exterior"],
            "지붕 (Roof)": ["지붕", "roof"],
            "기초 (Foundation)": ["기초", "foundation"],
            "창문 (Windows)": ["창문", "window"],
            "문 (Doors)": ["문", "door"],
            "발코니 (Balcony)": ["발코니", "balcony"],
            "계단 (Stairs)": ["계단", "stairs"],
            "기타 (Others)": ["기타", "others"],
        }

        for area, keywords in area_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                damage_info["affected_areas"].append(area)

        return damage_info

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]

        # Load and preprocess image
        image = self._load_image(sample["image_path"])

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor if no transform is provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image,
            "text": sample["text_content"],
            "damage_info": sample["damage_info"],
            "image_id": sample["image_id"],
        }

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess image"""
        try:
            # Load image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                # Try with PIL for other formats
                pil_image = Image.open(image_path).convert("RGB")
                image = np.array(pil_image)
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image
            target_size = DATA_CONFIG["max_image_size"]
            image = cv2.resize(image, target_size)

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return np.zeros((*DATA_CONFIG["max_image_size"], 3), dtype=np.uint8)


def get_transforms(is_training: bool = True) -> A.Compose:
    """Get image transformations for training or validation"""

    if is_training:
        # Training transforms with augmentation
        transform = A.Compose(
            [
                A.Rotate(limit=DATA_CONFIG["augmentation"]["rotation_limit"], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=DATA_CONFIG["augmentation"]["brightness_limit"],
                    contrast_limit=DATA_CONFIG["augmentation"]["contrast_limit"],
                    p=0.5,
                ),
                A.Blur(blur_limit=DATA_CONFIG["augmentation"]["blur_limit"], p=0.3),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        # Validation transforms without augmentation
        transform = A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    return transform


def create_data_loaders(
    train_split: float = 0.8, batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""

    if batch_size is None:
        batch_size = TRAINING_CONFIG["batch_size"]

    # Create full dataset
    full_dataset = BuildingDamageDataset(transform=get_transforms(is_training=True))

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Update transforms for validation dataset
    val_dataset.dataset.transform = get_transforms(is_training=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    logger.info(
        f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}"
    )

    return train_loader, val_loader


def test_data_loader():
    """Test function to verify data loading works correctly"""
    try:
        dataset = BuildingDamageDataset()
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Text: {sample['text'][:100]}...")
            print(f"Damage info: {sample['damage_info']}")

        # Test data loaders
        train_loader, val_loader = create_data_loaders()
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test one batch
        for batch in train_loader:
            print(f"Batch image shape: {batch['image'].shape}")
            print(f"Batch text length: {len(batch['text'])}")
            break

    except Exception as e:
        logger.error(f"Error in test_data_loader: {e}")
        raise


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    test_data_loader()
