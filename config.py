"""
Configuration settings for Building Damage Analysis LLM System
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Learning data paths
LEARNING_TEXTS_PATH = BASE_DIR / "learning_texts.xlsx"
LEARNING_PICTURES_PATH = BASE_DIR / "learning_pictures"

# Model configurations
MODEL_CONFIG = {
    "vision_encoder": {
        "model_name": "openai/clip-vit-large-patch14",
        "image_size": 1024,
        "embedding_dim": 768,
    },
    "language_model": {
        "model_name": "microsoft/DialoGPT-medium",
        "max_length": 512,
        "temperature": 0.7,
    },
    "multimodal": {"projection_dim": 512, "hidden_dim": 1024},
}

# Training configurations
TRAINING_CONFIG = {
    "batch_size": 16,  # Reduced for local training
    "learning_rate": 3e-5,
    "num_epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "save_steps": 100,
    "eval_steps": 50,
}

# Data processing configurations
DATA_CONFIG = {
    "image_extensions": [".jpg", ".jpeg", ".png", ".webp", ".avif"],
    "max_image_size": (224, 224),
    "augmentation": {
        "rotation_limit": 15,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "blur_limit": 3,
    },
}

# Damage classification categories
DAMAGE_CATEGORIES = {
    "severity_levels": {
        1: "경미한 손상 (Minor damage)",
        2: "보통 손상 (Moderate damage)",
        3: "심각한 손상 (Severe damage)",
        4: "매우 심각한 손상 (Critical damage)",
        5: "완전 파괴 (Complete destruction)",
    },
    "damage_types": [
        "균열 (Cracks)",
        "수해 (Water damage)",
        "화재 손상 (Fire damage)",
        "지붕 손상 (Roof damage)",
        "창문/문 손상 (Window/Door damage)",
        "기초 침하 (Foundation settlement)",
        "구조적 변형 (Structural deformation)",
        "외벽 손상 (Facade damage)",
        "전기/기계 시설 손상 (Electrical/Mechanical damage)",
    ],
    "affected_areas": [
        "외벽 (Exterior walls)",
        "지붕 (Roof)",
        "기초 (Foundation)",
        "창문 (Windows)",
        "문 (Doors)",
        "발코니 (Balcony)",
        "계단 (Stairs)",
        "기타 (Others)",
    ],
}

# Streamlit configurations
STREAMLIT_CONFIG = {
    "page_title": "건물 피해 분석 시스템",
    "page_icon": "🏢",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".webp"],
    "default_port": 8501,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {"handlers": ["default", "file"], "level": "DEBUG", "propagate": False}
    },
}

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)
