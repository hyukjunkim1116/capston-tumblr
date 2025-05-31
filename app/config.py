"""
애플리케이션 설정 및 초기화
"""

import streamlit as st
import os
import logging
from pathlib import Path

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce verbosity of external libraries
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Global variables for module availability
MODULES_LOADED = False
VECTOR_STORE_AVAILABLE = False
DAMAGE_CATEGORIES = {}
CACHE_DIR = Path("./cache")


def initialize_modules():
    """모듈 초기화 및 로드"""
    global MODULES_LOADED, VECTOR_STORE_AVAILABLE, DAMAGE_CATEGORIES, CACHE_DIR

    try:
        from analysis.integration import analyze_building_damage
        from utils.config import (
            DAMAGE_CATEGORIES as CONFIG_DAMAGE_CATEGORIES,
            CACHE_DIR as CONFIG_CACHE_DIR,
        )
        from faiss.vector_store import create_faiss_vector_store

        VECTOR_STORE_AVAILABLE = True
        MODULES_LOADED = True
        DAMAGE_CATEGORIES = CONFIG_DAMAGE_CATEGORIES
        CACHE_DIR = CONFIG_CACHE_DIR

        logger.info("✅ 모든 모듈이 성공적으로 로드되었습니다")

        return {
            "analyze_building_damage": analyze_building_damage,
            "create_faiss_vector_store": create_faiss_vector_store,
        }

    except ImportError as e:
        st.error(f"모듈 로딩 오류: {e}")
        st.error("필요한 모듈을 찾을 수 없습니다. 개발자에게 문의하세요.")

        MODULES_LOADED = False
        VECTOR_STORE_AVAILABLE = False
        DAMAGE_CATEGORIES = {}
        CACHE_DIR = Path("./cache")

        logger.error(f"❌ 모듈 로딩 실패: {e}")

        # Fallback functions
        @st.cache_resource
        def create_faiss_vector_store():
            return None

        def analyze_building_damage(*args, **kwargs):
            return {"error": "분석 모듈을 사용할 수 없습니다."}

        return {
            "analyze_building_damage": analyze_building_damage,
            "create_faiss_vector_store": create_faiss_vector_store,
        }


def setup_directories():
    """필요한 디렉토리 생성"""
    upload_dir = CACHE_DIR / "uploads"
    results_dir = CACHE_DIR / "results"

    for directory in [upload_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return upload_dir, results_dir


def get_app_config():
    """애플리케이션 설정 반환"""
    return {
        "modules_loaded": MODULES_LOADED,
        "vector_store_available": VECTOR_STORE_AVAILABLE,
        "damage_categories": DAMAGE_CATEGORIES,
        "cache_dir": CACHE_DIR,
    }
