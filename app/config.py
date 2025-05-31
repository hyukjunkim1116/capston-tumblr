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
logging.getLogger("transformers").setLevel(logging.WARNING)

# Global variables for module availability
MODULES_LOADED = False
DAMAGE_CATEGORIES = {}
CACHE_DIR = Path("./cache")


def initialize_modules():
    """모듈 초기화 및 로드"""
    global MODULES_LOADED, DAMAGE_CATEGORIES, CACHE_DIR

    try:
        # 새로운 분석 엔진 사용 (FAISS 불필요)
        # analysis_engine에서는 Pandas로 직접 Excel 처리

        # 기본 설정 값들
        DAMAGE_CATEGORIES = {
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
            ]
        }
        CACHE_DIR = Path("./cache")

        MODULES_LOADED = True
        logger.info("모든 모듈이 성공적으로 로드되었습니다")

        # 더미 함수들 (기존 호환성용)
        @st.cache_resource
        def create_faiss_vector_store():
            logger.info("FAISS 제거됨 - Pandas 직접 사용")
            return None

        def analyze_building_damage(*args, **kwargs):
            return {"message": "새로운 분석 엔진을 사용합니다."}

        return {
            "analyze_building_damage": analyze_building_damage,
            "create_faiss_vector_store": create_faiss_vector_store,
        }

    except Exception as e:
        st.error(f"모듈 로딩 오류: {e}")
        st.error("필요한 모듈을 찾을 수 없습니다. 개발자에게 문의하세요.")

        MODULES_LOADED = False
        DAMAGE_CATEGORIES = {}
        CACHE_DIR = Path("./cache")

        logger.error(f"모듈 로딩 실패: {e}")

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
        "vector_store_available": False,  # FAISS 제거
        "damage_categories": DAMAGE_CATEGORIES,
        "cache_dir": CACHE_DIR,
    }
