"""
애플리케이션 설정 및 초기화 - 환경별 자동 최적화
"""

import streamlit as st
import os
import logging
import platform
import sys
from pathlib import Path


# 환경 감지
def detect_environment():
    """실행 환경 자동 감지"""
    # Streamlit Cloud 감지
    if os.getenv("STREAMLIT_CLOUD") or "streamlit" in str(sys.path):
        return "streamlit_cloud"

    # Heroku 감지
    if os.getenv("DYNO"):
        return "heroku"

    # Railway 감지
    if os.getenv("RAILWAY_STATIC_URL"):
        return "railway"

    # Docker 감지
    if os.path.exists("/.dockerenv"):
        return "docker"

    # 로컬 개발 환경
    return "local"


# 환경별 자동 설정
ENVIRONMENT = detect_environment()
IS_DEPLOYMENT = ENVIRONMENT != "local"

# 환경별 최적화 설정
# 배포 환경도 로컬과 동일한 설정 사용 (느려도 정확도 우선)
LOG_LEVEL = logging.INFO
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") != "none" else "cpu"
BATCH_SIZE = 4
MAX_IMAGE_SIZE = 2048

# Setup logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Reduce verbosity of external libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger.info(f"🌍 환경 감지: {ENVIRONMENT} ({'배포' if IS_DEPLOYMENT else '로컬'})")
logger.info(f"⚙️ 디바이스: {DEVICE}, 배치크기: {BATCH_SIZE}")

# Global variables for module availability
MODULES_LOADED = False
DAMAGE_CATEGORIES = {}
CACHE_DIR = Path("./cache")
MODEL_CACHE_DIR = CACHE_DIR / "models"

# 모델 다운로드 URL 설정 (배포 환경용)
MODEL_URLS = {
    "yolo_custom": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "clip_base": "ViT-B/32",  # CLIP 기본 모델
}

# 외부 모델 저장소 URL (배포 환경에서 커스텀 모델 다운로드용)
# 실제 사용 시 아래 URL들을 실제 모델 URL로 변경하세요
EXTERNAL_MODEL_URLS = {
    # Google Drive 공유 링크 (다운로드 직링크로 변환 필요)
    "custom_yolo_gdrive": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID",
    # GitHub Releases (권장방법)
    "custom_yolo_github": "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/custom_yolo_damage.pt",
    # Hugging Face Hub
    "custom_yolo_hf": "https://huggingface.co/YOUR_USERNAME/building-damage-yolo/resolve/main/custom_yolo_damage.pt",
    # CLIP 파인튜닝 모델 (추후 사용)
    "clip_finetuned_hf": "https://huggingface.co/YOUR_USERNAME/building-damage-clip/resolve/main/clip_finetuned.pt",
}


def download_model_from_url(url: str, target_path: Path) -> bool:
    """외부 URL에서 모델 다운로드"""
    try:
        import requests

        logger.info(f"📥 모델 다운로드 시작: {url}")

        # 디렉토리 생성
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 헤더 설정 (일부 서비스에서 User-Agent 필요)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, stream=True, headers=headers, timeout=300)
        response.raise_for_status()

        # 파일 크기 확인
        total_size = int(response.headers.get("content-length", 0))

        with open(target_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # 진행률 로깅 (큰 파일의 경우)
                    if total_size > 0 and downloaded % (1024 * 1024) == 0:  # 1MB마다
                        progress = (downloaded / total_size) * 100
                        logger.info(f"📥 다운로드 진행률: {progress:.1f}%")

        # 파일 크기 검증
        if target_path.stat().st_size < 1024:  # 1KB 미만이면 오류
            logger.error("❌ 다운로드된 파일이 너무 작습니다")
            target_path.unlink()
            return False

        logger.info(
            f"✅ 모델 다운로드 완료: {target_path} ({target_path.stat().st_size / 1024 / 1024:.1f}MB)"
        )
        return True

    except Exception as e:
        logger.error(f"❌ 모델 다운로드 실패: {e}")
        if target_path.exists():
            target_path.unlink()  # 실패한 파일 삭제
        return False


@st.cache_resource
def initialize_optimized_models():
    """모든 환경에서 고정확도 모델 초기화 - 배포환경 커스텀 모델 지원"""
    logger.info("🚀 고정확도 모델 초기화 시작")

    models = {}

    try:
        # YOLOv8 모델 로딩 (배포환경 커스텀 모델 지원)
        from ultralytics import YOLO

        # 1순위: 로컬 커스텀 모델
        local_custom_path = Path("train/models/custom_yolo_damage.pt")

        # 2순위: 배포환경 캐시된 커스텀 모델
        cached_custom_path = MODEL_CACHE_DIR / "custom_yolo_damage.pt"

        if local_custom_path.exists():
            # 로컬 커스텀 모델 사용
            models["yolo"] = YOLO(str(local_custom_path))
            logger.info("✅ YOLOv8 로컬 커스텀 모델 로드 완료")

        elif cached_custom_path.exists():
            # 캐시된 커스텀 모델 사용
            models["yolo"] = YOLO(str(cached_custom_path))
            logger.info("✅ YOLOv8 캐시된 커스텀 모델 로드 완료")

        elif IS_DEPLOYMENT:
            # 배포환경에서 커스텀 모델 다운로드 시도
            logger.info("🌐 배포환경 감지 - 커스텀 모델 다운로드 시도")

            # 환경변수에서 모델 URL 확인
            custom_model_url = os.getenv("CUSTOM_YOLO_URL")

            if custom_model_url:
                logger.info(f"📥 환경변수에서 커스텀 모델 URL 발견: {custom_model_url}")
                if download_model_from_url(custom_model_url, cached_custom_path):
                    models["yolo"] = YOLO(str(cached_custom_path))
                    logger.info("✅ YOLOv8 다운로드된 커스텀 모델 로드 완료")
                else:
                    logger.warning("⚠️ 커스텀 모델 다운로드 실패, 기본 모델 사용")
                    models["yolo"] = YOLO("yolov8n.pt")
                    logger.info("✅ YOLOv8 기본 모델 로드 완료")
            else:
                logger.info("📝 CUSTOM_YOLO_URL 환경변수 없음, 기본 모델 사용")
                models["yolo"] = YOLO("yolov8n.pt")
                logger.info("✅ YOLOv8 기본 모델 로드 완료")
        else:
            # 로컬환경에서 커스텀 모델 없으면 기본 모델
            models["yolo"] = YOLO("yolov8n.pt")
            logger.info("✅ YOLOv8 기본 모델 로드 완료")

    except Exception as e:
        logger.warning(f"⚠️ YOLOv8 로드 실패: {e}")
        models["yolo"] = None

    try:
        # CLIP 모델 로딩 (배포환경 커스텀 모델 지원)
        import clip
        import torch

        device = DEVICE

        # 1순위: 로컬 커스텀 모델
        local_clip_path = Path("train/models/clip_finetuned.pt")

        # 2순위: 배포환경 캐시된 커스텀 모델
        cached_clip_path = MODEL_CACHE_DIR / "clip_finetuned.pt"

        if local_clip_path.exists():
            # 로컬 커스텀 모델 사용
            model, preprocess = clip.load(str(local_clip_path), device=device)
            logger.info("✅ CLIP 로컬 커스텀 모델 로드 완료")

        elif cached_clip_path.exists():
            # 캐시된 커스텀 모델 사용
            model, preprocess = clip.load(str(cached_clip_path), device=device)
            logger.info("✅ CLIP 캐시된 커스텀 모델 로드 완료")

        elif IS_DEPLOYMENT:
            # 배포환경에서 CLIP 커스텀 모델 다운로드 시도
            custom_clip_url = os.getenv("CUSTOM_CLIP_URL")

            if custom_clip_url:
                logger.info(f"📥 CLIP 커스텀 모델 다운로드: {custom_clip_url}")
                if download_model_from_url(custom_clip_url, cached_clip_path):
                    model, preprocess = clip.load(str(cached_clip_path), device=device)
                    logger.info("✅ CLIP 다운로드된 커스텀 모델 로드 완료")
                else:
                    logger.warning("⚠️ CLIP 커스텀 모델 다운로드 실패, 기본 모델 사용")
                    model, preprocess = clip.load("ViT-B/32", device=device)
                    logger.info("✅ CLIP 기본 모델 로드 완료")
            else:
                model, preprocess = clip.load("ViT-B/32", device=device)
                logger.info("✅ CLIP 기본 모델 로드 완료")
        else:
            # 로컬환경에서 커스텀 모델 없으면 기본 모델
            model, preprocess = clip.load("ViT-B/32", device=device)
            logger.info("✅ CLIP 기본 모델 로드 완료")

        models["clip"] = {"model": model, "preprocess": preprocess}

    except Exception as e:
        logger.warning(f"⚠️ CLIP 로드 실패: {e}")
        models["clip"] = None

    try:
        # OpenAI API 설정
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            models["openai"] = OpenAI(api_key=api_key)
            logger.info("✅ OpenAI API 연결 완료")
        else:
            logger.warning("⚠️ OpenAI API 키 없음")
            models["openai"] = None

    except Exception as e:
        logger.warning(f"⚠️ OpenAI API 초기화 실패: {e}")
        models["openai"] = None

    logger.info(f"🎯 고정확도 모델 초기화 완료 - 환경: {ENVIRONMENT}")
    return models


def initialize_modules():
    """모듈 초기화 및 로드"""
    global MODULES_LOADED, DAMAGE_CATEGORIES, CACHE_DIR

    try:
        # 환경별 최적화된 모델 로드
        models = initialize_optimized_models()

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
        logger.info("✅ 모든 모듈이 성공적으로 로드되었습니다")

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
            "models": models,
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
            "models": {},
        }


def setup_directories():
    """필요한 디렉토리 생성"""
    upload_dir = CACHE_DIR / "uploads"
    results_dir = CACHE_DIR / "results"

    # 모델 캐시 디렉토리도 생성
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for directory in [upload_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return upload_dir, results_dir


def get_app_config():
    """애플리케이션 설정 반환"""
    return {
        "environment": ENVIRONMENT,
        "is_deployment": IS_DEPLOYMENT,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "max_image_size": MAX_IMAGE_SIZE,
        "modules_loaded": MODULES_LOADED,
        "vector_store_available": False,  # FAISS 제거
        "damage_categories": DAMAGE_CATEGORIES,
        "cache_dir": CACHE_DIR,
        "model_cache_dir": MODEL_CACHE_DIR,
    }


def optimize_for_environment():
    """환경별 최적화 설정 적용"""
    # 모든 환경에서 동일한 성능 설정 (배포 환경 최적화 제거)
    if True:  # 항상 실행 (기존 IS_DEPLOYMENT 조건 제거)
        # 기본 환경 변수 설정 (과도한 최적화 제거)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 병렬 처리 활성화
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
        # 메모리 제한 제거
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # 제거
        # os.environ["OMP_NUM_THREADS"] = "2"  # 제거
        # os.environ["MKL_NUM_THREADS"] = "2"  # 제거

        # CPU 스레드 제한 해제
        import torch

        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(8)  # 더 많은 스레드 사용

    logger.info(f"🎯 환경별 최적화 완료: {ENVIRONMENT}")
