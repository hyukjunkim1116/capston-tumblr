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


# 환경별 자동 설정 - 모든 환경 동일하게 설정
ENVIRONMENT = detect_environment()
IS_DEPLOYMENT = ENVIRONMENT != "local"

# 모든 환경에서 동일한 고성능 설정 사용
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
logger.info("🎯 설정: 모든 환경에서 커스텀 YOLOv8 모델 강제 사용")

# Global variables for module availability
MODULES_LOADED = False
DAMAGE_CATEGORIES = {}
CACHE_DIR = Path("./cache")
MODEL_CACHE_DIR = CACHE_DIR / "models"

# 커스텀 모델 경로 정의 (우선순위 순서)
CUSTOM_YOLO_PATHS = [
    Path("train/models/custom_yolo_damage.pt"),  # 1순위: 로컬 훈련된 모델
    Path("custom_yolo_damage.pt"),  # 2순위: 루트 디렉토리
    MODEL_CACHE_DIR / "custom_yolo_damage.pt",  # 3순위: 캐시 디렉토리
]

CUSTOM_CLIP_PATHS = [
    Path("train/models/clip_finetuned.pt"),  # 1순위: 로컬 훈련된 모델
    Path("clip_finetuned.pt"),  # 2순위: 루트 디렉토리
    MODEL_CACHE_DIR / "clip_finetuned.pt",  # 3순위: 캐시 디렉토리
]

# 외부 모델 저장소 URL (배포 환경에서 커스텀 모델 다운로드용)
EXTERNAL_MODEL_URLS = {
    # GitHub Releases (권장방법)
    "custom_yolo_github": os.getenv("CUSTOM_YOLO_URL", ""),
    # CLIP 파인튜닝 모델
    "clip_finetuned_github": os.getenv("CUSTOM_CLIP_URL", ""),
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
    """모든 환경에서 커스텀 모델 강제 사용 - 기본 모델 사용 금지"""
    logger.info("🚀 커스텀 모델 강제 초기화 시작")

    models = {}

    # YOLOv8 커스텀 모델 강제 로딩
    try:
        from ultralytics import YOLO

        yolo_model_loaded = False

        # 우선순위 순서로 커스텀 YOLO 모델 찾기
        for model_path in CUSTOM_YOLO_PATHS:
            if model_path.exists():
                models["yolo"] = YOLO(str(model_path))
                logger.info(f"✅ YOLOv8 커스텀 모델 로드 완료: {model_path}")
                yolo_model_loaded = True
                break

        # 로컬에 커스텀 모델이 없고 배포환경인 경우 다운로드 시도
        if not yolo_model_loaded and IS_DEPLOYMENT:
            custom_model_url = EXTERNAL_MODEL_URLS.get("custom_yolo_github")
            if custom_model_url:
                target_path = MODEL_CACHE_DIR / "custom_yolo_damage.pt"
                logger.info(
                    f"📥 배포환경에서 커스텀 YOLO 모델 다운로드 시도: {custom_model_url}"
                )

                if download_model_from_url(custom_model_url, target_path):
                    models["yolo"] = YOLO(str(target_path))
                    logger.info("✅ YOLOv8 다운로드된 커스텀 모델 로드 완료")
                    yolo_model_loaded = True

        # 커스텀 모델을 찾을 수 없는 경우 오류 발생
        if not yolo_model_loaded:
            error_msg = """
❌ 커스텀 YOLOv8 모델을 찾을 수 없습니다!

다음 위치 중 하나에 custom_yolo_damage.pt 파일이 있어야 합니다:
1. train/models/custom_yolo_damage.pt (권장)
2. custom_yolo_damage.pt (루트 디렉토리)
3. cache/models/custom_yolo_damage.pt

배포환경인 경우 CUSTOM_YOLO_URL 환경변수를 설정하세요.
"""
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    except Exception as e:
        logger.error(f"❌ YOLOv8 커스텀 모델 로드 실패: {e}")
        raise e

    # CLIP 모델 로딩 (커스텀 우선, 없으면 기본 모델)
    try:
        import clip
        import torch

        device = DEVICE
        clip_model_loaded = False

        # 우선순위 순서로 커스텀 CLIP 모델 찾기
        for model_path in CUSTOM_CLIP_PATHS:
            if model_path.exists():
                try:
                    model, preprocess = clip.load(str(model_path), device=device)
                    models["clip"] = (model, preprocess)
                    logger.info(f"✅ CLIP 커스텀 모델 로드 완료: {model_path}")
                    clip_model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"⚠️ CLIP 커스텀 모델 로드 실패 ({model_path}): {e}")
                    continue

        # 배포환경에서 CLIP 커스텀 모델 다운로드 시도
        if not clip_model_loaded and IS_DEPLOYMENT:
            custom_clip_url = EXTERNAL_MODEL_URLS.get("clip_finetuned_github")
            if custom_clip_url:
                target_path = MODEL_CACHE_DIR / "clip_finetuned.pt"
                logger.info(
                    f"📥 배포환경에서 커스텀 CLIP 모델 다운로드 시도: {custom_clip_url}"
                )

                if download_model_from_url(custom_clip_url, target_path):
                    try:
                        model, preprocess = clip.load(str(target_path), device=device)
                        models["clip"] = (model, preprocess)
                        logger.info("✅ CLIP 다운로드된 커스텀 모델 로드 완료")
                        clip_model_loaded = True
                    except Exception as e:
                        logger.warning(f"⚠️ 다운로드된 CLIP 모델 로드 실패: {e}")

        # 커스텀 CLIP 모델이 없으면 기본 모델 사용 (CLIP은 허용)
        if not clip_model_loaded:
            logger.info("📝 커스텀 CLIP 모델 없음, 기본 ViT-B/32 모델 사용")
            model, preprocess = clip.load("ViT-B/32", device=device)
            models["clip"] = (model, preprocess)
            logger.info("✅ CLIP 기본 모델 로드 완료")

    except Exception as e:
        logger.warning(f"⚠️ CLIP 모델 로드 실패: {e}")
        models["clip"] = None

    # OpenAI 모델 설정 (변경 없음)
    try:
        models["openai"] = True  # OpenAI는 API 기반이므로 True로 설정
        logger.info("✅ OpenAI 모델 설정 완료")
    except Exception as e:
        logger.warning(f"⚠️ OpenAI 설정 실패: {e}")
        models["openai"] = False

    logger.info(
        f"🎯 모델 초기화 완료 - YOLOv8: ✅ (커스텀), CLIP: {'✅' if models.get('clip') else '❌'}, OpenAI: {'✅' if models.get('openai') else '❌'}"
    )

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
