#!/usr/bin/env python3
"""
건물 피해 분석 AI 모델 통합 훈련 스크립트
"""

import os
import sys
import time
import logging
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from app.yolo_trainer import train_custom_yolo
from app.clip_trainer import train_clip_finetuning

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def check_requirements():
    """필수 요구사항 체크"""
    logger.info("🔍 필수 요구사항 체크 중...")

    # 1. 데이터 폴더 체크
    required_paths = [
        "../datasets/learning_data/learning_pictures",
        "../datasets/learning_data/learning_texts.xlsx",
    ]

    for path in required_paths:
        if not Path(path).exists():
            logger.error(f"❌ 필수 데이터 없음: {path}")
            return False

    # 2. Python 패키지 체크
    required_packages = ["torch", "ultralytics", "clip", "pandas", "PIL"]

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} 설치됨")
        except ImportError:
            logger.error(f"❌ {package} 설치 필요")
            return False

    # 3. GPU 사용 가능 여부 체크
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"🚀 GPU 사용 가능: {torch.cuda.get_device_name()}")
        else:
            logger.info("💻 CPU로 훈련 진행")
    except:
        logger.warning("⚠️ PyTorch GPU 체크 실패")

    return True


def train_yolo_model():
    """YOLOv8 모델 훈련"""
    logger.info("🎯 YOLOv8 건물 피해 감지 모델 훈련 시작...")

    try:
        start_time = time.time()

        # YOLOv8 훈련 실행
        results = train_custom_yolo()

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"✅ YOLOv8 훈련 완료! ({duration:.1f}초)")
        return True, results

    except Exception as e:
        logger.error(f"❌ YOLOv8 훈련 실패: {e}")
        return False, None


def train_clip_model():
    """CLIP 모델 Fine-tuning"""
    logger.info("🔍 CLIP 건설 도메인 Fine-tuning 시작...")

    try:
        start_time = time.time()

        # CLIP Fine-tuning 실행
        fine_tuner = train_clip_finetuning()

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"✅ CLIP Fine-tuning 완료! ({duration:.1f}초)")
        return True, fine_tuner

    except Exception as e:
        logger.error(f"❌ CLIP Fine-tuning 실패: {e}")
        return False, None


def update_analysis_engine():
    """analysis_engine.py에서 새로운 모델 사용하도록 업데이트"""
    logger.info("🔧 analysis_engine.py 업데이트 중...")

    try:
        # 새로운 분석 엔진에 커스텀 모델 경로 추가
        engine_file = Path("../../app/analysis_engine.py")

        if engine_file.exists():
            content = engine_file.read_text(encoding="utf-8")

            # YOLOv8 모델 경로 업데이트
            if "yolov8n.pt" in content:
                updated_content = content.replace(
                    "yolov8n.pt", "train/models/custom_yolo_damage.pt"
                )

                # CLIP 모델 경로도 업데이트 (필요시)
                # updated_content = updated_content.replace(...)

                engine_file.write_text(updated_content, encoding="utf-8")
                logger.info("✅ analysis_engine.py 업데이트 완료")

        return True

    except Exception as e:
        logger.error(f"❌ analysis_engine.py 업데이트 실패: {e}")
        return False


def main():
    """메인 훈련 프로세스"""
    logger.info("🚀 건물 피해 분석 AI 모델 훈련 시작!")
    logger.info("=" * 60)

    # 1단계: 요구사항 체크
    if not check_requirements():
        logger.error("❌ 요구사항 체크 실패. 훈련을 중단합니다.")
        return False

    logger.info("✅ 모든 요구사항 충족!")
    logger.info("=" * 60)

    total_start_time = time.time()
    success_count = 0

    # 2단계: YOLOv8 훈련
    yolo_success, yolo_results = train_yolo_model()
    if yolo_success:
        success_count += 1

    logger.info("=" * 60)

    # 3단계: CLIP Fine-tuning
    clip_success, clip_fine_tuner = train_clip_model()
    if clip_success:
        success_count += 1

    logger.info("=" * 60)

    # 4단계: 분석 엔진 업데이트
    if yolo_success:
        update_success = update_analysis_engine()
        if update_success:
            success_count += 1

    # 최종 결과
    total_duration = time.time() - total_start_time

    logger.info("🎉 훈련 완료!")
    logger.info(f"📊 성공한 작업: {success_count}/3")
    logger.info(f"⏰ 총 소요 시간: {total_duration:.1f}초")

    if success_count >= 2:
        logger.info("✅ 훈련 성공! 새로운 모델을 사용할 수 있습니다.")
        logger.info("📁 생성된 파일:")

        model_files = [
            "../models/custom_yolo_damage.pt",
            "../models/clip_finetuned.pt",
            "../datasets/yolo_dataset/",
            "../runs/detect/building_damage/",
        ]

        for file_path in model_files:
            if Path(file_path).exists():
                logger.info(f"   - {file_path}")

        return True
    else:
        logger.error("❌ 훈련 중 일부 실패. 로그를 확인하세요.")
        return False


if __name__ == "__main__":
    print("🏗️ 건물 피해 분석 AI 모델 훈련")
    print("이 스크립트는 YOLOv8과 CLIP 모델을 순차적으로 훈련합니다.")
    print()

    # 사용자 확인
    response = input("훈련을 시작하시겠습니까? (y/N): ").strip().lower()

    if response in ["y", "yes"]:
        success = main()

        if success:
            print("\n🎉 모든 훈련이 완료되었습니다!")
            print("이제 Streamlit 앱을 실행하여 개선된 성능을 확인하세요:")
            print("  streamlit run streamlit_app.py")
        else:
            print("\n❌ 훈련 중 오류가 발생했습니다.")
            print("training.log 파일을 확인하세요.")
    else:
        print("훈련을 취소했습니다.")
