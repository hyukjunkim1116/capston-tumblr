"""
AI 분석 엔진 모듈 - 환경별 성능 최적화
YOLOv8 + CLIP + GPT-4 파이프라인
"""

import streamlit as st
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw
import torch
import os
from openai import OpenAI
from datetime import datetime

# 로거 먼저 정의
logger = logging.getLogger(__name__)

# 설정 및 환경 정보 가져오기
from app.config import get_app_config, IS_DEPLOYMENT, DEVICE, BATCH_SIZE, MAX_IMAGE_SIZE

# 전역 설정
APP_CONFIG = get_app_config()

# YOLOv8 관련 - 환경별 최적화
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
    logger.info("✅ YOLOv8 패키지 로드 성공")
except ImportError as e:
    YOLO_AVAILABLE = False
    logger.warning(f"⚠️ YOLOv8 not available: {e}")
except Exception as e:
    YOLO_AVAILABLE = False
    logger.warning(f"⚠️ YOLOv8 로딩 오류: {e}")

# CLIP 관련 - 환경별 최적화
try:
    import clip

    CLIP_AVAILABLE = True
    logger.info("✅ CLIP 패키지 로드 성공")
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("⚠️ CLIP not available")

# LangChain 관련 - 최신 패키지로 업데이트
try:
    from langchain_openai import OpenAI as LangChainOpenAI
    from langchain.prompts import PromptTemplate

    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain 패키지 로드 성공")
except ImportError:
    try:
        from langchain_community.llms import OpenAI as LangChainOpenAI
        from langchain.prompts import PromptTemplate

        LANGCHAIN_AVAILABLE = True
        logger.info("✅ LangChain Community 패키지 로드 성공")
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logger.warning("⚠️ LangChain not available")

# 새로운 기준 데이터 매니저 import
from app.criteria_loader import get_criteria_manager

# 피해 유형 매핑 (CLIP 분류용)
DAMAGE_TYPES = [
    "crack damage",
    "water damage",
    "fire damage",
    "roof damage",
    "window damage",
    "door damage",
    "foundation damage",
    "structural deformation",
    "facade damage",
    "normal building",
]

# 한국어-영어 매핑
DAMAGE_TYPE_KR_MAP = {
    "crack damage": "균열 피해",
    "water damage": "수해 피해",
    "fire damage": "화재 피해",
    "roof damage": "지붕 피해",
    "window damage": "창문 피해",
    "door damage": "문 피해",
    "foundation damage": "기초 피해",
    "structural deformation": "구조적 변형",
    "facade damage": "외벽 피해",
    "normal building": "정상",
}


@st.cache_resource
def get_shared_models():
    """전역 공유 모델 인스턴스 - 메모리 효율성"""
    from app.config import initialize_optimized_models

    return initialize_optimized_models()


class AnalysisEngine:
    """통합 분석 엔진 - 환경별 최적화"""

    def __init__(self):
        """분석 엔진 초기화 - 모든 환경에서 동일한 고성능 설정"""
        logger.info("🚀 AnalysisEngine 초기화 시작")

        # 시작 시간 기록
        self.start_time = time.time()

        # 모든 환경에서 동일한 고성능 설정
        self.config = APP_CONFIG
        self.device = DEVICE
        self.is_deployment = IS_DEPLOYMENT  # 로깅용으로만 사용
        self.max_image_size = MAX_IMAGE_SIZE  # 모든 환경에서 2048

        # 공유 모델 인스턴스 사용 (메모리 절약)
        self.shared_models = get_shared_models()

        # 각 모듈 초기화 (모델 재사용) - 모든 환경에서 고성능 설정
        self.yolo_model = OptimizedYOLODetector(self.shared_models.get("yolo"))
        self.clip_model = OptimizedCLIPClassifier(self.shared_models.get("clip"))
        self.gpt_model = OptimizedGPTGenerator(self.shared_models.get("openai"))

        # 데이터 프로세서 (이미지 검증용)
        from app.data_processor import DataProcessor

        self.data_processor = DataProcessor()

        init_time = time.time() - self.start_time
        logger.info(
            f"✅ AnalysisEngine 초기화 완료 ({init_time:.2f}초) - 환경: {self.device}"
        )
        logger.info("🎯 설정: 모든 환경에서 커스텀 모델 사용 강제")

    @st.cache_data
    def generate_comprehensive_analysis(
        self, image_path: str, area: float, user_message: str = ""
    ) -> Dict[str, Any]:
        """종합 분석 실행 - 캐싱 적용"""
        try:
            logger.info("🔍 종합 분석 시작")
            self.start_time = time.time()

            # 이미지 최적화 전처리
            processed_image_path = self._optimize_image_for_analysis(image_path)

            # 1. 이미지 검증
            validation_result = self.data_processor.validate_image_content(
                processed_image_path
            )
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "error_type": "validation_error",
                }

            # 2. YOLO 피해 탐지 (모든 환경에서 고정확도 설정)
            yolo_result = self.yolo_model.detect_damage_areas(
                processed_image_path,
                use_tta=True,  # 모든 환경에서 TTA 활성화
            )

            if not yolo_result:
                yolo_result = self._create_fallback_detection(processed_image_path)

            # 3. CLIP 분류 (배치 처리 최적화)
            clip_results = self.clip_model.classify_damage_areas_batch(
                processed_image_path, yolo_result
            )

            # 결과 통합
            for i, detection in enumerate(yolo_result):
                if i < len(clip_results):
                    best_damage_type = max(clip_results[i], key=clip_results[i].get)
                    detection["class"] = best_damage_type
                    detection["confidence"] = clip_results[i][best_damage_type]

            # 4. 구조화된 데이터 생성
            structured_data = self._create_structured_analysis_data(
                yolo_result, clip_results, area, user_message
            )

            # 5. 텍스트 분석 생성
            text_analysis = self._generate_user_friendly_text(structured_data)

            processing_time = time.time() - self.start_time

            return {
                "success": True,
                "analysis_text": text_analysis,
                "structured_data": structured_data,
                "damage_areas": structured_data["damage_areas"],
                "yolo_detections": yolo_result,
                "clip_classifications": clip_results,
                "processing_time": processing_time,
                "environment": self.config["environment"],
                "optimizations_applied": self._get_optimization_summary(),
            }

        except Exception as e:
            logger.error(f"❌ 종합 분석 오류: {e}")
            return {
                "success": False,
                "error": f"분석 중 오류 발생: {str(e)}",
                "error_type": "analysis_error",
            }

    def _optimize_image_for_analysis(self, image_path: str) -> str:
        """이미지 최적화 전처리 - 환경별 크기 조정"""
        try:
            with Image.open(image_path) as img:
                # 환경별 최대 크기 제한
                max_size = self.max_image_size

                if max(img.size) > max_size:
                    # 비율 유지하면서 크기 조정
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                    # 임시 파일로 저장
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jpg"
                    ) as tmp:
                        img.save(tmp.name, "JPEG", quality=85, optimize=True)
                        logger.info(
                            f"📏 이미지 최적화: {image_path} -> {tmp.name} (max: {max_size})"
                        )
                        return tmp.name

            return image_path

        except Exception as e:
            logger.warning(f"⚠️ 이미지 최적화 실패: {e}, 원본 사용")
            return image_path

    def _create_fallback_detection(self, image_path: str) -> List[Dict]:
        """폴백 감지 결과 생성"""
        with Image.open(image_path) as img:
            w, h = img.size
            return [
                {
                    "bbox": [0, 0, w, h],
                    "confidence": 0.7,
                    "class_id": 0,
                    "area_id": "full_image_fallback",
                }
            ]

    def _get_optimization_summary(self) -> Dict[str, Any]:
        """적용된 설정 요약 (모든 환경에서 동일한 고정확도 설정)"""
        return {
            "environment": self.config["environment"],
            "device": self.device,
            "max_image_size": self.max_image_size,  # 모든 환경에서 2048
            "models_loaded": {
                "yolo": self.shared_models.get("yolo") is not None,
                "clip": self.shared_models.get("clip") is not None,
                "openai": self.shared_models.get("openai") is not None,
            },
            "high_accuracy_mode": True,  # 모든 환경에서 고정확도
            "tta_enabled": True,  # 모든 환경에서 TTA 활성화
            "batch_size": BATCH_SIZE,  # 모든 환경에서 4
        }

    def _create_structured_analysis_data(
        self, detections: list, classifications: dict, area: float, user_message: str
    ) -> Dict[str, Any]:
        """구조화된 분석 데이터 생성"""
        from app.criteria_loader import get_criteria_manager

        criteria_manager = get_criteria_manager()
        damage_areas = []

        for i, detection in enumerate(detections):
            damage_type = detection.get("class", "unknown")
            confidence = detection.get("confidence", 0.0)

            # CLIP 분류 결과 반영
            if damage_type in classifications:
                classification = classifications[damage_type]
                damage_type_kr = classification.get("damage_type_kr", damage_type)
            else:
                damage_type_kr = self._get_korean_damage_type(damage_type)

            # 기준 데이터에서 상세 정보 조회
            criteria = criteria_manager.get_damage_assessment_criteria(damage_type)

            # 심각도 계산
            severity_level = min(5, max(1, int(confidence * 5) + 1))
            severity_desc = (
                list(criteria.get("severity_levels", {}).values())[severity_level - 1]
                if criteria.get("severity_levels")
                else "보통 손상"
            )

            # 구조화된 피해 영역 데이터
            damage_area = {
                "name": f"피해영역 {i+1}",
                "damage_type": damage_type,
                "damage_type_kr": damage_type_kr,
                "confidence": confidence,
                "severity_level": severity_level,
                "description": f"{damage_type_kr} - {severity_desc} (신뢰도: {confidence:.2f})",
                "basis": self._get_recovery_basis(damage_type, criteria),
                "process": self._get_process_name(damage_type, criteria),
                "materials": self._get_material_list(damage_type, criteria),
                "coordinates": detection.get("bbox", [0, 0, 0, 0]),
            }

            damage_areas.append(damage_area)

        return {
            "basic_info": {
                "analysis_date": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
                "analysis_area": area,
                "user_message": user_message,
                "total_damages": len(
                    [d for d in damage_areas if d["damage_type"] != "normal building"]
                ),
                "total_areas": len(damage_areas),
            },
            "damage_areas": damage_areas,
            "summary": {
                "total_detections": len(detections),
                "damage_count": len(
                    [d for d in damage_areas if d["damage_type"] != "normal building"]
                ),
                "normal_count": len(
                    [d for d in damage_areas if d["damage_type"] == "normal building"]
                ),
            },
        }

    def _get_korean_damage_type(self, damage_type: str) -> str:
        """피해 유형 한국어 변환"""
        return DAMAGE_TYPE_KR_MAP.get(damage_type, damage_type)

    def _get_recovery_basis(self, damage_type: str, criteria: Dict) -> str:
        """복구 근거 생성 (표준시방서 기반)"""
        basis_templates = {
            "crack damage": "「건축공사 표준시방서」 제6장 콘크리트공사 6.5.3 균열보수에 따라 균열폭 0.3mm 이상 시 에폭시 수지 주입공법을 적용하며, 「KCS 41 30 02 현장치기콘크리트공사」의 균열보수 기준을 준수합니다.",
            "water damage": "「건축공사 표준시방서」 제12장 방수공사 및 「KCS 41 40 00 방수공사」에 따라 침수 피해 부위의 기존 방수층 제거 후 바탕처리 및 신규 방수재 시공을 실시합니다.",
            "fire damage": "「건축공사 표준시방서」 제14장 마감공사 및 「재난 안전 관리 지침」에 따라 화재 손상 부위의 구조 안전성 검토 후 손상 정도에 따른 단계적 복구를 실시합니다.",
            "roof damage": "「건축공사 표준시방서」 제11장 지붕공사 및 「KCS 41 50 00 지붕공사」에 따라 지붕재 교체 및 방수층 보강을 실시하며, 구조체 점검을 선행합니다.",
            "window damage": "「건축공사 표준시방서」 제9장 창호공사 및 「KCS 41 60 00 창호공사」에 따라 손상된 창호의 해체 및 신규 창호 설치를 실시합니다.",
            "door damage": "「건축공사 표준시방서」 제9장 창호공사 및 「KCS 41 60 00 창호공사」에 따라 손상된 문짝 및 문틀의 보수 또는 교체를 실시합니다.",
            "foundation damage": "「건축공사 표준시방서」 제5장 기초공사 및 「KCS 41 20 00 기초공사」에 따라 기초 구조체의 안전성 검토 후 언더피닝 또는 보강공법을 적용합니다.",
            "structural deformation": "「건축구조기준」 및 「KCS 41 30 00 콘크리트공사」에 따라 구조 안전성 정밀진단 후 구조보강 설계에 의한 보강공사를 실시합니다.",
            "facade damage": "「건축공사 표준시방서」 제14장 마감공사 및 「KCS 41 70 00 마감공사」에 따라 외벽 마감재의 해체 및 신규 시공을 실시합니다.",
        }

        return basis_templates.get(
            damage_type,
            f"「건축공사 표준시방서」 및 관련 KCS 기준에 따라 {damage_type} 복구공사를 실시합니다.",
        )

    def _get_process_name(self, damage_type: str, criteria: Dict) -> str:
        """공정명 생성"""
        process_names = {
            "crack damage": "균열보수공사 → 에폭시수지 주입공법 → 표면 마감공사",
            "water damage": "기존 방수층 제거 → 바탕처리 → 방수재 도포 → 보호층 시공",
            "fire damage": "화재손상부 해체 → 구조보강 → 마감재 시공 → 도장공사",
            "roof damage": "기존 지붕재 해체 → 방수층 보강 → 지붕재 설치 → 마감공사",
            "window damage": "기존 창호 해체 → 개구부 정리 → 신규 창호 설치 → 실링공사",
            "door damage": "기존 문 해체 → 문틀 점검 → 신규 문 설치 → 마감공사",
            "foundation damage": "기초 굴착 → 구조보강 → 언더피닝 → 되메우기",
            "structural deformation": "구조진단 → 보강설계 → 구조보강공사 → 마감복구",
            "facade damage": "기존 마감재 해체 → 바탕처리 → 신규 마감재 시공 → 실링공사",
        }

        return process_names.get(damage_type, f"{damage_type} 복구공사")

    def _get_material_list(self, damage_type: str, criteria: Dict) -> list:
        """복구 예상 자재 리스트 생성"""
        materials_dict = {
            "crack damage": [
                {"name": "에폭시 수지", "usage": "균열 주입용"},
                {"name": "프라이머", "usage": "접착력 향상"},
                {"name": "실링재", "usage": "표면 마감"},
            ],
            "water damage": [
                {"name": "우레탄 방수재", "usage": "방수층 형성"},
                {"name": "프라이머", "usage": "바탕처리"},
                {"name": "보호몰탈", "usage": "방수층 보호"},
            ],
            "fire damage": [
                {"name": "내화 보드", "usage": "내화성능 확보"},
                {"name": "구조용 접착제", "usage": "구조 보강"},
                {"name": "내화 도료", "usage": "마감 및 보호"},
            ],
            "roof damage": [
                {"name": "기와 또는 슬레이트", "usage": "지붕재"},
                {"name": "방수시트", "usage": "방수층"},
                {"name": "단열재", "usage": "단열성능"},
            ],
            "window damage": [
                {"name": "알루미늄 창호", "usage": "창호 교체"},
                {"name": "복층유리", "usage": "단열성능"},
                {"name": "실링재", "usage": "기밀성 확보"},
            ],
            "door damage": [
                {"name": "목재 또는 스틸 도어", "usage": "문짝 교체"},
                {"name": "경첩 및 손잡이", "usage": "하드웨어"},
                {"name": "실링재", "usage": "기밀성 확보"},
            ],
            "foundation damage": [
                {"name": "구조용 콘크리트", "usage": "기초 보강"},
                {"name": "철근", "usage": "구조 보강"},
                {"name": "방수재", "usage": "지하 방수"},
            ],
            "structural deformation": [
                {"name": "구조용 강재", "usage": "구조 보강"},
                {"name": "고강도 볼트", "usage": "접합부"},
                {"name": "무수축 몰탈", "usage": "충전재"},
            ],
            "facade damage": [
                {"name": "외장 마감재", "usage": "외벽 마감"},
                {"name": "단열재", "usage": "단열성능"},
                {"name": "마감 도료", "usage": "최종 마감"},
            ],
        }

        return materials_dict.get(
            damage_type, [{"name": "표준 건축자재", "usage": "복구용"}]
        )

    def _generate_user_friendly_text(self, structured_data: Dict) -> str:
        """사용자 친화적 텍스트 생성 (표준시방서 근거 포함)"""
        basic_info = structured_data["basic_info"]
        damage_areas = structured_data["damage_areas"]

        sections = []

        # 1. 피해 현황
        sections.append("피해 현황")
        sections.append(
            f"총 {basic_info['total_areas']}개 영역 중 {basic_info['total_damages']}개 피해 영역을 발견했습니다."
        )

        for area in damage_areas:
            if area["damage_type"] != "normal building":
                sections.append(f"• {area['name']}: {area['description']}")

        # 2. 복구 근거
        sections.append("\n복구 근거")
        for area in damage_areas:
            if area["damage_type"] != "normal building":
                sections.append(f"**{area['name']}**: {area['basis']}")

        # 3. 공정명
        sections.append("\n복구 공정")
        for area in damage_areas:
            if area["damage_type"] != "normal building":
                sections.append(f"**{area['name']}**: {area['process']}")

        # 4. 복구 예상 자재
        sections.append("\n복구 예상 자재")
        for area in damage_areas:
            if area["damage_type"] != "normal building":
                sections.append(f"**{area['name']}**:")
                for material in area["materials"]:
                    sections.append(f"  - {material['name']}: {material['usage']}")

        return "\n".join(sections)


class OptimizedYOLODetector:
    """모든 환경에서 동일한 고성능 YOLOv8 건물 피해 감지"""

    def __init__(self, shared_model=None):
        """공유 모델 인스턴스 사용"""
        self.model = shared_model
        self.device = DEVICE

        if self.model:
            logger.info("✅ YOLO 커스텀 모델 공유 인스턴스 사용")
        else:
            logger.error("❌ YOLO 커스텀 모델 없음 - 시스템 중단")
            raise ValueError("커스텀 YOLO 모델이 필요합니다")

    def detect_damage_areas(self, image_path: str, use_tta: bool = True) -> List[Dict]:
        """모든 환경에서 동일한 고성능 피해 영역 감지"""
        if not self.model:
            logger.error("❌ YOLO 모델 없음")
            raise ValueError("YOLO 모델이 로드되지 않았습니다")

        try:
            # 모든 환경에서 동일한 고성능 설정 사용
            return self._detect_with_high_accuracy(image_path, use_tta)

        except Exception as e:
            logger.error(f"❌ YOLO 감지 오류: {e}")
            # 폴백 대신 에러 발생 (커스텀 모델 강제)
            raise e

    def _detect_with_high_accuracy(
        self, image_path: str, use_tta: bool = True
    ) -> List[Dict]:
        """고성능 감지 - 모든 환경에서 동일한 설정"""
        # 모든 환경에서 동일한 고성능 설정
        conf_threshold = 0.3  # 낮은 임계값으로 더 많은 감지
        max_det = 50  # 더 많은 감지 허용

        if use_tta:
            # TTA 적용 - 모든 환경에서 활성화
            results_list = []

            # 원본 이미지
            results_list.append(
                self.model(
                    image_path,
                    conf=conf_threshold,
                    max_det=max_det,
                    device=self.device,
                    verbose=False,
                )
            )

            # 좌우 반전
            results_list.append(
                self.model(
                    image_path,
                    conf=conf_threshold,
                    max_det=max_det,
                    device=self.device,
                    verbose=False,
                    augment=True,
                )
            )

            # 최고 신뢰도 결과 선택
            best_results = results_list[0]
            for results in results_list[1:]:
                for r in results:
                    if r.boxes is not None and len(r.boxes) > len(
                        best_results[0].boxes or []
                    ):
                        best_results = results
                        break

            results = best_results
        else:
            results = self.model(
                image_path,
                conf=conf_threshold,
                max_det=max_det,
                device=self.device,
                verbose=False,
            )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "class_id": class_id,
                            "area_id": f"custom_yolo_area_{i}",
                        }
                    )

        if not detections:
            # 최소한의 폴백 (전체 이미지)
            logger.warning("⚠️ YOLO에서 감지된 피해 없음, 전체 이미지 분석")
            return self._minimal_fallback_detection(image_path)

        return detections

    def _minimal_fallback_detection(self, image_path: str) -> List[Dict]:
        """최소한의 폴백 감지 (전체 이미지만)"""
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                return [
                    {
                        "bbox": [0, 0, w, h],
                        "confidence": 0.6,
                        "class_id": 0,
                        "area_id": "full_image_analysis",
                    }
                ]
        except:
            return [
                {
                    "bbox": [0, 0, 800, 600],
                    "confidence": 0.5,
                    "class_id": 0,
                    "area_id": "default_analysis",
                }
            ]


class OptimizedCLIPClassifier:
    """모든 환경에서 동일한 고성능 CLIP 기반 피해 유형 분류"""

    def __init__(self, shared_model_data=None):
        """공유 모델 인스턴스 사용"""
        self.device = DEVICE

        if shared_model_data and len(shared_model_data) == 2:
            self.model, self.preprocess = shared_model_data
            logger.info("✅ CLIP 모델 공유 인스턴스 사용")
        else:
            self.model = None
            self.preprocess = None
            logger.warning("⚠️ CLIP 모델 없음, fallback 모드")

    def classify_damage_type(self, image_crop: Image.Image) -> Dict[str, float]:
        """모든 환경에서 동일한 고성능 피해 유형 분류"""
        if not self.model:
            return self._fallback_classification()

        try:
            # 모든 환경에서 동일한 고해상도 처리
            image_crop = image_crop.resize((224, 224), Image.Resampling.LANCZOS)

            image_input = self.preprocess(image_crop).unsqueeze(0).to(self.device)

            # 모든 환경에서 전체 피해 유형 분류 (제한 없음)
            damage_types = DAMAGE_TYPES  # 전체 피해 유형 사용
            text_inputs = torch.cat(
                [
                    self._tokenize_with_fallback(f"a photo of {damage_type}")
                    for damage_type in damage_types
                ]
            ).to(self.device)

            # 추론 실행
            with torch.no_grad():
                logits_per_image, _ = self.model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # 결과 매핑
            result = {}
            for i, damage_type in enumerate(damage_types):
                result[damage_type] = float(probs[i])

            return result

        except Exception as e:
            logger.error(f"❌ CLIP 분류 오류: {e}")
            return self._fallback_classification()

    def _tokenize_with_fallback(self, text: str):
        """안전한 토큰화"""
        try:
            import clip

            return clip.tokenize(text)
        except:
            # Fallback - 더미 토큰
            return torch.zeros(1, 77, dtype=torch.long)

    def _fallback_classification(self) -> Dict[str, float]:
        """모델 실패 시 기본 분류"""
        return {damage_type: 0.1 for damage_type in DAMAGE_TYPES}

    def classify_damage_areas_batch(
        self, image_path: str, detections: List[Dict]
    ) -> List[Dict]:
        """고정확도 배치 처리로 여러 영역 분류"""
        results = []

        try:
            with Image.open(image_path) as img:
                # 모든 환경에서 동일한 배치 크기 사용
                batch_size = min(BATCH_SIZE, len(detections))  # 설정된 배치 크기 사용

                for i in range(0, len(detections), batch_size):
                    batch_detections = detections[i : i + batch_size]
                    batch_results = []

                    for detection in batch_detections:
                        bbox = detection["bbox"]
                        crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        result = self.classify_damage_type(crop)
                        batch_results.append(result)

                    results.extend(batch_results)

            return results

        except Exception as e:
            logger.error(f"❌ 배치 분류 오류: {e}")
            return [self._fallback_classification() for _ in detections]


class OptimizedGPTGenerator:
    """모든 환경에서 동일한 고성능 GPT 보고서 생성기"""

    def __init__(self, shared_client=None):
        """공유 OpenAI 클라이언트 사용"""
        self.client = shared_client

        if self.client:
            logger.info("✅ OpenAI 클라이언트 공유 인스턴스 사용")
        else:
            logger.warning("⚠️ OpenAI 클라이언트 없음, fallback 모드")

        # LangChain 설정 (선택적)
        self.llm = None
        self.prompt = None

        if LANGCHAIN_AVAILABLE and self.client:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm = LangChainOpenAI(
                        temperature=0.3,  # 모든 환경에서 동일한 온도
                        openai_api_key=api_key,
                        max_tokens=2000,  # 모든 환경에서 높은 토큰 제한
                    )

                    self.prompt_template = """
당신은 건물 피해 분석 전문가입니다. 다음 분석 데이터를 바탕으로 상세한 건물 피해 분석 보고서를 작성해주세요.

분석 데이터: {analysis_data}

다음 구조로 상세한 보고서를 작성해주세요:
1. 피해 현황 요약
2. 주요 피해 영역 분석 (각 영역별 상세 설명)
3. 복구 권고사항 (우선순위 포함)
4. 안전성 평가

보고서는 전문적이면서도 실용적으로 작성해주세요.
"""

                    self.prompt = PromptTemplate(
                        input_variables=["analysis_data"],
                        template=self.prompt_template,
                    )

                    logger.info("✅ LangChain GPT 설정 완료")
            except Exception as e:
                logger.warning(f"⚠️ LangChain 설정 실패: {e}")

    def generate_report(
        self, analysis_results: Dict, criteria_data: Dict = None
    ) -> str:
        """모든 환경에서 동일한 고성능 보고서 생성"""
        try:
            if self.llm and self.prompt:
                # LangChain 사용 (최신 패턴)
                formatted_prompt = self.prompt.format(
                    analysis_data=str(analysis_results)[:3000]  # 더 많은 토큰 허용
                )
                response = self.llm.invoke(formatted_prompt)
                return response

            elif self.client:
                # 직접 OpenAI API 사용 - 모든 환경에서 동일한 고성능 설정
                model = "gpt-4o"  # 모든 환경에서 최고 품질 모델
                max_tokens = 2000  # 모든 환경에서 높은 토큰 제한

                messages = [
                    {
                        "role": "system",
                        "content": "당신은 건물 피해 분석 전문가입니다. 상세하고 정확한 보고서를 작성합니다.",
                    },
                    {
                        "role": "user",
                        "content": f"""
다음 분석 결과를 바탕으로 상세한 건물 피해 분석 보고서를 작성해주세요:

{str(analysis_results)[:3000]}

구조:
1. 피해 현황 요약
2. 주요 피해 영역 상세 분석
3. 복구 권고사항 (우선순위 포함)
4. 안전성 평가

상세하고 전문적인 보고서를 작성해주세요.
""",
                    },
                ]

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=max_tokens,
                )

                return response.choices[0].message.content

            else:
                return self._generate_fallback_report(analysis_results)

        except Exception as e:
            logger.error(f"❌ 보고서 생성 오류: {e}")
            return self._generate_fallback_report(analysis_results)

    def _generate_fallback_report(self, analysis_results: Dict) -> str:
        """GPT 실패 시 구조화된 기본 보고서"""
        damage_count = len(analysis_results.get("damage_areas", []))

        return f"""
# 건물 피해 분석 보고서

## 분석 개요
- 분석 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 감지된 피해 영역: {damage_count}개
- 분석 환경: {APP_CONFIG.get('environment', 'unknown')}

## 주요 결과
{analysis_results.get('analysis_text', '분석 결과 처리 중입니다.')}

## 권고사항
감지된 피해에 대해 전문가의 상세 검토가 필요합니다.
우선순위에 따라 보수 계획을 수립하시기 바랍니다.

---
*본 보고서는 AI 분석 결과이며, 정확한 진단을 위해서는 전문가 검토가 필요합니다.*
"""


def analyze_damage_with_ai(
    image_path: str,
    area: float,
    user_message: str,
) -> str:

    start_time = time.time()

    try:
        # Show single loading message
        with st.spinner("응답 생성 중..."):

            # 1단계: YOLOv8로 피해 영역 감지
            logger.info("1단계: YOLOv8 피해 영역 감지 시작")
            yolo_detector = OptimizedYOLODetector()
            damage_areas = yolo_detector.detect_damage_areas(image_path)

            # 2단계: CLIP으로 각 영역의 피해 유형 분류
            logger.info("2단계: CLIP 피해 유형 분류 시작")
            clip_classifier = OptimizedCLIPClassifier()

            image = Image.open(image_path)
            classified_damages = []

            for area_info in damage_areas:
                bbox = area_info["bbox"]

                # 이미지 크롭
                crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

                # CLIP 분류
                classification = clip_classifier.classify_damage_type(crop)

                # 최고 확률 피해 유형 선택
                best_damage_type = max(classification, key=classification.get)
                confidence = classification[best_damage_type]

                classified_damages.append(
                    {
                        "area_id": area_info["area_id"],
                        "bbox": bbox,
                        "damage_type": best_damage_type,
                        "damage_type_kr": DAMAGE_TYPE_KR_MAP.get(
                            best_damage_type, best_damage_type
                        ),
                        "confidence": confidence,
                        "yolo_confidence": area_info["confidence"],
                    }
                )

            # 3단계: 새로운 CriteriaDataManager로 기준 데이터 매핑
            logger.info("3단계: 기준 데이터 매핑 시작")
            criteria_manager = get_criteria_manager()

            repair_specifications = []
            for damage in classified_damages:
                if damage["damage_type"] != "normal building":
                    # 새로운 매니저를 사용하여 상세한 평가 기준 조회
                    damage_criteria = criteria_manager.get_damage_assessment_criteria(
                        damage["damage_type"]
                    )
                    damage_criteria["damage_info"] = damage
                    repair_specifications.append(damage_criteria)

            # 4단계: 분석 결과 구성
            analysis_results = {
                "image_path": image_path,
                "area_sqm": area,
                "user_message": user_message,
                "damage_areas": classified_damages,
                "repair_specifications": repair_specifications,
                "total_damages": len(
                    [
                        d
                        for d in classified_damages
                        if d["damage_type"] != "normal building"
                    ]
                ),
                "analysis_time": time.time() - start_time,
            }

            # 5단계: 새로운 CriteriaDataManager로 종합 보고서 생성
            logger.info("5단계: 종합 보고서 생성 시작")
            final_report = criteria_manager.generate_comprehensive_report(
                analysis_results
            )

            # 시각화 이미지 생성 및 저장
            try:
                annotated_image = create_damage_visualization(
                    image_path, classified_damages
                )

                # 임시 파일로 저장 (UI에서 표시용)
                import tempfile

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".png"
                ) as tmp_file:
                    annotated_image.save(tmp_file.name)
                    analysis_results["annotated_image_path"] = tmp_file.name

                logger.info("피해 감지 시각화 이미지 생성 완료")
            except Exception as e:
                logger.warning(f"시각화 생성 실패: {e}")

            analysis_time = time.time() - start_time
            logger.info(f"새로운 분석 파이프라인 완료: {analysis_time:.2f}초")

            return final_report

    except Exception as e:
        logger.error(f"분석 오류: {e}")
        return f"""
건물 피해 분석 오류

분석 중 오류가 발생했습니다: {str(e)}

잠시 후 다시 시도해주세요.
"""


def create_damage_visualization(
    image_path: str, damage_results: List[Dict]
) -> Image.Image:
    """피해 감지 결과를 시각화한 이미지 생성"""
    try:
        # 원본 이미지 로드
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # 색상 매핑 (피해 유형별)
        color_map = {
            "crack damage": "#FF6B6B",  # 빨간색
            "water damage": "#4ECDC4",  # 청록색
            "fire damage": "#FF8E53",  # 주황색
            "roof damage": "#95E1D3",  # 연두색
            "window damage": "#A8E6CF",  # 민트색
            "door damage": "#FFEAA7",  # 노란색
            "foundation damage": "#DDA0DD",  # 자주색
            "structural deformation": "#FF7675",  # 분홍색
            "facade damage": "#74B9FF",  # 파란색
            "normal building": "#00B894",  # 초록색
        }

        # 각 감지 영역에 바운딩 박스 그리기
        for damage in damage_results:
            if "bbox" in damage:
                bbox = damage["bbox"]
                damage_type = damage.get("damage_type", "normal building")
                confidence = damage.get("confidence", 0.0)

                # 바운딩 박스 색상
                color = color_map.get(damage_type, "#GRAY")

                # 바운딩 박스 그리기
                draw.rectangle(bbox, outline=color, width=3)

                # 라벨 텍스트
                damage_kr = damage.get("damage_type_kr", damage_type)
                label = f"{damage_kr} ({confidence:.2f})"

                # 라벨 배경
                text_bbox = draw.textbbox((bbox[0], bbox[1] - 25), label)
                draw.rectangle(text_bbox, fill=color)

                # 라벨 텍스트
                draw.text((bbox[0], bbox[1] - 25), label, fill="white")

        return image

    except Exception as e:
        logger.error(f"시각화 생성 오류: {e}")
        # 원본 이미지 반환
        return Image.open(image_path)
