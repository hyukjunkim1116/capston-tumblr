"""
AI 분석 엔진 모듈 - 새로운 기술 스택
YOLOv8 + CLIP + Pandas + GPT-4 파이프라인
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
import cv2
import os
from openai import OpenAI
from datetime import datetime

# YOLOv8 관련
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv8 not available. Install with: pip install ultralytics")

# CLIP 관련
try:
    import clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning(
        "CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git"
    )

# LangChain 관련 - 최신 패키지로 업데이트
try:
    from langchain_openai import OpenAI as LangChainOpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to community package
        from langchain_community.llms import OpenAI as LangChainOpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logging.warning(
            "LangChain not available. Install with: pip install langchain-openai langchain-community"
        )

# 새로운 기준 데이터 매니저 import
from app.criteria_loader import get_criteria_manager

logger = logging.getLogger(__name__)

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


class AnalysisEngine:
    """통합 분석 엔진 - YOLOv8 + CLIP + GPT-4 파이프라인"""

    def __init__(self):
        """분석 엔진 초기화"""
        logger.info("AnalysisEngine 초기화 시작")

        # 시작 시간 기록
        self.start_time = time.time()

        # 각 모듈 초기화
        self.yolo_model = YOLODamageDetector()
        self.clip_model = CLIPDamageClassifier()
        self.gpt_model = GPTReportGenerator()

        # 데이터 프로세서 (이미지 검증용)
        from app.data_processor import DataProcessor

        self.data_processor = DataProcessor()

        logger.info("AnalysisEngine 초기화 완료")

    def generate_comprehensive_analysis(
        self, image_path: str, area: float, user_message: str = ""
    ) -> Dict[str, Any]:
        """종합 분석 실행 및 결과 반환"""
        try:
            logger.info("종합 분석 시작")
            self.start_time = time.time()

            # 1. 이미지 검증
            validation_result = self.data_processor.validate_image_content(image_path)
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "error_type": "validation_error",
                }

            # 2. YOLO 피해 탐지
            yolo_result = self.yolo_model.detect_damage_areas(image_path)
            if not yolo_result:  # 빈 리스트면 폴백 처리
                yolo_result = [
                    {
                        "bbox": [0, 0, 100, 100],
                        "confidence": 0.5,
                        "class_id": 0,
                        "area_id": "fallback",
                    }
                ]

            # 3. CLIP 분류
            image = Image.open(image_path)
            clip_results = {}

            for detection in yolo_result:
                bbox = detection["bbox"]
                crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                classification = self.clip_model.classify_damage_type(crop)

                # 최고 확률 피해 유형 선택
                best_damage_type = max(classification, key=classification.get)
                detection["class"] = best_damage_type
                detection["confidence"] = classification[best_damage_type]

                clip_results[best_damage_type] = {
                    "damage_type_kr": DAMAGE_TYPE_KR_MAP.get(
                        best_damage_type, best_damage_type
                    ),
                    "confidence": classification[best_damage_type],
                }

            # 4. 기준 데이터 조회 및 구조화된 데이터 생성
            structured_data = self._create_structured_analysis_data(
                yolo_result, clip_results, area, user_message
            )

            # 5. 사용자 친화적 텍스트 생성
            text_analysis = self._generate_user_friendly_text(structured_data)

            # 6. 결과 반환
            return {
                "success": True,
                "analysis_text": text_analysis,
                "structured_data": structured_data,
                "damage_areas": structured_data["damage_areas"],
                "yolo_detections": yolo_result,
                "clip_classifications": clip_results,
                "processing_time": time.time() - self.start_time,
            }

        except Exception as e:
            logger.error(f"종합 분석 오류: {e}")
            return {
                "success": False,
                "error": f"분석 중 오류 발생: {str(e)}",
                "error_type": "analysis_error",
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


class YOLODamageDetector:
    """YOLOv8 건물 피해 감지"""

    def __init__(self, model_path="train/models/custom_yolo_damage.pt"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """YOLO 모델 로드"""
        try:
            if YOLO_AVAILABLE and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"커스텀 YOLO 모델 로드: {self.model_path}")
            else:
                logger.warning("커스텀 모델 없음, 기본 모델 사용")
                if YOLO_AVAILABLE:
                    self.model = YOLO("yolov8n.pt")
                else:
                    logger.error("YOLO 모델 로드 실패")
                    self.model = None
        except Exception as e:
            logger.error(f"YOLO 모델 로드 오류: {e}")
            self.model = None

    def detect_damage_areas(self, image_path: str, use_tta: bool = True) -> List[Dict]:
        """
        건물 피해 영역 감지 (TTA 적용 가능)

        Args:
            image_path: 이미지 경로
            use_tta: Test Time Augmentation 사용 여부
        """
        if not self.model:
            logger.warning("YOLO 모델 없음, 폴백 감지 사용")
            return self._fallback_detection(image_path)

        try:
            if use_tta:
                return self._detect_with_tta(image_path)
            else:
                return self._detect_single(image_path)
        except Exception as e:
            logger.error(f"YOLO 감지 오류: {e}")
            return self._fallback_detection(image_path)

    def _detect_single(self, image_path: str) -> List[Dict]:
        """단일 이미지 감지"""
        results = self.model(image_path, conf=0.3)
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
                            "area_id": f"area_{i}",
                        }
                    )

        return detections if detections else self._fallback_detection(image_path)

    def _detect_with_tta(self, image_path: str) -> List[Dict]:
        """TTA를 사용한 감지 (여러 증강 이미지의 결과 앙상블)"""
        from PIL import Image, ImageEnhance, ImageOps
        import numpy as np
        import tempfile
        import os

        # 원본 이미지 로드
        original_image = Image.open(image_path)

        # TTA 변형 설정
        augmentations = [
            ("original", lambda img: img),  # 원본
            (
                "flip_horizontal",
                lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            ),  # 좌우 반전
            (
                "brightness_up",
                lambda img: ImageEnhance.Brightness(img).enhance(1.2),
            ),  # 밝기 증가
            (
                "brightness_down",
                lambda img: ImageEnhance.Brightness(img).enhance(0.8),
            ),  # 밝기 감소
            (
                "contrast_up",
                lambda img: ImageEnhance.Contrast(img).enhance(1.2),
            ),  # 대비 증가
            (
                "contrast_down",
                lambda img: ImageEnhance.Contrast(img).enhance(0.8),
            ),  # 대비 감소
            ("rotate_5", lambda img: img.rotate(5, expand=True)),  # 5도 회전
            ("rotate_-5", lambda img: img.rotate(-5, expand=True)),  # -5도 회전
        ]

        all_detections = []

        for aug_name, aug_func in augmentations:
            try:
                # 증강 이미지 생성
                aug_image = aug_func(original_image.copy())

                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as tmp_file:
                    aug_image.save(tmp_file.name, "JPEG")
                    tmp_path = tmp_file.name

                # 감지 실행
                results = self.model(tmp_path, conf=0.25)  # TTA에서는 confidence 낮게

                # 결과 처리
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())

                            # 좌우 반전의 경우 bbox 좌표 보정
                            if aug_name == "flip_horizontal":
                                img_width = aug_image.width
                                x1, x2 = img_width - x2, img_width - x1

                            detection = {
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": confidence,
                                "class_id": class_id,
                                "augmentation": aug_name,
                            }
                            all_detections.append(detection)

                # 임시 파일 삭제
                os.unlink(tmp_path)

            except Exception as e:
                logger.warning(f"TTA 증강 '{aug_name}' 실패: {e}")
                continue

        # NMS 적용하여 중복 제거 및 앙상블
        ensemble_detections = self._ensemble_detections(all_detections)

        return (
            ensemble_detections
            if ensemble_detections
            else self._fallback_detection(image_path)
        )

    def _ensemble_detections(self, all_detections: List[Dict]) -> List[Dict]:
        """여러 TTA 결과를 앙상블하여 최종 감지 결과 생성"""
        if not all_detections:
            return []

        # IoU 기반 클러스터링
        clusters = []
        iou_threshold = 0.5

        for detection in all_detections:
            bbox = detection["bbox"]
            assigned = False

            for cluster in clusters:
                # 클러스터 내 첫 번째 detection과 IoU 계산
                cluster_bbox = cluster[0]["bbox"]
                iou = self._calculate_iou(bbox, cluster_bbox)

                if iou > iou_threshold:
                    cluster.append(detection)
                    assigned = True
                    break

            if not assigned:
                clusters.append([detection])

        # 각 클러스터에서 평균값 계산
        ensemble_results = []
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:  # 최소 2개 이상의 감지에서 동의해야 함
                continue

            # 클러스터 내 평균 계산
            avg_bbox = self._average_bbox([d["bbox"] for d in cluster])
            avg_confidence = np.mean([d["confidence"] for d in cluster])
            most_common_class = max(
                set([d["class_id"] for d in cluster]),
                key=[d["class_id"] for d in cluster].count,
            )

            # 신뢰도 보정 (여러 모델에서 동의할수록 높은 신뢰도)
            consensus_boost = min(len(cluster) / len(augmentations), 1.0) * 0.2
            final_confidence = min(avg_confidence + consensus_boost, 1.0)

            ensemble_results.append(
                {
                    "bbox": avg_bbox,
                    "confidence": final_confidence,
                    "class_id": most_common_class,
                    "area_id": f"tta_area_{i}",
                    "consensus_count": len(cluster),
                }
            )

        return ensemble_results

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """두 바운딩 박스의 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 교집합 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _average_bbox(self, bboxes: List[List[int]]) -> List[int]:
        """바운딩 박스들의 평균 계산"""
        import numpy as np

        bboxes_array = np.array(bboxes)
        avg_bbox = np.mean(bboxes_array, axis=0)
        return [int(coord) for coord in avg_bbox]

    def _fallback_detection(self, image_path: str) -> List[Dict]:
        """YOLO 실패 시 폴백 감지"""
        # 전체 이미지를 하나의 감지 영역으로 처리
        image = Image.open(image_path)
        w, h = image.size

        return [
            {
                "bbox": [0, 0, w, h],
                "confidence": 0.5,
                "class_id": 0,
                "area_id": "full_image",
            }
        ]


class CLIPDamageClassifier:
    """CLIP 기반 피해 유형 분류"""

    def __init__(
        self, model_name="ViT-B/32", custom_model_path="train/models/clip_finetuned.pt"
    ):
        """Fine-tuned CLIP 모델 사용"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.custom_model_path = custom_model_path
        self.model = None
        self.preprocess = None

        try:
            # Fine-tuned 모델 우선 로드
            if self.custom_model_path and Path(self.custom_model_path).exists():
                try:
                    self.model, self.preprocess = clip.load(
                        self.custom_model_path, device=self.device
                    )
                    logger.info(f"Fine-tuned CLIP 모델 로드: {self.custom_model_path}")
                except Exception as e:
                    logger.warning(f"Fine-tuned 모델 로드 실패, 기본 모델 사용: {e}")
                    self.model, self.preprocess = clip.load(
                        "ViT-B/32", device=self.device
                    )
            else:
                logger.info("Fine-tuned 모델 없음, 기본 CLIP 사용")
                if CLIP_AVAILABLE:
                    self.model, self.preprocess = clip.load(
                        "ViT-B/32", device=self.device
                    )
                    logger.info("기본 CLIP 모델 로드 완료")
                else:
                    logger.error("CLIP 라이브러리 없음")
                    self.model = None
                    self.preprocess = None

        except Exception as e:
            logger.error(f"CLIP 모델 로드 실패: {e}")
            # 최종 폴백: 모델 없이 진행
            self.model = None
            self.preprocess = None

    def classify_damage_type(self, image_crop: Image.Image) -> Dict[str, float]:
        """크롭된 이미지의 피해 유형 분류"""
        if not self.model:
            return {"normal building": 1.0}

        try:
            # 이미지 전처리
            image_input = self.preprocess(image_crop).unsqueeze(0).to(self.device)

            # 텍스트 토큰화
            text_inputs = torch.cat(
                [
                    clip.tokenize(f"a photo of {damage_type}")
                    for damage_type in DAMAGE_TYPES
                ]
            ).to(self.device)

            # 추론
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # 결과 매핑
            result = {}
            for i, damage_type in enumerate(DAMAGE_TYPES):
                result[damage_type] = float(probs[i])

            return result

        except Exception as e:
            logger.error(f"CLIP 분류 오류: {e}")
            return {"normal building": 1.0}


class GPTReportGenerator:
    """OpenAI GPT-4 기반 보고서 생성기"""

    def __init__(self):
        self.client = None
        self.llm_chain = None

        # OpenAI 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)

            # LangChain 설정
            if LANGCHAIN_AVAILABLE:
                try:
                    llm = LangChainOpenAI(temperature=0.3, openai_api_key=api_key)

                    prompt_template = """
당신은 건물 피해 분석 전문가입니다. 다음 분석 데이터를 바탕으로 종합적인 건물 피해 분석 보고서를 작성해주세요.

분석 데이터:
{analysis_data}

기준 데이터:
{criteria_data}

다음 구조로 보고서를 작성해주세요:
1. 피해 현황 요약
2. 감지된 피해 영역별 상세 분석
3. 복구 방법 및 우선순위
4. 예상 비용 및 기간
5. 안전 권고사항

보고서는 전문적이면서도 이해하기 쉽게 작성해주세요.
"""

                    prompt = PromptTemplate(
                        input_variables=["analysis_data", "criteria_data"],
                        template=prompt_template,
                    )

                    self.llm_chain = LLMChain(llm=llm, prompt=prompt)
                    logger.info("GPT-4 보고서 생성기 초기화 완료")

                except Exception as e:
                    logger.error(f"LangChain 초기화 오류: {e}")

    def generate_report(self, analysis_results: Dict, criteria_data: Dict) -> str:
        """분석 결과를 바탕으로 보고서 생성"""
        try:
            if self.llm_chain:
                # LangChain 사용
                response = self.llm_chain.run(
                    analysis_data=str(analysis_results),
                    criteria_data=str(criteria_data),
                )
                return response

            elif self.client:
                # 직접 OpenAI API 사용
                messages = [
                    {
                        "role": "system",
                        "content": "당신은 건물 피해 분석 전문가입니다.",
                    },
                    {
                        "role": "user",
                        "content": f"""
다음 분석 데이터를 바탕으로 건물 피해 분석 보고서를 작성해주세요:

분석 결과: {analysis_results}
기준 데이터: {criteria_data}

전문적이고 상세한 보고서를 작성해주세요.
""",
                    },
                ]

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages, temperature=0.3
                )

                return response.choices[0].message.content

            else:
                return self._generate_fallback_report(analysis_results)

        except Exception as e:
            logger.error(f"보고서 생성 오류: {e}")
            return self._generate_fallback_report(analysis_results)

    def _generate_fallback_report(self, analysis_results: Dict) -> str:
        """GPT 실패 시 기본 보고서"""
        return f"""
# 건물 피해 분석 보고서

## 분석 개요
- 분석 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 감지된 피해 영역: {len(analysis_results.get('damage_areas', []))}개

## 주요 결과
{analysis_results}

## 권고사항
감지된 피해에 대해 전문가의 상세 검토가 필요합니다.
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
            yolo_detector = YOLODamageDetector()
            damage_areas = yolo_detector.detect_damage_areas(image_path)

            # 2단계: CLIP으로 각 영역의 피해 유형 분류
            logger.info("2단계: CLIP 피해 유형 분류 시작")
            clip_classifier = CLIPDamageClassifier()

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
