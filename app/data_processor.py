"""
데이터 처리 모듈
파일 업로드, 표준 수리 데이터 처리 등
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
from PIL import Image
import torch

# CLIP 모델 import
try:
    import clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_standard_repair_data(damage_type: str, area: float) -> Dict[str, Any]:
    """표준 수리 데이터 조회 (캐시됨)"""

    base_materials = {
        "균열": ["시멘트 모르타르", "에폭시 수지", "방수재"],
        "누수": ["방수 시트", "실리콘 실란트", "우레탄 방수재"],
        "화재": ["내화재", "단열재", "석고보드"],
        "부식": ["방청제", "아연도금강판", "부식방지제"],
    }

    base_equipment = {
        "균열": ["믹서기", "압축기", "주입기"],
        "누수": ["토치", "롤러", "압착기"],
        "화재": ["절단기", "용접기", "리프트"],
        "부식": ["샌딩기", "스프레이건", "브러시"],
    }

    labor_composition = {
        "균열": {"특급기능사": 1, "고급기능사": 1, "보통인부": 2},
        "누수": {"특급기능사": 1, "고급기능사": 2, "보통인부": 1},
        "화재": {"특급기능사": 2, "고급기능사": 2, "보통인부": 3},
        "부식": {"특급기능사": 1, "고급기능사": 1, "보통인부": 1},
    }

    # Calculate costs based on area
    material_cost_per_sqm = {"균열": 25000, "누수": 35000, "화재": 80000, "부식": 45000}
    labor_cost_per_sqm = {"균열": 15000, "누수": 20000, "화재": 40000, "부식": 25000}

    damage_key = next(
        (key for key in base_materials.keys() if key in damage_type), "균열"
    )

    return {
        "materials": base_materials.get(damage_key, base_materials["균열"]),
        "equipment": base_equipment.get(damage_key, base_equipment["균열"]),
        "labor": labor_composition.get(damage_key, labor_composition["균열"]),
        "material_cost": material_cost_per_sqm.get(damage_key, 25000) * area,
        "labor_cost": labor_cost_per_sqm.get(damage_key, 15000) * area,
        "duration_days": max(1, int(area / 10)),  # Rough estimate
    }


@st.cache_data(ttl=600)  # Cache for 10 minutes
def save_uploaded_file(
    uploaded_file_bytes: bytes, filename: str, upload_dir: Path
) -> Path:
    """업로드된 파일 저장 (캐시됨)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(filename).suffix
    cached_filename = f"upload_{timestamp}{file_extension}"
    file_path = upload_dir / cached_filename

    with open(file_path, "wb") as f:
        f.write(uploaded_file_bytes)

    return file_path


def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """업로드된 파일 유효성 검사"""
    if uploaded_file is None:
        return False, "파일이 선택되지 않았습니다."

    # Check file size (10MB limit)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "파일 크기가 10MB를 초과합니다."

    # Check file extension
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension not in allowed_extensions:
        return (
            False,
            f"지원되지 않는 파일 형식입니다. ({', '.join(allowed_extensions)} 형식만 지원)",
        )

    return True, "파일이 유효합니다."


def validate_area_input(area: float) -> tuple[bool, str]:
    """면적 입력 유효성 검사"""
    if area <= 0:
        return False, "면적은 0보다 큰 값이어야 합니다."

    if area > 10000:  # 10,000 m² limit
        return False, "면적이 너무 큽니다. (최대 10,000 m²)"

    return True, "면적이 유효합니다."


@st.cache_resource
def load_clip_for_validation():
    """이미지 검증용 CLIP 모델 로드 (캐시됨)"""
    if not CLIP_AVAILABLE:
        return None, None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        logger.info("CLIP 이미지 검증 모델 로드 완료")
        return model, preprocess
    except Exception as e:
        logger.error(f"CLIP 모델 로드 실패: {e}")
        return None, None


def validate_image_content(uploaded_file) -> Tuple[bool, str]:
    """이미지 내용이 건물인지 검증 (CLIP 사용)"""

    if not CLIP_AVAILABLE:
        # CLIP이 없으면 통과 (기본적으로 허용)
        return True, "이미지 내용 검증을 건너뜀 (CLIP 모델 없음)"

    try:
        # CLIP 모델 로드
        model, preprocess = load_clip_for_validation()
        if model is None:
            return True, "이미지 내용 검증을 건너뜀 (모델 로드 실패)"

        # 이미지 로드 및 전처리
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_input = preprocess(image).unsqueeze(0).to(device)

        # 건물 관련 vs 비건물 텍스트 프롬프트
        building_prompts = [
            "a photo of a building",
            "a photo of a house",
            "a photo of architecture",
            "a photo of a structure",
            "a photo of construction",
            "a photo of a damaged building",
            "a photo of building exterior",
            "a photo of building interior",
        ]

        non_building_prompts = [
            "a photo of a person",
            "a photo of people",
            "a photo of an animal",
            "a photo of food",
            "a photo of a vehicle",
            "a photo of nature landscape",
            "a photo of a document",
            "a photo of text",
        ]

        all_prompts = building_prompts + non_building_prompts
        text_inputs = torch.cat([clip.tokenize(prompt) for prompt in all_prompts]).to(
            device
        )

        # CLIP 추론
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # 건물 관련 확률 계산
        building_prob = sum(probs[: len(building_prompts)])
        non_building_prob = sum(probs[len(building_prompts) :])

        # 임계값 설정 (건물 관련 확률이 40% 이상이어야 통과)
        building_threshold = 0.4

        logger.info(
            f"이미지 내용 분석: 건물={building_prob:.3f}, 비건물={non_building_prob:.3f}"
        )

        if building_prob >= building_threshold:
            return True, f"건물 이미지로 확인됨 (신뢰도: {building_prob:.1%})"
        else:
            # 가장 높은 확률의 비건물 카테고리 찾기
            max_non_building_idx = (
                len(building_prompts) + probs[len(building_prompts) :].argmax()
            )
            detected_category = all_prompts[max_non_building_idx].replace(
                "a photo of ", ""
            )

            return (
                False,
                f"""
건물 이미지가 아닙니다

**감지된 내용**: {detected_category} (신뢰도: {probs[max_non_building_idx]:.1%})
**건물 관련 확률**: {building_prob:.1%}

**올바른 이미지 예시**:
• 건물 외부 전경
• 건물 내부 사진  
• 구조물 및 건축물
• 손상된 건물 부위

**지원하지 않는 이미지**:
• 인물 사진
• 동물 사진
• 음식 사진
• 자연 풍경 (건물 없음)
• 문서나 텍스트

건물 손상 분석을 위한 적절한 건물 사진을 업로드해 주세요.
""",
            )

    except Exception as e:
        logger.error(f"이미지 내용 검증 오류: {e}")
        # 오류 시에는 통과시킴 (사용자 경험 우선)
        return True, f"이미지 내용 검증 중 오류 발생: {str(e)}"
