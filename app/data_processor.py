"""
데이터 처리 모듈
파일 업로드, 표준 수리 데이터 처리 등
"""

import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


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
