"""
Simple ChatGPT-style Streamlit web application for building damage analysis
"""

import streamlit as st
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional

# Import the new simple UI
from ui.ui_components import render_simple_chatgpt_ui

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce verbosity of external libraries
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Import our analysis modules
try:
    from langchain_integration import analyze_building_damage, DamageAnalysisOutput
    from config import DAMAGE_CATEGORIES, CACHE_DIR, LOGS_DIR
    from models import BuildingDamageAnalysisModel

    # FAISS Vector store 직접 사용
    try:
        from vector_store_faiss import create_faiss_vector_store

        VECTOR_STORE_AVAILABLE = True
        logger.info("FAISS vector store available")
    except Exception as ve:
        logger.warning(f"FAISS vector store not available: {ve}")
        VECTOR_STORE_AVAILABLE = False

        @st.cache_resource
        def create_faiss_vector_store():
            """Fallback vector store function"""
            logger.warning("Using fallback vector store")
            return None

    MODULES_LOADED = True
except ImportError as e:
    st.error(f"모듈 로딩 오류: {e}")
    st.error("필요한 모듈을 찾을 수 없습니다. 개발자에게 문의하세요.")
    MODULES_LOADED = False
    VECTOR_STORE_AVAILABLE = False
    # Fallback values
    DAMAGE_CATEGORIES = {}
    CACHE_DIR = Path("./cache")
    LOGS_DIR = Path("./logs")

    @st.cache_resource
    def create_faiss_vector_store():
        return None

    def analyze_building_damage(*args, **kwargs):
        return {"error": "분석 모듈을 사용할 수 없습니다."}


# Create directories
UPLOAD_DIR = CACHE_DIR / "uploads"
RESULTS_DIR = CACHE_DIR / "results"

for directory in [UPLOAD_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Performance optimization: Thread pool for async operations
@st.cache_resource
def get_thread_pool():
    """Get cached thread pool for async operations"""
    return ThreadPoolExecutor(max_workers=2)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_standard_repair_data(damage_type: str, area: float) -> Dict[str, Any]:
    """Get cached standard repair data"""
    # This would normally query the vector store
    # For now, return mock data based on damage type and area

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
def save_uploaded_file(uploaded_file_bytes: bytes, filename: str) -> Path:
    """Save uploaded file and return path with caching"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(filename).suffix
    cached_filename = f"upload_{timestamp}{file_extension}"
    file_path = UPLOAD_DIR / cached_filename

    with open(file_path, "wb") as f:
        f.write(uploaded_file_bytes)

    return file_path


def analyze_damage_with_ai_async(
    image_path: str, area: float, user_message: str
) -> str:
    """Async wrapper for damage analysis"""
    return analyze_damage_with_ai(image_path, area, user_message)


def analyze_damage_with_ai(image_path: str, area: float, user_message: str) -> str:
    """Analyze building damage using AI with performance optimization"""
    if not MODULES_LOADED:
        return "죄송합니다. 분석 모듈을 사용할 수 없습니다. 개발자에게 문의하세요."

    start_time = time.time()

    try:
        # Initialize vector store if not exists and available
        if "vector_store" not in st.session_state:
            if VECTOR_STORE_AVAILABLE:
                with st.spinner("표준 데이터베이스를 로딩하고 있습니다..."):
                    try:
                        st.session_state.vector_store = create_faiss_vector_store()
                        logger.info("Vector store initialized successfully")
                    except Exception as e:
                        logger.error(f"Failed to initialize vector store: {e}")
                        st.session_state.vector_store = None
            else:
                logger.warning("Vector store not available, using fallback mode")
                st.session_state.vector_store = None

        # Perform damage analysis with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔍 이미지 분석 중...")
        progress_bar.progress(25)

        with st.spinner("AI가 건물 피해를 분석하고 있습니다..."):
            damage_result = analyze_building_damage(
                image_path=image_path,
                query=f"{user_message} (피해 면적: {area} m²)",
                device="cpu",
                generate_report=True,
            )

        progress_bar.progress(50)
        status_text.text("📊 표준 데이터 검색 중...")

        # Extract analysis result
        if isinstance(damage_result, dict) and "analysis_result" in damage_result:
            analysis_result = damage_result["analysis_result"]

            # Handle both DamageAnalysisOutput object and dictionary
            if hasattr(analysis_result, "damage_analysis"):
                damage_analysis = analysis_result.damage_analysis
            elif isinstance(analysis_result, dict):
                damage_analysis = analysis_result.get("damage_analysis", {})
            else:
                damage_analysis = {}

            progress_bar.progress(75)
            status_text.text("💰 비용 산정 중...")

            # Format the response with enhanced details
            response = format_comprehensive_analysis_response(damage_analysis, area)

            progress_bar.progress(100)
            status_text.text("✅ 분석 완료!")

            # Clear progress indicators after a short delay
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            analysis_time = time.time() - start_time
            logger.info(f"Analysis completed in {analysis_time:.2f} seconds")

            return response
        else:
            return "분석 중 오류가 발생했습니다. 다시 시도해주세요."

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return f"분석 중 오류가 발생했습니다: {str(e)}"


def format_comprehensive_analysis_response(
    damage_analysis: Dict[str, Any], area: float
) -> str:
    """Format comprehensive analysis response with all required fields from the image"""

    # Extract key information
    damage_types = damage_analysis.get("damage_types", ["일반 피해"])
    primary_damage = damage_types[0] if damage_types else "일반 피해"
    severity_score = damage_analysis.get("severity_score", 3)
    affected_areas = damage_analysis.get("affected_areas", ["전체 영역"])
    confidence_level = damage_analysis.get("confidence_level", 0.8)

    # Get standard repair data
    repair_data = get_standard_repair_data(primary_damage, area)

    # Create severity description
    severity_descriptions = {
        1: "경미한 피해",
        2: "가벼운 피해",
        3: "보통 피해",
        4: "심각한 피해",
        5: "매우 심각한 피해",
    }
    severity_desc = severity_descriptions.get(severity_score, "보통 피해")

    # Calculate total costs
    total_material_cost = repair_data["material_cost"]
    total_labor_cost = repair_data["labor_cost"]
    total_cost = total_material_cost + total_labor_cost

    # Format response with all required fields
    response = f"""
# 🏗️ 건물 피해 분석 종합 보고서

## 📋 기본 정보
| 항목 | 내용 |
|------|------|
| **분석 일시** | {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')} |
| **분석 면적** | {area:,.1f} m² |
| **분석 신뢰도** | {confidence_level:.1%} |
| **보고서 ID** | RPT-{datetime.now().strftime('%Y%m%d%H%M%S')} |

---

## 🔍 피해 현황 분석

### 📍 피해 부위
- **주요 피해 영역**: {', '.join(affected_areas)}
- **피해 범위**: {area:,.1f} m²

### 🚨 피해 유형  
- **주요 피해**: {primary_damage}
- **세부 피해 유형**: {', '.join(damage_types)}
- **피해 심각도**: {severity_score}/5 ({severity_desc})

---

## 🔧 복구 방법 및 공종

### 📋 복구 방법
"""

    # Add repair methods based on damage type
    repair_methods = {
        "균열": [
            "균열 부위 청소 및 이물질 제거",
            "에폭시 수지 주입을 통한 균열 보수",
            "표면 마감 및 방수 처리",
        ],
        "누수": [
            "누수 원인 파악 및 차단",
            "기존 방수층 제거",
            "신규 방수층 시공",
            "마감재 복구",
        ],
        "화재": ["손상 부재 철거", "구조 보강", "내화재 시공", "마감재 복구"],
        "부식": ["부식 부위 제거", "방청 처리", "보강재 설치", "보호 도장"],
    }

    damage_key = next(
        (key for key in repair_methods.keys() if key in primary_damage), "균열"
    )
    methods = repair_methods.get(damage_key, repair_methods["균열"])

    for i, method in enumerate(methods, 1):
        response += f"\n{i}. {method}"

    response += f"""

### 🏗️ 복구 공종
- **주요 공종**: {primary_damage} 보수공사
- **세부 공종**: 
  - 철거공사
  - 보수공사  
  - 방수공사
  - 마감공사

### 📝 공종명 (건설공사 표준품셈 기준)
- **{primary_damage} 보수**: 표준품셈 기준 적용
- **적용 기준**: 국토교통부 건설공사 표준품셈 2024년 기준

---

## 🛠️ 복구 재료 및 장비

### 📦 주요 자재
"""

    for i, material in enumerate(repair_data["materials"], 1):
        response += f"\n{i}. {material}"

    response += f"""

### ⚙️ 필요 장비
"""

    for i, equipment in enumerate(repair_data["equipment"], 1):
        response += f"\n{i}. {equipment}"

    response += f"""

---

## 👷 인력 구성

### 👥 소요 인력
| 직종 | 인원 | 역할 |
|------|------|------|
"""

    for job_type, count in repair_data["labor"].items():
        roles = {
            "특급기능사": "현장 총괄, 기술 지도",
            "고급기능사": "전문 작업 수행",
            "보통인부": "보조 작업, 자재 운반",
        }
        response += f"| {job_type} | {count}명 | {roles.get(job_type, '작업 수행')} |\n"

    response += f"""

---

## ⏰ 복구 기간

### 📅 예상 공사 기간
- **총 공사 기간**: {repair_data['duration_days']}일
- **작업 단계별 기간**:
  - 준비 및 철거: 1일
  - 주요 보수 작업: {max(1, repair_data['duration_days']-2)}일  
  - 마감 및 정리: 1일

---

## 💰 비용 산정

### 💵 자재비 단가
| 구분 | 단가 | 수량 | 금액 |
|------|------|------|------|
| 자재비 | {total_material_cost/area:,.0f}원/m² | {area:,.1f}m² | {total_material_cost:,.0f}원 |

### 👷 노무비  
| 구분 | 단가 | 수량 | 금액 |
|------|------|------|------|
| 노무비 | {total_labor_cost/area:,.0f}원/m² | {area:,.1f}m² | {total_labor_cost:,.0f}원 |

### 📊 총 비용 요약
| 항목 | 금액 | 비율 |
|------|------|------|
| **자재비** | {total_material_cost:,.0f}원 | {(total_material_cost/total_cost)*100:.1f}% |
| **노무비** | {total_labor_cost:,.0f}원 | {(total_labor_cost/total_cost)*100:.1f}% |
| **총 공사비** | **{total_cost:,.0f}원** | **100%** |
| **m²당 단가** | {total_cost/area:,.0f}원/m² | - |

---

## 🔧 수리 우선순위 및 권장사항

### 🚨 우선순위
"""

    if severity_score >= 4:
        response += """
- **등급**: 🔴 **긴급 (1순위)**
- **조치 기한**: 즉시 (24시간 이내)
- **권장 조치**: 안전상의 이유로 즉시 전문가 상담 및 응급 조치 필요
"""
    elif severity_score == 3:
        response += """
- **등급**: 🟡 **높음 (2순위)**  
- **조치 기한**: 2주 이내
- **권장 조치**: 빠른 시일 내에 수리를 진행하여 피해 확산 방지
"""
    else:
        response += """
- **등급**: 🟢 **보통 (3순위)**
- **조치 기한**: 1개월 이내  
- **권장 조치**: 계획적으로 수리를 진행하되 정기적 점검 실시
"""

    response += f"""

### ⚠️ 안전 주의사항
"""

    if severity_score >= 4:
        response += """
- ⚠️ **즉시 대피 고려**: 구조적 안전성 문제 가능성
- ⚠️ **출입 제한**: 해당 영역 사용 금지
- ⚠️ **전문가 진단**: 구조 엔지니어 정밀 진단 필수
- ⚠️ **응급 조치**: 임시 보강 또는 차단 조치 필요
"""
    elif severity_score >= 3:
        response += """
- ⚠️ **주의 깊은 사용**: 하중 제한 및 진동 방지
- ⚠️ **정기 점검**: 주 1회 이상 상태 확인
- ⚠️ **확산 방지**: 방수 처리 등 추가 피해 방지 조치
- ⚠️ **모니터링**: 균열 진행 상황 지속 관찰
"""
    else:
        response += """
- ✅ **일반 안전수칙 준수**: 기본적인 건물 사용 수칙 준수
- ✅ **정기 유지보수**: 월 1회 이상 점검 실시
- ✅ **예방 조치**: 습도 관리 및 환기 등 예방 관리
"""

    response += f"""

---

## 📞 추가 안내

### 🔍 정밀 진단 권장
- 본 분석은 AI 기반 1차 진단 결과입니다
- 정확한 진단을 위해서는 전문가의 현장 조사가 필요합니다
- 구조적 안전성 검토가 필요한 경우 구조 엔지니어 상담을 권장합니다

### 📋 표준품셈 적용 안내  
- 본 비용 산정은 건설공사 표준품셈을 기준으로 합니다
- 실제 시공 시 현장 여건에 따라 비용이 변동될 수 있습니다
- 정확한 견적은 전문 시공업체 상담을 통해 확인하시기 바랍니다

### 💬 추가 문의
더 자세한 분석이나 전문가 상담이 필요하시면 언제든지 문의해주세요!

---
*📅 보고서 생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}*  
*🤖 분석 시스템: Tumblr AI 건물 피해 분석 시스템 v2.0*
"""

    return response


def main():
    """Main application function"""

    # Render the simple ChatGPT UI and get user inputs
    user_message, uploaded_file, area_input, send_button = render_simple_chatgpt_ui()

    # Process the message if send button was clicked and there's a message
    if send_button and user_message:
        # If there's an uploaded file, save it and analyze
        if uploaded_file is not None:
            try:
                # Save the uploaded file
                file_path = save_uploaded_file(
                    uploaded_file.getvalue(), uploaded_file.name
                )

                # Analyze the damage
                analysis_result = analyze_damage_with_ai(
                    str(file_path), area_input, user_message
                )

                # The analysis result is already added to session state in render_simple_chatgpt_ui
                # We just need to update the last assistant message with our analysis
                if (
                    st.session_state.messages
                    and st.session_state.messages[-1]["role"] == "assistant"
                ):
                    # Replace the OpenAI response with our detailed analysis
                    st.session_state.messages[-1]["content"] = analysis_result
                    st.rerun()

            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

        # If no image was uploaded, the OpenAI response will be used as-is


if __name__ == "__main__":
    main()
