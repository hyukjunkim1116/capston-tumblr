"""
Simple ChatGPT-style Streamlit web application for building damage analysis
"""

import streamlit as st
import os

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any

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

    def create_faiss_vector_store():
        return None

    def analyze_building_damage(*args, **kwargs):
        return {"error": "분석 모듈을 사용할 수 없습니다."}


# Create directories
UPLOAD_DIR = CACHE_DIR / "uploads"
RESULTS_DIR = CACHE_DIR / "results"

for directory in [UPLOAD_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file and return path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(uploaded_file.name).suffix
    filename = f"upload_{timestamp}{file_extension}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def analyze_damage_with_ai(image_path: str, area: float, user_message: str) -> str:
    """Analyze building damage using AI"""
    if not MODULES_LOADED:
        return "죄송합니다. 분석 모듈을 사용할 수 없습니다. 개발자에게 문의하세요."

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

        # Perform damage analysis
        with st.spinner("AI가 건물 피해를 분석하고 있습니다..."):
            damage_result = analyze_building_damage(
                image_path=image_path,
                query=f"{user_message} (피해 면적: {area} m²)",
                device="cpu",
                generate_report=True,
            )

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

            # Format the response
            response = format_analysis_response(damage_analysis, area)
            return response
        else:
            return "분석 중 오류가 발생했습니다. 다시 시도해주세요."

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return f"분석 중 오류가 발생했습니다: {str(e)}"


def format_analysis_response(damage_analysis: Dict[str, Any], area: float) -> str:
    """Format the analysis response for display"""

    # Extract key information
    damage_types = damage_analysis.get("damage_types", ["일반 피해"])
    severity_score = damage_analysis.get("severity_score", 3)
    affected_areas = damage_analysis.get("affected_areas", ["전체 영역"])
    confidence_level = damage_analysis.get("confidence_level", 0.8)

    # Create severity description
    severity_descriptions = {
        1: "경미한 피해",
        2: "가벼운 피해",
        3: "보통 피해",
        4: "심각한 피해",
        5: "매우 심각한 피해",
    }
    severity_desc = severity_descriptions.get(severity_score, "보통 피해")

    # Estimate repair cost (simplified calculation)
    base_cost_per_sqm = {
        1: 30000,  # 경미한 피해
        2: 50000,  # 가벼운 피해
        3: 80000,  # 보통 피해
        4: 120000,  # 심각한 피해
        5: 200000,  # 매우 심각한 피해
    }

    estimated_cost = base_cost_per_sqm.get(severity_score, 80000) * area

    # Format response
    response = f"""
## 🏠 건물 피해 분석 결과

### 📊 기본 정보
- **분석 면적**: {area} m²
- **신뢰도**: {confidence_level:.1%}

### 🔍 피해 분석
- **주요 피해 유형**: {', '.join(damage_types)}
- **피해 심각도**: {severity_score}/5 ({severity_desc})
- **영향 받은 영역**: {', '.join(affected_areas)}

### 💰 예상 수리 비용
- **총 예상 비용**: {estimated_cost:,}원
- **평방미터당 비용**: {estimated_cost/area:,.0f}원/m²

### 🔧 수리 우선순위
"""

    if severity_score >= 4:
        response += """
- **우선순위**: 🔴 **긴급** - 즉시 수리 필요
- **권장 조치**: 안전상의 이유로 즉시 전문가 상담을 받으시기 바랍니다.
"""
    elif severity_score == 3:
        response += """
- **우선순위**: 🟡 **높음** - 2주 내 수리 권장
- **권장 조치**: 빠른 시일 내에 수리를 진행하시기 바랍니다.
"""
    else:
        response += """
- **우선순위**: 🟢 **보통** - 1개월 내 수리 권장
- **권장 조치**: 계획적으로 수리를 진행하시면 됩니다.
"""

    response += f"""

### ⚠️ 안전 주의사항
"""

    if severity_score >= 4:
        response += """
- 구조적 안전성에 문제가 있을 수 있습니다
- 해당 영역의 사용을 제한하시기 바랍니다
- 전문가의 정밀 진단을 받으시기 바랍니다
"""
    elif severity_score >= 3:
        response += """
- 피해가 확산되지 않도록 주의하시기 바랍니다
- 정기적인 점검을 실시하시기 바랍니다
- 필요시 임시 보강 조치를 취하시기 바랍니다
"""
    else:
        response += """
- 일반적인 안전 수칙을 준수하시기 바랍니다
- 정기적인 유지보수를 실시하시기 바랍니다
"""

    response += """

### 📞 추가 도움
더 자세한 분석이나 전문가 상담이 필요하시면 언제든지 문의해주세요!
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
                file_path = save_uploaded_file(uploaded_file)

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
