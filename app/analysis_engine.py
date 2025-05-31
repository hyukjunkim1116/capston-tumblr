"""
AI 분석 엔진 모듈
건물 피해 분석 AI 처리 로직
"""

import streamlit as st
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def initialize_vector_store(
    create_faiss_vector_store_func, vector_store_available: bool
):
    """벡터 스토어 초기화"""
    if "vector_store" not in st.session_state:
        if vector_store_available:
            with st.spinner("표준 데이터베이스를 로딩하고 있습니다..."):
                try:
                    st.session_state.vector_store = create_faiss_vector_store_func()
                    logger.info("Vector store initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize vector store: {e}")
                    st.session_state.vector_store = None
        else:
            logger.warning("Vector store not available, using fallback mode")
            st.session_state.vector_store = None


def analyze_damage_with_ai(
    image_path: str,
    area: float,
    user_message: str,
    analyze_building_damage_func,
    create_faiss_vector_store_func,
    modules_loaded: bool,
    vector_store_available: bool,
) -> str:
    """AI를 사용한 건물 피해 분석 (성능 최적화)"""

    if not modules_loaded:
        return "죄송합니다. 분석 모듈을 사용할 수 없습니다. 개발자에게 문의하세요."

    start_time = time.time()

    try:
        # Initialize vector store if not exists and available
        initialize_vector_store(create_faiss_vector_store_func, vector_store_available)

        # Perform damage analysis with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔍 이미지 분석 중...")
        progress_bar.progress(25)

        with st.spinner("AI가 건물 피해를 분석하고 있습니다..."):
            damage_result = analyze_building_damage_func(
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

            # Import report formatter here to avoid circular imports
            from .report_formatter import format_comprehensive_analysis_response

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


def get_analysis_progress_status():
    """분석 진행 상태 반환"""
    return {
        "steps": [
            {"name": "이미지 로딩", "progress": 25},
            {"name": "AI 분석", "progress": 50},
            {"name": "표준 데이터 검색", "progress": 75},
            {"name": "보고서 생성", "progress": 100},
        ]
    }
