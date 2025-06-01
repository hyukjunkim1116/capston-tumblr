"""
Simple ChatGPT-style Streamlit web application for building damage analysis
환경별 성능 최적화 적용
"""

import streamlit as st
import logging
from datetime import datetime

# Import UI components
from ui.ui_components import render_simple_chatgpt_ui

# Import app modules with optimization
from app.config import setup_directories, get_app_config, optimize_for_environment
from app.data_processor import (
    save_uploaded_file,
    validate_uploaded_file,
    validate_area_input,
    validate_image_content,
)
from app.analysis_engine import AnalysisEngine

# 환경별 최적화 적용
optimize_for_environment()

# 설정 정보 확인
app_config = get_app_config()

# Setup logging
logging.basicConfig(
    level=logging.WARNING if app_config["is_deployment"] else logging.INFO
)
logger = logging.getLogger(__name__)

# 환경 정보 로깅
logger.info(f"🚀 앱 시작 - 환경: {app_config['environment']}")
logger.info(
    f"📊 설정: {app_config['device']}, 이미지 최대 크기: {app_config['max_image_size']}"
)


# Initialize analysis engine with optimization
@st.cache_resource
def get_analysis_engine():
    """캐시된 분석 엔진 인스턴스"""
    return AnalysisEngine()


analysis_engine = get_analysis_engine()


def main():
    """메인 애플리케이션 함수"""

    upload_dir, _ = setup_directories()

    # Render the simple ChatGPT UI and get user inputs
    user_input, area_input, uploaded_file = render_simple_chatgpt_ui()

    # Process text input
    if user_input:
        user_message = str(user_input)

        # 사용자 메시지를 항상 세션에 추가
        message_data = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(),
            "area": area_input,
        }

        # 이미지가 있으면 추가
        if uploaded_file is not None:
            message_data["image"] = uploaded_file
            message_data["image_size"] = len(uploaded_file.getvalue())

        st.session_state.messages.append(message_data)

        # If there's an uploaded file, validate and analyze
        if uploaded_file is not None:

            # 1. 파일 형식 및 크기 검증
            file_valid, file_message = validate_uploaded_file(uploaded_file)
            if not file_valid:
                st.error(file_message)
                return

            # 2. 이미지 내용 검증 (건물인지 확인)
            content_valid, content_message = validate_image_content(uploaded_file)
            if not content_valid:
                st.error(content_message)
                return

            # 3. 면적 입력 검증
            area_valid, area_message = validate_area_input(area_input)
            if not area_valid:
                st.error(area_message)
                return

            try:
                # 분석 진행
                with st.spinner("분석 진행 중..."):
                    # Save the uploaded file
                    file_path = save_uploaded_file(
                        uploaded_file.getvalue(), uploaded_file.name, upload_dir
                    )

                    # 환경별 최적화된 분석 실행
                    analysis_result = analysis_engine.generate_comprehensive_analysis(
                        image_path=str(file_path),
                        area=area_input,
                        user_message=user_message,
                    )

                if analysis_result["success"]:
                    # 분석 성공
                    response_content = analysis_result["analysis_text"]

                    # PDF 데이터 준비 (구조화된 데이터 포함)
                    pdf_data = {
                        "analysis_result": analysis_result["analysis_text"],
                        "image_path": str(file_path),
                        "area": area_input,
                        "user_message": user_message,
                        "damage_areas": analysis_result["damage_areas"],
                    }

                    # Add the analysis result as assistant message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response_content,
                            "timestamp": datetime.now(),
                            "response_type": "comprehensive",
                            "pdf_data": pdf_data,
                        }
                    )

                    logger.info("분석 완료")

                else:
                    # 분석 실패
                    error_message = analysis_result.get(
                        "error", "알 수 없는 오류가 발생했습니다."
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"❌ 분석 중 오류가 발생했습니다: {error_message}",
                            "timestamp": datetime.now(),
                            "response_type": "error",
                        }
                    )

                st.rerun()

            except Exception as e:
                logger.error(f"❌ Error processing file: {e}")
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

        else:
            # 텍스트만 있는 경우
            text_response = """

**안내**: 더 정확한 건물 피해 분석을 위해서는 피해 사진을 첨부해주세요. 
위의 파일 업로드 버튼을 사용하여 이미지를 업로드할 수 있습니다.

**일반적인 건물 피해 분석 정보**:
- 구조적 균열: 벽체, 기둥, 보의 균열 확인
- 누수 피해: 지붕, 벽체 침수 여부 점검  
- 외부 마감재: 외벽, 창호 손상 확인
- 설비 피해: 전기, 배관 시설 점검
"""

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": text_response,
                    "timestamp": datetime.now(),
                    "response_type": "text_only",
                }
            )
            st.rerun()

    # 파일이 업로드된 경우에도 처리
    elif uploaded_file is not None:
        # 파일만 업로드된 경우 기본 메시지와 함께 처리
        user_message = "업로드된 이미지를 분석해주세요."

        message_data = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(),
            "area": area_input,
            "image": uploaded_file,
            "image_size": len(uploaded_file.getvalue()),
        }

        st.session_state.messages.append(message_data)

        # 파일 검증 및 분석 진행 (위와 동일한 로직)
        file_valid, file_message = validate_uploaded_file(uploaded_file)
        if not file_valid:
            st.error(file_message)
            return

        content_valid, content_message = validate_image_content(uploaded_file)
        if not content_valid:
            st.error(content_message)
            return

        area_valid, area_message = validate_area_input(area_input)
        if not area_valid:
            st.error(area_message)
            return

        try:
            with st.spinner("분석 진행 중..."):
                file_path = save_uploaded_file(
                    uploaded_file.getvalue(), uploaded_file.name, upload_dir
                )

                analysis_result = analysis_engine.generate_comprehensive_analysis(
                    image_path=str(file_path),
                    area=area_input,
                    user_message=user_message,
                )

            if analysis_result["success"]:
                response_content = analysis_result["analysis_text"]

                pdf_data = {
                    "analysis_result": analysis_result["analysis_text"],
                    "image_path": str(file_path),
                    "area": area_input,
                    "user_message": user_message,
                    "damage_areas": analysis_result["damage_areas"],
                }

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.now(),
                        "response_type": "comprehensive",
                        "pdf_data": pdf_data,
                    }
                )

                logger.info("분석 완료")
            else:
                error_message = analysis_result.get(
                    "error", "알 수 없는 오류가 발생했습니다."
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ 분석 중 오류가 발생했습니다: {error_message}",
                        "timestamp": datetime.now(),
                        "response_type": "error",
                    }
                )

            st.rerun()

        except Exception as e:
            logger.error(f"❌ Error processing file: {e}")
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    main()
