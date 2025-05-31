"""
Simple ChatGPT-style Streamlit web application for building damage analysis
"""

import streamlit as st
import logging
from datetime import datetime

# Import UI components
from ui.ui_components import render_simple_chatgpt_ui

# Import app modules
from app.config import setup_directories
from app.data_processor import (
    save_uploaded_file,
    validate_uploaded_file,
    validate_area_input,
    validate_image_content,
)
from app.analysis_engine import AnalysisEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize analysis engine
analysis_engine = AnalysisEngine()


def main():
    """메인 애플리케이션 함수"""

    upload_dir, _ = setup_directories()

    # Render the simple ChatGPT UI and get user inputs
    user_input, area_input = render_simple_chatgpt_ui()

    # Process the message if user submitted input via chat_input
    if user_input:

        # Extract message and file from user_input (dict-like object)
        if hasattr(user_input, "text") and hasattr(user_input, "files"):
            user_message = user_input.text if user_input.text else ""
            uploaded_files = user_input.files if user_input.files else []
            uploaded_file = uploaded_files[0] if uploaded_files else None
        else:
            # user_input이 문자열인 경우 (파일 없는 경우)
            user_message = str(user_input)
            uploaded_file = None

        # 사용자 메시지를 항상 세션에 추가 (파일 여부와 관계없이)
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
                # 사용자 메시지는 이미 추가되었으므로 오류 메시지만 추가
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": content_message,
                        "timestamp": datetime.now(),
                        "response_type": "validation_error",
                    }
                )
                st.rerun()

            # 3. 면적 입력 검증
            area_valid, area_message = validate_area_input(area_input)
            if not area_valid:
                st.error(area_message)
                return

            try:
                # Save the uploaded file
                file_path = save_uploaded_file(
                    uploaded_file.getvalue(), uploaded_file.name, upload_dir
                )

                # Analyze the damage using the modular analysis engine
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

                    logger.info(
                        f"분석 완료 - 처리 시간: {analysis_result.get('processing_time', 0):.2f}초"
                    )

                else:
                    # 분석 실패
                    error_message = analysis_result.get(
                        "error", "알 수 없는 오류가 발생했습니다."
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"분석 중 오류가 발생했습니다: {error_message}",
                            "timestamp": datetime.now(),
                            "response_type": "error",
                        }
                    )

                st.rerun()

            except Exception as e:
                logger.error(f"Error processing file: {e}")
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

        else:
            # 텍스트만 있는 경우 - 기본 응답 제공
            text_response = f"""

**안내**: 더 정확한 건물 피해 분석을 위해서는 피해 사진을 첨부해주세요. 
파일 첨부 아이콘(📎)을 클릭하거나 이미지를 채팅창에 드래그 앤 드롭하여 업로드할 수 있습니다.

**일반적인 건물 피해 분석 정보**:
- 구조적 균열: 벽체, 기둥, 보의 균열 확인
- 누수 피해: 지붕, 벽체 침수 여부 점검  
- 외부 마감재: 외벽, 창호 손상 확인
- 설비 피해: 전기, 배관 시설 점검"""

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": text_response,
                    "timestamp": datetime.now(),
                    "response_type": "text_only",
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()
