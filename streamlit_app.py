"""
Simple ChatGPT-style Streamlit web application for building damage analysis
"""

import streamlit as st
import logging

# Import UI components
from ui.ui_components import render_simple_chatgpt_ui

# Import app modules
from app.config import initialize_modules, setup_directories, get_app_config
from app.data_processor import (
    save_uploaded_file,
    validate_uploaded_file,
    validate_area_input,
)
from app.analysis_engine import analyze_damage_with_ai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """메인 애플리케이션 함수"""

    # Initialize modules and setup
    modules = initialize_modules()
    upload_dir, results_dir = setup_directories()
    app_config = get_app_config()

    # Render the simple ChatGPT UI and get user inputs
    user_message, uploaded_file, area_input = render_simple_chatgpt_ui()

    # Process the message if user submitted a message via chat_input
    if user_message:

        # If there's an uploaded file, validate and analyze
        if uploaded_file is not None:

            # Validate uploaded file
            file_valid, file_message = validate_uploaded_file(uploaded_file)
            if not file_valid:
                st.error(file_message)
                return

            # Validate area input
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
                analysis_result = analyze_damage_with_ai(
                    image_path=str(file_path),
                    area=area_input,
                    user_message=user_message,
                    analyze_building_damage_func=modules["analyze_building_damage"],
                    create_faiss_vector_store_func=modules["create_faiss_vector_store"],
                    modules_loaded=app_config["modules_loaded"],
                    vector_store_available=app_config["vector_store_available"],
                )

                # Update the chat with the analysis result
                if (
                    st.session_state.messages
                    and st.session_state.messages[-1]["role"] == "assistant"
                ):
                    # Replace the OpenAI response with our detailed analysis
                    st.session_state.messages[-1]["content"] = analysis_result
                    st.rerun()

            except Exception as e:
                logger.error(f"Error processing file: {e}")
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")

        # If no image was uploaded, the OpenAI response will be used as-is


if __name__ == "__main__":
    main()
