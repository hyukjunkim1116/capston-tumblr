"""
Simple ChatGPT-style Streamlit web application for building damage analysis
환경별 성능 최적화 적용
"""

import streamlit as st
import logging
from datetime import datetime
import gc
import os

# 메모리 모니터링을 위한 import (선택적)
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 정보 로깅
logger.info(f"🚀 앱 시작 - 환경: {app_config['environment']}")
logger.info(
    f"📊 설정: {app_config['device']}, 이미지 최대 크기: {app_config['max_image_size']}"
)


# 메모리 사용량 체크 함수
def log_memory_usage(stage: str):
    """메모리 사용량을 로그에 기록"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"{stage} - 메모리 사용량: {memory_mb:.1f}MB")
            return memory_mb
        except Exception as e:
            logger.debug(f"메모리 모니터링 오류: {e}")
            return 0
    return 0


def cleanup_memory_if_needed():
    """메모리 사용량이 높으면 정리"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            # 800MB 이상 사용 시 정리 (성능 우선으로 임계값 높임)
            if memory_mb > 800:
                logger.warning(f"메모리 사용량 높음({memory_mb:.1f}MB), 정리 실행")

                # 세션 메시지 정리 (적당히)
                if (
                    "messages" in st.session_state
                    and len(st.session_state.messages) > 20
                ):
                    st.session_state.messages = st.session_state.messages[
                        -10:
                    ]  # 최신 10개 유지
                    logger.info("세션 메시지 정리 (최신 10개 유지)")

                # 가비지 컬렉션
                gc.collect()

                # PyTorch 메모리 정리
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass

                # 임시 파일 정리
                cleanup_temp_files()

                logger.info("메모리 정리 완료")

        except Exception as e:
            logger.debug(f"메모리 정리 오류: {e}")


def cleanup_temp_files():
    """임시 파일들 정리"""
    try:
        import tempfile
        import shutil
        from pathlib import Path

        # 임시 디렉토리에서 오래된 파일들 삭제 (2시간 이상)
        temp_dir = Path(tempfile.gettempdir())

        current_time = datetime.now().timestamp()
        for file_path in temp_dir.glob("tmp*"):
            try:
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > 7200:  # 2시간
                        file_path.unlink()
            except Exception:
                continue

        # 캐시 디렉토리 정리 (4시간 이상)
        cache_dir = Path("./cache")
        if cache_dir.exists():
            for file_path in cache_dir.rglob("*"):
                try:
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > 14400:  # 4시간
                            file_path.unlink()
                except Exception:
                    continue

    except Exception as e:
        logger.debug(f"임시 파일 정리 오류: {e}")


# Initialize analysis engine with optimization
@st.cache_resource
def get_analysis_engine():
    """캐시된 분석 엔진 인스턴스"""
    return AnalysisEngine()


analysis_engine = get_analysis_engine()


def main():
    """메인 애플리케이션 함수 (성능 우선)"""

    upload_dir, _ = setup_directories()

    # 세션 상태 초기화 (안전하게)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    # 위젯 상태 초기화 (UI 위젯들을 위한 안전한 초기화)
    if "area_input" not in st.session_state:
        st.session_state.area_input = 100.0
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    # 앱 시작 시 메모리 정리 (적당히)
    cleanup_memory_if_needed()

    # 세션 상태 메모리 관리 (적당히)
    if "messages" in st.session_state and len(st.session_state.messages) > 30:
        # 오래된 메시지 제거 (최신 15개 유지)
        st.session_state.messages = st.session_state.messages[-15:]
        logger.info("세션 메시지 정리 완료 (최신 15개 유지)")

    # Render the simple ChatGPT UI and get user inputs
    user_input, area_input, uploaded_file = render_simple_chatgpt_ui()

    # 새로운 파일이 업로드되었는지 확인
    current_file_info = None
    if uploaded_file is not None:
        current_file_info = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "type": uploaded_file.type,
        }

    # 이미 처리된 파일인지 확인
    is_same_file = (
        current_file_info is not None
        and st.session_state.last_processed_file is not None
        and current_file_info == st.session_state.last_processed_file
    )

    # Process text input
    if user_input:
        # 사용자 메시지를 세션에 추가
        message_data = {
            "role": "user",
            "content": str(user_input),
            "timestamp": datetime.now(),
            "area": area_input,
        }

        # 이미지가 있으면 추가
        if uploaded_file is not None:
            # 이미지 크기 확인용으로만 사용
            image_size = len(uploaded_file.getvalue())
            message_data["image_name"] = uploaded_file.name
            message_data["image_size"] = image_size

        st.session_state.messages.append(message_data)

        # 이미지와 텍스트가 모두 있는 경우만 분석 실행
        if (
            uploaded_file is not None
            and not is_same_file
            and not st.session_state.analysis_running
        ):
            st.session_state.analysis_running = True  # 분석 시작 플래그 설정
            st.session_state.processing_done = True  # 처리 완료 플래그 설정

            # 1. 파일 형식 및 크기 검증
            file_valid, file_message = validate_uploaded_file(uploaded_file)
            if not file_valid:
                st.error(file_message)
                st.session_state.analysis_running = False  # 분석 완료 플래그 해제
                st.stop()  # rerun 대신 stop 사용

            # 2. 이미지 내용 검증 (건물인지 확인)
            content_valid, content_message = validate_image_content(uploaded_file)
            if not content_valid:
                st.error(content_message)
                st.session_state.analysis_running = False  # 분석 완료 플래그 해제
                st.stop()

            # 3. 면적 입력 검증
            area_valid, area_message = validate_area_input(area_input)
            if not area_valid:
                st.error(area_message)
                st.session_state.analysis_running = False  # 분석 완료 플래그 해제
                st.stop()

            try:
                # 분석 진행
                with st.spinner("분석 진행 중..."):
                    memory_before = log_memory_usage("분석 시작 전")

                    # 분석 전 메모리 정리 (적당히)
                    if memory_before > 600:
                        cleanup_memory_if_needed()

                    # Save the uploaded file
                    file_path = save_uploaded_file(
                        uploaded_file.getvalue(), uploaded_file.name, upload_dir
                    )

                    # 환경별 최적화된 분석 실행
                    analysis_result = analysis_engine.generate_comprehensive_analysis(
                        image_path=str(file_path),
                        area=area_input,
                        user_message=str(user_input),
                    )

                    memory_after = log_memory_usage("분석 완료 후")

                    # 분석 후 메모리 정리 (적당히)
                    if memory_after > 800:
                        cleanup_memory_if_needed()

                if analysis_result["success"]:
                    # 분석 성공
                    response_content = analysis_result["analysis_text"]

                    # PDF 데이터 준비 (구조화된 데이터 포함)
                    pdf_data = {
                        "analysis_result": analysis_result["analysis_text"],
                        "image_path": str(file_path),
                        "area": area_input,
                        "user_message": str(user_input),
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

                    # 메모리 정리 (적당히)
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

                    # 처리된 파일 정보 저장
                    if current_file_info:
                        st.session_state.last_processed_file = current_file_info

                    # 분석 완료 처리
                    st.session_state.analysis_running = False
                    st.success("분석이 완료되었습니다!")

                    # 화면 새로고침으로 결과 표시
                    st.rerun()

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
                    st.session_state.analysis_running = False
                    st.rerun()

            except Exception as e:
                logger.error(f"Error processing file: {e}")
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                st.session_state.analysis_running = False
                st.session_state.processing_done = False

        elif uploaded_file is None:
            # 텍스트만 있는 경우 안내 메시지
            text_response = """
**안내**: 더 정확한 건물 피해 분석을 위해서는 피해 사진을 첨부해주세요. 
위의 파일 업로드 버튼을 사용하여 이미지를 업로드한 후 다시 메시지를 전송해주세요.

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
            st.info("이미지를 업로드하고 메시지를 전송하면 AI 분석을 시작합니다.")

        elif is_same_file:
            st.info(
                "이 파일은 이미 분석되었습니다. 새로운 파일을 업로드하거나 다른 질문을 입력해주세요."
            )

    # 이미지만 업로드되고 메시지가 없는 경우
    elif uploaded_file is not None and not user_input:
        st.info("이미지가 업로드되었습니다. 분석 요청 메시지를 입력하고 전송해주세요.")

    # 분석이 실행 중인 경우 상태 표시
    elif st.session_state.analysis_running:
        st.info("분석이 진행 중입니다. 잠시만 기다려주세요...")

    # 새 세션 시작을 위한 버튼 추가
    if st.session_state.processing_done:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("새로운 분석 시작", type="secondary"):
                # 상태 초기화
                st.session_state.processing_done = False
                st.session_state.analysis_running = False
                st.session_state.last_processed_file = None
                st.success("새로운 분석을 시작할 수 있습니다!")
                # rerun 없이 상태만 초기화


if __name__ == "__main__":
    main()
