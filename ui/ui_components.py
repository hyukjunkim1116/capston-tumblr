import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

# .env 파일에서 환경변수 로드
load_dotenv()

# OpenAI 클라이언트 지연 초기화 (사용할 때만 로드)
client = None
openai_available = None


def get_openai_client():
    """OpenAI 클라이언트를 지연 로딩으로 가져오기"""
    global client, openai_available

    if openai_available is not None:
        return client, openai_available

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key, timeout=30.0, max_retries=2)
            openai_available = True
            return client, True
        else:
            openai_available = False
            return None, False
    except Exception as e:
        st.error(f"❌ OpenAI 클라이언트 초기화 실패: {str(e)}")
        openai_available = False
        return None, False


# Performance optimization: Cache API responses
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_openai_response(messages_hash: str, messages: list) -> str:
    """Get cached OpenAI response"""
    client, available = get_openai_client()

    if not available or client is None:
        return "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다."

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"


def render_simple_chatgpt_ui():
    """최신 ChatGPT 스타일 UI 렌더링"""

    # 최소한의 CSS - 기본 스타일링
    st.markdown(
        """
    <style>
  
    
    .main > .block-container {
        max-width: 900px !important;
        margin: 0 auto !important;
        background-color: #212121 !important;
    }
    
    
    /* 타이틀 */
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0;
    }
    
    /* 사이드바만 다른 색상 */
    section[data-testid="stSidebar"] {
        background: #212121 !important;
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stAlert {
        background: linear-gradient(135deg, #451a03 0%, #78350f 100%) !important;
        border: 1px solid #92400e !important;
        color: #fef3c7 !important;
    }
    
    /* 모든 텍스트 색상 통일 */
    .stMarkdown, .stText, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #e5e7eb !important;
    }
    
    .stChatInput > div > div > input {
        background-color: #303030 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 8px !important;
        color: #e5e7eb !important;
        padding: 0.75rem !important;
    }
    
    .stChatInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        outline: none !important;
    }
    
    .stChatInput > div > div > input::placeholder {
        color: #9ca3af !important;
    }
    
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Tumblr AI 타이틀
    st.markdown('<div class="main-title">Tumblr AI</div>', unsafe_allow_html=True)

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # 채팅 메시지 표시
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar=None):
                st.write(message["content"])

                # 이미지가 있으면 표시
                if "image" in message:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(
                            message["image"],
                            caption="업로드된 이미지",
                            use_container_width=True,
                        )

        else:
            response_type = message.get("response_type", "comprehensive")
            avatar = None

            with st.chat_message("assistant", avatar=avatar):
                st.write(message["content"])
                if response_type == "comprehensive" and "pdf_data" in message:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        try:
                            # PDF 생성기 import (지연 import로 오류 방지)
                            from app.pdf_generator import create_damage_report_pdf

                            pdf_data = message["pdf_data"]

                            # PDF 생성
                            with st.spinner("PDF 보고서 생성 중..."):
                                pdf_buffer = create_damage_report_pdf(
                                    analysis_result=pdf_data["analysis_result"],
                                    image_path=pdf_data["image_path"],
                                    area=pdf_data["area"],
                                    user_message=pdf_data["user_message"],
                                    damage_areas=pdf_data.get("damage_areas", []),
                                )

                            # 파일명 생성
                            timestamp = message["timestamp"].strftime("%Y%m%d_%H%M%S")
                            filename = f"건물손상분석보고서_{timestamp}.pdf"

                            # 바로 다운로드 버튼 (클릭하면 즉시 다운로드)
                            st.download_button(
                                label="PDF 보고서 다운로드",
                                data=pdf_buffer.getvalue(),
                                file_name=filename,
                                mime="application/pdf",
                                key=f"download_btn_{message['timestamp']}",
                                type="primary",
                            )

                        except Exception as e:
                            st.error(f"PDF 생성 중 오류가 발생했습니다: {str(e)}")
                            st.info(
                                "분석 결과는 위의 텍스트 형태로 확인하실 수 있습니다."
                            )

    # 파일 업로드를 위한 별도 컴포넌트
    uploaded_file = st.file_uploader(
        "건물 손상 이미지 업로드",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="분석할 건물 손상 이미지를 업로드하세요",
        disabled=st.session_state.processing,
        key="image_uploader",
    )

    # st.chat_input을 사용한 메시지 입력 (하단에 자동 고정됨)
    user_input = st.chat_input(
        placeholder="건물 손상에 대한 질문이나 요청사항을 입력하세요...",
        disabled=st.session_state.processing,
        key="main_chat_input",
    )

    # 별도 입력 필드들 제거하고 chat_input만 사용
    area_input = st.number_input(
        "피해 면적 (m²)",
        min_value=0.1,
        max_value=1000000.0,
        value=10.0,
        step=0.1,
        help="분석할 건물의 면적을 입력하세요",
        disabled=st.session_state.processing,
    )

    with st.sidebar:
        st.markdown("### 공종명 대체 적용 안내")

        st.markdown(
            """
        본 분석에 사용된 공종명은 **건설공사 표준품셈**을 기준으로 하되, 실제 피해 상황과 1:1로 정확히 매칭되지 않는 일부 공종에 대해서는 다음과 같은 기준에 따라 유사하거나 기능적으로 대응 가능한 공종으로 대체 적용하였습니다.
        """
        )

        st.markdown("#### 대체 기준")
        st.markdown(
            """
        - **구조적 유사성**: 동일한 구조적 특성을 가진 공종
        - **시공 방법의 유사성**: 비슷한 시공 절차와 방법  
        - **사용 재료 및 작업 목적의 유사성**: 동일한 재료와 목적
        
        등을 종합적으로 고려하여 판단하였습니다.
        """
        )

        st.markdown("#### 대체 적용 예시")
        st.markdown(
            """
        • **금속 외장 패널** → 경량벽체 철골틀 설치
        
        • **금속기와 파손** → 금속기와 잇기
        
        • **방수층 파손** → 기존 방수층 제거 및 바탕처리 + 방수재 도포
        """
        )

        st.markdown("#### 복구 방법 선택 안내")
        st.markdown(
            """
        동일한 피해 항목에 대해 복수의 공종명 또는 인력 구성을 적용한 사례를 포함합니다. 사용자는 해당 작업의 실제 상황(규모, 환경, 장비 가용성 등)에 따라 적절한 시공 방식을 선택할 수 있습니다.
        """
        )

        st.markdown("#### 중요 유의사항")
        st.warning(
            """
        **실제 시공 시**에는 현장 여건, 자재 사양, 공법 등을 종합적으로 고려하여 **설계자 또는 감리자의 최종 판단**이 필요합니다.
        
        모든 단가 및 기간은 **참고용**으로, 실제 현장에서는 **전문가의 판단**이 필요합니다.
        """
        )

    return user_input, area_input, uploaded_file
