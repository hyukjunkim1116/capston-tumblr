import streamlit as st
import os
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# OpenAI API 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    client = OpenAI()
else:
    client = None


# Performance optimization: Cache API responses
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_openai_response(messages_hash: str, messages: list) -> str:
    """Get cached OpenAI response"""
    if client is None:
        return "OpenAI API 키가 설정되지 않았습니다."

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
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])

                # 이미지가 있으면 표시
                if "image" in message:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(
                            message["image"],
                            caption="업로드된 이미지",
                            use_column_width=True,
                        )

                    # 파일 정보를 더 간결하게 표시
                    st.markdown(
                        f"""
                    📊 **분석 정보** | 📁 {message['image'].name} | 📏 {message.get('image_size', 0) / 1024:.1f} KB | 📐 {message.get('area', 0)} m²
                    """
                    )

                # 면적 정보만 있는 경우
                elif "area" in message:
                    st.markdown(f"📐 **분석 면적**: {message['area']} m²")

        else:
            response_type = message.get("response_type", "comprehensive")
            avatar = "🏗️" if response_type == "comprehensive" else "🤖"

            with st.chat_message("assistant", avatar=avatar):
                st.write(message["content"])

    # st.chat_input을 사용한 메시지 입력 (하단에 자동 고정됨)
    user_message = st.chat_input(
        placeholder="건물 피해에 대해 무엇이든 물어보세요...",
        disabled=st.session_state.processing,
    )
    uploaded_file = st.file_uploader(
        "📎 건물 피해 사진 업로드",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="건물 피해 사진을 업로드하여 AI 분석을 받아보세요",
        disabled=st.session_state.processing,
    )

    area_input = st.number_input(
        "면적 (m²)",
        min_value=0.1,
        max_value=10000.0,
        value=10.0,
        step=0.1,
        help="분석할 건물의 면적을 입력하세요",
        disabled=st.session_state.processing,
    )
    # 업로드된 파일 미리보기
    if uploaded_file is not None:
        with st.expander("📷 업로드된 파일 미리보기", expanded=False):
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(uploaded_file, width=120)
            with col_info:
                st.markdown(
                    f"""
                **파일 정보**
                - 📁 파일명: `{uploaded_file.name}`
                - 📏 크기: `{len(uploaded_file.getvalue()) / 1024:.1f} KB`
                - 🏷️ 형식: `{uploaded_file.type.split('/')[-1].upper()}`
                """
                )
    # 메시지 처리 로직 - st.chat_input이 리턴하는 값 처리
    if user_message and not st.session_state.processing:
        st.session_state.processing = True

        # 사용자 메시지 데이터 구성
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

        # API 호출 로직
        if uploaded_file is not None:
            # 이미지가 있는 경우 - 자체 AI 분석 사용 (메인 앱에서 처리)
            try:
                # 이 부분은 메인 앱에서 처리됨
                pass
            except Exception as e:
                st.error(f"이미지 분석 중 오류가 발생했습니다: {str(e)}")
        else:
            # 텍스트만 있는 경우 - OpenAI API 사용
            try:
                # OpenAI API 호출을 위한 메시지 준비
                api_messages = []
                for msg in st.session_state.messages[-5:]:  # 최근 5개 메시지만 사용
                    if msg["role"] in ["user", "assistant"]:
                        content = msg["content"]

                        # 컨텍스트 정보 추가
                        if "area" in msg:
                            content += f"\n[피해 면적: {msg['area']} m²]"

                        api_messages.append({"role": msg["role"], "content": content})

                # 시스템 프롬프트 추가
                system_prompt = """당신은 건물 피해 분석 전문가입니다. 
                사용자의 질문에 대해 전문적이고 상세한 답변을 제공하세요.
                
                다음 항목들을 포함해서 답변하세요:
                1. 피해 유형 분석
                2. 심각도 평가 (1-5 단계)
                3. 수리 우선순위
                4. 예상 수리 비용
                5. 안전 주의사항
                """

                api_messages.insert(0, {"role": "system", "content": system_prompt})

                # 메시지 해시 생성 (캐싱용)
                messages_hash = str(hash(str(api_messages)))

                response = get_openai_response(messages_hash, api_messages)

                # 어시스턴트 응답 추가
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now(),
                        "response_type": "text_only",
                    }
                )

            except Exception as e:
                st.error(f"텍스트 분석 중 오류가 발생했습니다: {str(e)}")

        st.session_state.processing = False
        st.rerun()

    # 사이드바 - 공종명 대체 적용 안내
    with st.sidebar:
        st.markdown("### 📋 공종명 대체 적용 안내")

        st.markdown(
            """
        본 분석에 사용된 공종명은 **건설공사 표준품셈**을 기준으로 하되, 실제 피해 상황과 1:1로 정확히 매칭되지 않는 일부 공종에 대해서는 다음과 같은 기준에 따라 유사하거나 기능적으로 대응 가능한 공종으로 대체 적용하였습니다.
        """
        )

        st.markdown("---")

        st.markdown("#### 🔄 대체 기준")
        st.markdown(
            """
        - **구조적 유사성**: 동일한 구조적 특성을 가진 공종
        - **시공 방법의 유사성**: 비슷한 시공 절차와 방법  
        - **사용 재료 및 작업 목적의 유사성**: 동일한 재료와 목적
        
        등을 종합적으로 고려하여 판단하였습니다.
        """
        )

        st.markdown("---")

        st.markdown("#### 📝 대체 적용 예시")
        st.markdown(
            """
        • **금속 외장 패널** → 경량벽체 철골틀 설치
        
        • **금속기와 파손** → 금속기와 잇기
        
        • **방수층 파손** → 기존 방수층 제거 및 바탕처리 + 방수재 도포
        """
        )

        st.markdown("---")

        st.markdown("#### 🛠️ 복구 방법 선택 안내")
        st.markdown(
            """
        동일한 피해 항목에 대해 복수의 공종명 또는 인력 구성을 적용한 사례를 포함합니다. 사용자는 해당 작업의 실제 상황(규모, 환경, 장비 가용성 등)에 따라 적절한 시공 방식을 선택할 수 있습니다.
        """
        )

        st.markdown("---")

        st.markdown("#### ⚠️ 중요 유의사항")
        st.warning(
            """
        **실제 시공 시**에는 현장 여건, 자재 사양, 공법 등을 종합적으로 고려하여 **설계자 또는 감리자의 최종 판단**이 필요합니다.
        
        모든 단가 및 기간은 **참고용**으로, 실제 현장에서는 **전문가의 판단**이 필요합니다.
        """
        )

    return user_message, uploaded_file, area_input
