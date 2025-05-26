import streamlit as st
import os
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import time

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


@st.dialog("공종명 대체 적용 안내")
def show_construction_guide():
    """공종명 대체 적용 안내 모달"""
    st.markdown(
        """
    ### 📋 공종명 대체 적용 안내
    
    본 분석에 사용된 공종명은 「건설공사 표준품셈」을 기준으로 하되, 실제 피해 상황과 1:1로 정확히 매칭되지 않는 일부 공종에 대해서는 다음과 같은 기준에 따라 유사하거나 기능적으로 대응 가능한 공종으로 대체 적용하였습니다.
    
    #### 🔧 대체 기준:
    구조적 유사성, 시공 방법의 유사성, 사용 재료 및 작업 목적의 유사성 등을 고려하여 판단하였습니다.
    
    #### 📝 예시:
    - **'금속 외장 패널'** → **'경량벽체 철골틀 설치'**
    - **'금속기와 파손'** → **'금속기와 잇기'**
    - **'방수층 파손'** → **'기존 방수층 제거 및 바탕처리 + 방수재 도포'**
    
    #### ⚠️ 유의사항:
    실제 시공 시에는 현장 여건, 자재 사양, 공법 등을 종합적으로 고려하여 설계자 또는 감리자의 최종 판단이 필요합니다.
    
    ---
    
    ### 🛠️ 선택 가능한 복구 방법 안내
    
    ※ 본 자료는 동일한 피해 항목에 대해 복수의 공종명 또는 인력 구성을 적용한 사례를 포함합니다. 사용자는 해당 작업의 실제 상황(규모, 환경, 장비 가용성 등)에 따라 적절한 시공 방식을 선택할 수 있습니다.
    
    ※ 본 자료는 사진 기반 피해 현황을 분석하여, 건설공사 표준품셈 기준 복구 공종과 자재, 인력 정보를 구성한 것입니다.
    
    공종명과 인력 구성은 동일 피해라도 작업 난이도나 범위에 따라 여러 선택지가 존재할 수 있으며, 사용자는 제시된 대안 중에서 상황에 적합한 방식을 선택할 수 있습니다.
    
    **모든 단가 및 기간은 참고용으로, 실제 현장에서는 감리자 또는 전문가의 판단이 필요합니다.**
    """
    )

    if st.button("확인", type="primary", use_container_width=True):
        st.rerun()


def render_simple_chatgpt_ui():
    """간단한 ChatGPT 스타일 UI 렌더링 - 성능 최적화 버전"""

    # CSS 스타일 - 최적화된 버전
    st.markdown(
        """
    <style>
    .main {
        max-width: 900px;
        margin: 0 auto;
        padding: 1.5rem;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 성능 최적화를 위한 GPU 가속 */
    .chat-container, .user-message, .assistant-message {
        transform: translateZ(0);
        will-change: transform;
    }
    
    /* 사이드바 스타일링 최적화 */
    .css-1d391kg {
        padding-top: 1rem !important;
    }
    
    .sidebar .stMarkdown h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
        margin-top: 0 !important;
        margin-bottom: 1.5rem !important;
        text-align: center !important;
        letter-spacing: -0.02em !important;
    }
    
    /* 입력 필드 최적화 */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
    }
    
    /* 버튼 스타일 최적화 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 세션 상태 초기화 - 성능 최적화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # 입력 섹션 - 개선된 레이아웃
    with st.container():
        # 메인 입력 영역
        col1, col2 = st.columns([3, 1])

        with col1:
            user_message = st.text_input(
                "💬 메시지를 입력하세요:",
                placeholder="건물 피해에 대해 질문해보세요... (예: 이 균열의 심각도는 어느 정도인가요?)",
                disabled=st.session_state.processing,
                key="user_input",
            )

        with col2:
            send_button = st.button(
                "🚀 전송",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing or not user_message,
            )

        # 파일 업로드와 면적 입력을 같은 행에 배치 - 개선된 UI
        st.markdown("### 📋 분석 정보 입력")
        col3, col4 = st.columns([2, 1])

        with col3:
            uploaded_file = st.file_uploader(
                "🖼️ 건물 피해 이미지 업로드",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="건물 피해 사진을 업로드하세요 (최대 200MB)",
                disabled=st.session_state.processing,
            )

        with col4:
            area_input = st.number_input(
                "📐 피해 면적 (m²)",
                min_value=0.1,
                max_value=10000.0,
                value=10.0,
                step=0.1,
                help="정확한 피해 면적을 입력하세요",
                disabled=st.session_state.processing,
            )

        # 추가 옵션 (접을 수 있는 섹션)
        with st.expander("🔧 고급 옵션", expanded=False):
            analysis_detail = st.selectbox(
                "분석 상세도",
                ["기본 분석", "상세 분석", "전문가 수준"],
                index=1,
                help="분석의 상세 수준을 선택하세요",
            )

            include_cost = st.checkbox(
                "비용 분석 포함", value=True, help="수리 비용 추정을 포함합니다"
            )

    # 메시지 처리 - 성능 최적화된 버전
    if send_button and user_message and not st.session_state.processing:
        st.session_state.processing = True

        # 사용자 메시지 추가
        message_data = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(),
            "analysis_detail": analysis_detail,
            "include_cost": include_cost,
        }

        # 이미지가 있으면 추가
        if uploaded_file is not None:
            message_data["image"] = uploaded_file
            message_data["image_size"] = len(uploaded_file.getvalue())

        # 면적 정보 추가
        message_data["area"] = area_input

        st.session_state.messages.append(message_data)

        # API 호출 최적화
        if uploaded_file is not None:
            # 이미지가 있는 경우 - 자체 AI 분석 사용
            try:
                with st.spinner("🤖 AI가 이미지를 분석하고 있습니다..."):
                    # 이 부분은 메인 앱에서 처리됨
                    pass
            except Exception as e:
                st.error(f"이미지 분석 중 오류가 발생했습니다: {str(e)}")
        else:
            # 텍스트만 있는 경우 - OpenAI API 사용
            try:
                # OpenAI API 호출을 위한 메시지 준비
                api_messages = []
                for msg in st.session_state.messages[
                    -5:
                ]:  # 최근 5개 메시지만 사용 (성능 최적화)
                    if msg["role"] in ["user", "assistant"]:
                        content = msg["content"]

                        # 컨텍스트 정보 추가
                        if "area" in msg:
                            content += f"\n[피해 면적: {msg['area']} m²]"
                        if "analysis_detail" in msg:
                            content += f"\n[분석 상세도: {msg['analysis_detail']}]"

                        api_messages.append({"role": msg["role"], "content": content})

                # 시스템 프롬프트 추가
                system_prompt = f"""당신은 건물 피해 분석 전문가입니다. 
                사용자의 질문에 대해 전문적이고 상세한 답변을 제공하세요.
                분석 상세도: {analysis_detail}
                비용 분석 포함: {'예' if include_cost else '아니오'}
                
                다음 항목들을 포함해서 답변하세요:
                1. 피해 유형 분석
                2. 심각도 평가 (1-5 단계)
                3. 수리 우선순위
                {'4. 예상 수리 비용' if include_cost else ''}
                5. 안전 주의사항
                """

                api_messages.insert(0, {"role": "system", "content": system_prompt})

                # 메시지 해시 생성 (캐싱용)
                messages_hash = str(hash(str(api_messages)))

                with st.spinner("🤖 AI가 답변을 생성하고 있습니다..."):
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

    # 채팅 메시지 표시 - 성능 최적화된 버전
    if st.session_state.messages:
        st.markdown("### 💬 대화 내역")

        # 최근 메시지부터 표시 (성능 최적화)
        display_messages = st.session_state.messages[-10:]  # 최근 10개만 표시

        for i, message in enumerate(reversed(display_messages)):
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    # 사용자 메시지 표시
                    st.markdown(f"**👤 사용자:** {message['content']}")

                    # 이미지 표시 (최적화된 크기)
                    if "image" in message:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(
                                message["image"],
                                caption="업로드된 이미지",
                                width=250,
                                use_column_width=False,
                            )
                        with col2:
                            # 이미지 정보 표시
                            st.info(
                                f"""
                            📊 **이미지 정보**
                            - 파일명: {message['image'].name}
                            - 크기: {message.get('image_size', 0) / 1024:.1f} KB
                            - 분석 면적: {message.get('area', 0)} m²
                            """
                            )

                    # 분석 옵션 표시
                    if "analysis_detail" in message:
                        st.caption(
                            f"🔧 분석 설정: {message['analysis_detail']}, 비용분석: {'포함' if message.get('include_cost') else '미포함'}"
                        )

                else:
                    # AI 응답 표시
                    response_type = message.get("response_type", "comprehensive")
                    if response_type == "comprehensive":
                        st.markdown("**🤖 AI 건물 피해 분석 전문가:**")
                    else:
                        st.markdown("**🤖 AI 어시스턴트:**")

                    st.markdown(message["content"])

                # 타임스탬프 표시
                if "timestamp" in message:
                    st.caption(
                        f"⏰ {message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )

        # 더 많은 메시지가 있는 경우 알림
        if len(st.session_state.messages) > 10:
            st.info(
                f"💡 총 {len(st.session_state.messages)}개의 메시지 중 최근 10개를 표시하고 있습니다."
            )

    # 사이드바에 사용법 안내 - 최적화된 버전
    with st.sidebar:
        # 커스텀 제목 스타일링
        st.markdown(
            """
        <div style="
            text-align: center; 
            margin-top: -3rem; 
            margin-bottom: 2rem;
        ">
            <h1 style="
                font-size: 6.2rem !important;
                font-weight: 800 !important;
                color: #1f2937 !important;
                margin: 0 !important;
                letter-spacing: -0.03em !important;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">
                Tumblr AI
            </h1>
            <p style="
                color: #6b7280;
                font-size: 0.9rem;
                margin-top: 0.5rem;
            ">
                v2.0 - 건물 피해 분석 전문 시스템
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # 시스템 상태 표시
        st.markdown("### 📊 시스템 상태")
        if st.session_state.processing:
            st.warning("🔄 분석 진행 중...")
        else:
            st.success("✅ 대기 중")

        st.metric(
            "분석 완료",
            len([m for m in st.session_state.messages if m["role"] == "assistant"]),
        )

        st.markdown("### 📋 사용법")
        st.markdown(
            """
        1. **💬 텍스트 입력**: 건물 피해에 대한 질문을 입력하세요
        2. **🖼️ 이미지 업로드**: 피해 사진을 업로드하세요 (권장)
        3. **📐 면적 입력**: 피해 면적을 m² 단위로 정확히 입력하세요
        4. **🔧 고급 옵션**: 필요시 분석 상세도를 조정하세요
        5. **🚀 전송**: 버튼을 클릭하여 AI 분석을 받으세요
        """
        )

        st.markdown("### 🔧 지원 파일 형식")
        st.markdown("- **이미지**: JPG, JPEG, PNG, BMP, TIFF")
        st.markdown("- **최대 크기**: 200MB")

        st.markdown("### ⚠️ 주의사항")
        st.markdown(
            """
        - 정확한 분석을 위해 **면적을 정확히** 입력해주세요
        - **고해상도 이미지**를 사용하면 더 정확한 분석이 가능합니다
        - 본 시스템은 **1차 진단용**이며, 정밀 진단은 전문가 상담이 필요합니다
        """
        )

        # 성능 정보
        with st.expander("⚡ 성능 정보", expanded=False):
            st.markdown(
                """
            - **분석 시간**: 평균 15-30초
            - **지원 언어**: 한국어, 영어
            - **AI 모델**: GPT-4 + 전용 건물 분석 모델
            - **데이터**: 건설공사 표준품셈 기반
            """
            )

        # 공종명 안내 모달 버튼
        if st.button("📋 공종명 대체 적용 안내", use_container_width=True):
            show_construction_guide()

        # 대화 초기화 버튼
        if st.button("🗑️ 대화 초기화", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.processing = False
            st.rerun()

    return user_message, uploaded_file, area_input, send_button
