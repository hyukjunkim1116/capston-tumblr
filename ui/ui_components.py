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
    """간단한 ChatGPT 스타일 UI 렌더링"""

    # CSS 스타일
    st.markdown(
        """
    <style>
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    
    .user-message {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background: #f1f8e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    
    /* 사이드바 스타일링 */
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
    
    /* 사이드바 전체 패딩 조정 */
    .css-1lcbmhc {
        padding-top: 0.5rem !important;
    }
    
    /* 사이드바 컨테이너 */
    .css-1y4p8pa {
        padding-top: 0.5rem !important;
    }
    
    /* 메인 사이드바 영역 */
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem !important;
    }
    
    /* 사이드바 내부 컨텐츠 */
    .css-1cypcdb {
        padding-top: 0 !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 입력 섹션
    with st.container():

        # 텍스트 입력
        user_message = st.text_input(
            "메시지를 입력하세요:", placeholder="건물 피해에 대해 질문해보세요..."
        )

        # 파일 업로드와 면적 입력을 같은 행에 배치
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "건물 피해 이미지 업로드",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="건물 피해 사진을 업로드하세요",
            )

        with col2:
            area_input = st.number_input(
                "피해 면적 (m²)",
                min_value=0.1,
                max_value=10000.0,
                value=10.0,
                step=0.1,
                help="피해 면적을 입력하세요",
            )

        # 전송 버튼
        send_button = st.button("전송", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 메시지 처리
    if send_button and user_message:
        # 사용자 메시지 추가
        message_data = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(),
        }

        # 이미지가 있으면 추가
        if uploaded_file is not None:
            message_data["image"] = uploaded_file

        # 면적 정보 추가
        message_data["area"] = area_input

        st.session_state.messages.append(message_data)

        # OpenAI API 호출을 위한 메시지 준비
        api_messages = []
        for msg in st.session_state.messages:
            if msg["role"] in ["user", "assistant"]:
                content = msg["content"]

                # 이미지와 면적 정보가 있으면 컨텍스트에 추가
                if "image" in msg:
                    content += f"\n[이미지가 업로드됨: {msg['image'].name}]"
                if "area" in msg:
                    content += f"\n[피해 면적: {msg['area']} m²]"

                api_messages.append({"role": msg["role"], "content": content})

        # 시스템 프롬프트 추가
        system_prompt = """당신은 건물 피해 분석 전문가입니다. 
        사용자가 업로드한 건물 피해 이미지와 면적 정보를 바탕으로 상세한 분석을 제공하세요.
        다음 항목들을 포함해서 답변하세요:
        1. 피해 유형 분석
        2. 심각도 평가 (1-5 단계)
        3. 수리 우선순위
        4. 예상 수리 비용
        5. 안전 주의사항
        """

        api_messages.insert(0, {"role": "system", "content": system_prompt})

        try:
            # OpenAI API 호출
            if client is None:
                st.error(
                    "OpenAI API 키가 설정되지 않았습니다. 환경변수를 확인해주세요."
                )
                return

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
            )
            response = completion.choices[0].message.content

            # 어시스턴트 응답 추가
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "timestamp": datetime.now()}
            )

        except Exception as e:
            st.error(f"API 호출 중 오류가 발생했습니다: {str(e)}")
            st.info("OpenAI API 키를 확인해주세요.")

    # 채팅 메시지 표시 (최신 메시지가 위에 오도록)
    if st.session_state.messages:
        st.markdown("### 💬 대화 내역")

        for message in reversed(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(f"**사용자:** {message['content']}")

                    # 이미지 표시
                    if "image" in message:
                        st.image(message["image"], caption="업로드된 이미지", width=300)

                    # 면적 정보 표시
                    if "area" in message:
                        st.info(f"📐 입력된 피해 면적: {message['area']} m²")

                else:
                    st.write(f"**AI 분석가:** {message['content']}")

                # 타임스탬프 표시
                if "timestamp" in message:
                    st.caption(
                        f"⏰ {message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )

    # 사이드바에 사용법 안내
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
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 사용법")
        st.markdown(
            """
        1. **텍스트 입력**: 건물 피해에 대한 질문을 입력하세요
        2. **이미지 업로드**: 피해 사진을 업로드하세요(필수)
        3. **면적 입력**: 피해 면적을 m² 단위로 입력하세요
        4. **전송**: 버튼을 클릭하여 AI 분석을 받으세요
        """
        )

        st.markdown("### 🔧 지원 파일 형식")
        st.markdown("- JPG, JPEG, PNG, BMP, TIFF")

        st.markdown("### ⚠️ 주의사항")
        st.markdown(
            """
        - 정확한 분석을 위해 면적을 정확히 입력해주세요
        """
        )
        # 공종명 안내 모달 버튼
        if st.button("📋 공종명 대체 적용 안내", use_container_width=True):
            show_construction_guide()

    return user_message, uploaded_file, area_input, send_button
