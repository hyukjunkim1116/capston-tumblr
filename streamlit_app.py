"""
ChatGPT-style Streamlit web application for building damage analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
import plotly.express as px
import plotly.graph_objects as go
import base64

# Import our modules
from langchain_integration import analyze_building_damage, DamageAnalysisOutput
from config import DAMAGE_CATEGORIES, CACHE_DIR, LOGS_DIR
from models import BuildingDamageAnalysisModel
from vector_store import create_vector_store

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🏢 건물 피해 분석 AI 어시스턴트",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Create directories
UPLOAD_DIR = CACHE_DIR / "uploads"
RESULTS_DIR = CACHE_DIR / "results"

for directory in [UPLOAD_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요! 저는 건물 피해 분석 AI 어시스턴트입니다. 🏢\n\n건물 이미지를 업로드하고 피해 면적을 알려주시면, 상세한 피해 분석과 복구 계획을 제공해드리겠습니다.\n\n어떤 도움이 필요하신가요?",
                "timestamp": datetime.now(),
            }
        ]
    if "vector_store" not in st.session_state:
        with st.spinner("표준 데이터베이스를 로딩하고 있습니다..."):
            try:
                st.session_state.vector_store = create_vector_store()
                logger.info("Vector store initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                st.session_state.vector_store = None
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_area" not in st.session_state:
        st.session_state.current_area = None


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file and return path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(uploaded_file.name).suffix
    filename = f"upload_{timestamp}{file_extension}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_comprehensive_analysis(
    damage_result: Dict[str, Any], area: float, vector_store
) -> Dict[str, Any]:
    """Create comprehensive analysis using vector store"""
    if not vector_store:
        return {"error": "표준 데이터베이스를 사용할 수 없습니다."}

    analysis_result = damage_result.get("analysis_result", {})
    damage_analysis = analysis_result.get("damage_analysis", {})

    # Extract damage information
    severity_level = damage_analysis.get("severity_score", 3)
    damage_types = damage_analysis.get("damage_types", [])
    affected_areas = damage_analysis.get("affected_areas", [])

    comprehensive_data = {
        "basic_analysis": damage_analysis,
        "area_info": {
            "total_area": area,
            "damaged_area": area,  # Assume full area is damaged for now
            "area_unit": "m²",
        },
    }

    # Get damage risk index
    primary_damage = damage_types[0] if damage_types else "일반 피해"
    risk_data = vector_store.get_damage_risk_index(primary_damage, severity_level)
    comprehensive_data["risk_assessment"] = risk_data

    # Get repair standards and cost estimates
    repair_data = vector_store.get_repair_standards(primary_damage, area)
    comprehensive_data["repair_plan"] = repair_data

    # Calculate priority score
    priority_score = vector_store.calculate_priority_score(damage_analysis, area)
    comprehensive_data["priority_score"] = priority_score

    # Calculate detailed costs and timeline
    cost_details = calculate_detailed_costs(repair_data, area, severity_level)
    comprehensive_data["cost_breakdown"] = cost_details

    # Generate timeline
    timeline = generate_repair_timeline(repair_data, area, severity_level)
    comprehensive_data["repair_timeline"] = timeline

    return comprehensive_data


def calculate_detailed_costs(
    repair_data: Dict[str, Any], area: float, severity: int
) -> Dict[str, Any]:
    """Calculate detailed cost breakdown"""
    base_unit_cost = repair_data.get("unit_cost", 50000)

    # Adjust costs based on severity
    severity_multiplier = {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5, 5: 2.0}
    multiplier = severity_multiplier.get(severity, 1.2)

    material_cost = base_unit_cost * area * multiplier
    labor_days = max(1, int(area / 50))  # 50m² per day base rate
    labor_cost = 200000 * labor_days * multiplier

    # Additional costs
    equipment_cost = material_cost * 0.1  # 10% of material cost
    overhead_cost = (material_cost + labor_cost) * 0.15  # 15% overhead

    total_cost = material_cost + labor_cost + equipment_cost + overhead_cost

    return {
        "material_cost": int(material_cost),
        "labor_cost": int(labor_cost),
        "equipment_cost": int(equipment_cost),
        "overhead_cost": int(overhead_cost),
        "total_cost": int(total_cost),
        "cost_per_sqm": int(total_cost / area) if area > 0 else 0,
        "labor_days": labor_days,
        "workers_per_day": 1,
    }


def generate_repair_timeline(
    repair_data: Dict[str, Any], area: float, severity: int
) -> Dict[str, Any]:
    """Generate repair timeline"""
    base_days = max(1, int(area / 50))  # Base: 50m² per day

    # Adjust based on severity
    severity_multiplier = {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5, 5: 2.0}
    multiplier = severity_multiplier.get(severity, 1.2)

    total_days = int(base_days * multiplier)

    # Create timeline phases
    phases = []

    if severity >= 4:
        phases.append(
            {"phase": "긴급 안전 조치", "days": 1, "description": "즉시 안전 확보 작업"}
        )

    phases.extend(
        [
            {
                "phase": "준비 및 자재 조달",
                "days": max(1, total_days // 4),
                "description": "자재 구매 및 현장 준비",
            },
            {
                "phase": "주요 복구 작업",
                "days": max(1, total_days // 2),
                "description": "핵심 손상 부위 복구",
            },
            {
                "phase": "마감 및 정리",
                "days": max(1, total_days // 4),
                "description": "마감 작업 및 현장 정리",
            },
            {
                "phase": "검수 및 완료",
                "days": 1,
                "description": "최종 검수 및 인수인계",
            },
        ]
    )

    return {
        "total_duration": total_days,
        "work_hours_per_day": 8,
        "phases": phases,
        "estimated_completion": (
            datetime.now() + pd.Timedelta(days=total_days)
        ).strftime("%Y-%m-%d"),
    }


def create_cost_visualization(cost_data: Dict[str, Any]) -> go.Figure:
    """Create cost breakdown visualization"""
    costs = {
        "자재비": cost_data["material_cost"],
        "인건비": cost_data["labor_cost"],
        "장비비": cost_data["equipment_cost"],
        "관리비": cost_data["overhead_cost"],
    }

    fig = px.pie(
        values=list(costs.values()),
        names=list(costs.keys()),
        title="복구 비용 구성",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400)

    return fig


def create_timeline_chart(timeline_data: Dict[str, Any]) -> go.Figure:
    """Create timeline Gantt chart"""
    phases = timeline_data["phases"]

    # Calculate start and end dates for each phase
    start_date = datetime.now()
    chart_data = []

    current_date = start_date
    for phase in phases:
        end_date = current_date + pd.Timedelta(days=phase["days"])
        chart_data.append(
            {
                "Phase": phase["phase"],
                "Start": current_date,
                "End": end_date,
                "Days": phase["days"],
                "Description": phase["description"],
            }
        )
        current_date = end_date

    df = pd.DataFrame(chart_data)

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Phase",
        title="복구 작업 일정",
        hover_data=["Days", "Description"],
    )

    fig.update_layout(height=400)
    return fig


def format_analysis_response(comprehensive_data: Dict[str, Any]) -> str:
    """Format comprehensive analysis into readable response"""
    if "error" in comprehensive_data:
        return f"❌ 분석 중 오류가 발생했습니다: {comprehensive_data['error']}"

    basic = comprehensive_data["basic_analysis"]
    area_info = comprehensive_data["area_info"]
    risk = comprehensive_data["risk_assessment"]
    repair = comprehensive_data["repair_plan"]
    cost = comprehensive_data["cost_breakdown"]
    timeline = comprehensive_data["repair_timeline"]
    priority = comprehensive_data["priority_score"]

    response = f"""
## 📋 종합 피해 분석 결과

### 🏗️ 기본 정보
- **분석 면적**: {area_info['total_area']:.1f} {area_info['area_unit']}
- **피해 심각도**: {basic.get('severity_score', 'N/A')}/5
- **위험 지수**: {risk['risk_index']}/5
- **복구 우선순위**: {priority}/100

### 🔍 피해 현황
- **주요 피해 유형**: {', '.join(basic.get('damage_types', ['확인되지 않음']))}
- **영향 받은 영역**: {', '.join(basic.get('affected_areas', ['확인되지 않음']))}

### 🛠️ 복구 계획
- **공종명**: {repair['work_type']}
- **예상 자재**: {', '.join(repair['materials']) if repair['materials'] else '표준 자재'}
- **작업 기간**: {timeline['total_duration']}일 (1일 8시간 기준)
- **필요 인력**: {cost['workers_per_day']}명/일

### 💰 비용 분석
- **총 복구 비용**: {cost['total_cost']:,}원
- **㎡당 단가**: {cost['cost_per_sqm']:,}원
- **자재비**: {cost['material_cost']:,}원
- **인건비**: {cost['labor_cost']:,}원
- **장비비**: {cost['equipment_cost']:,}원
- **관리비**: {cost['overhead_cost']:,}원

### 📅 작업 일정
"""

    for phase in timeline["phases"]:
        response += (
            f"- **{phase['phase']}**: {phase['days']}일 - {phase['description']}\n"
        )

    response += f"\n**예상 완료일**: {timeline['estimated_completion']}"

    if priority >= 80:
        response += "\n\n⚠️ **긴급**: 즉시 복구 작업이 필요합니다."
    elif priority >= 60:
        response += "\n\n🔶 **중요**: 빠른 시일 내 복구 작업을 권장합니다."
    else:
        response += "\n\n🔵 **일반**: 계획적인 복구 작업을 진행하세요."

    return response


def display_chat_interface():
    """Display modern chat interface"""
    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            # User message with modern styling
            st.markdown(
                f"""
                <div class="user-message">
                    <strong>👤 사용자</strong><br>
                    {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display image if present
            if "image" in msg:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(
                        msg["image"],
                        caption="📸 업로드된 이미지",
                        use_column_width=True,
                        clamp=True,
                    )

            # Display area if present
            if "area" in msg:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                                padding: 0.75rem; border-radius: 10px; margin: 0.5rem 2rem 1rem 2rem;
                                border-left: 4px solid #2196f3;">
                        📐 <strong>입력된 면적:</strong> {msg['area']} m²
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            # Assistant message with modern styling
            content_with_breaks = msg["content"].replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="assistant-message">
                    <strong>🤖 AI 어시스턴트</strong><br>
                    {content_with_breaks}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display visualizations if present
            if "visualizations" in msg:
                viz = msg["visualizations"]

                st.markdown("### 📊 분석 결과 차트")

                # Create tabs for better organization
                if "cost_chart" in viz and "timeline_chart" in viz:
                    tab1, tab2 = st.tabs(["💰 비용 분석", "📅 일정 계획"])

                    with tab1:
                        st.plotly_chart(viz["cost_chart"], use_container_width=True)

                    with tab2:
                        st.plotly_chart(viz["timeline_chart"], use_container_width=True)

                elif "cost_chart" in viz:
                    st.plotly_chart(viz["cost_chart"], use_container_width=True)

                elif "timeline_chart" in viz:
                    st.plotly_chart(viz["timeline_chart"], use_container_width=True)

        # Add timestamp
        timestamp = msg["timestamp"].strftime("%H:%M")
        st.markdown(
            f"""
            <div style="text-align: {'right' if msg['role'] == 'user' else 'left'}; 
                        color: #999; font-size: 12px; margin: 0.25rem 2rem;">
                {timestamp}
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    """Main application"""
    # Initialize session state
    init_session_state()

    # Custom CSS for modern vertical layout
    st.markdown(
        """
    <style>
    /* Main container styling */
    .main > div {
        padding-top: 1rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Chat message styling */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem 2rem;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: white;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 2rem 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Input section styling */
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    
    .upload-area {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: 2px dashed white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        color: white;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(240, 147, 251, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Form styling */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e9ecef;
        padding: 1rem;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 15px;
        border: 2px dashed #667eea;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main > div {
            padding: 0.5rem;
        }
        
        .user-message, .assistant-message {
            margin: 1rem 0.5rem;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Modern header
    st.markdown(
        """
        <div class="main-header">
            <h1>🏢 건물 피해 분석 AI</h1>
            <p style="font-size: 18px; margin-top: 10px; opacity: 0.9;">
                AI 기반 건물 손상 진단 및 복구 계획 수립 시스템
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chat interface container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    display_chat_interface()
    st.markdown("</div>", unsafe_allow_html=True)

    # Input section
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Create input form with vertical layout
    with st.form("user_input_form", clear_on_submit=True):
        # Message input
        st.markdown("### 💬 메시지 입력")
        user_input = st.text_area(
            "",
            height=100,
            placeholder="건물 피해에 대해 질문하거나 분석을 요청해주세요...",
            help="구체적인 질문이나 요청사항을 입력하세요",
        )

        st.markdown("---")

        # File upload section
        st.markdown("### 📸 이미지 업로드")
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png", "webp"],
            help="건물 피해 이미지를 업로드하세요 (JPG, PNG, WEBP 지원)",
        )

        # Area input and device selection in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📏 피해 면적")
            area_input = st.number_input(
                "",
                min_value=0.1,
                max_value=10000.0,
                value=10.0,
                step=0.1,
                help="피해가 발생한 면적을 m² 단위로 입력하세요",
            )

        with col2:
            st.markdown("### ⚙️ 처리 설정")
            device = st.selectbox(
                "", ["cpu", "cuda"], help="GPU 사용 가능시 cuda 선택 (더 빠른 처리)"
            )

        # Submit button
        submit_button = st.form_submit_button("🚀 분석 시작", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    # Process user input
    if submit_button and (user_input or uploaded_file):
        # Add user message
        user_message = {
            "role": "user",
            "content": user_input or "이미지 분석을 요청합니다.",
            "timestamp": datetime.now(),
        }

        # Add image and area info if provided
        if uploaded_file:
            image = Image.open(uploaded_file)
            user_message["image"] = image
            st.session_state.current_image = uploaded_file

        if area_input:
            user_message["area"] = area_input
            st.session_state.current_area = area_input

        st.session_state.messages.append(user_message)

        # Generate AI response
        with st.spinner("🔍 AI가 이미지를 분석하고 있습니다..."):
            try:
                if uploaded_file and area_input:
                    # Perform comprehensive analysis
                    file_path = save_uploaded_file(uploaded_file)

                    # Basic damage analysis
                    damage_result = analyze_building_damage(
                        image_path=file_path,
                        query=user_input or "건물의 피해 상황을 분석해주세요.",
                        device=device,
                        generate_report=True,
                    )

                    # Comprehensive analysis with vector store
                    comprehensive_data = create_comprehensive_analysis(
                        damage_result, area_input, st.session_state.vector_store
                    )

                    # Format response
                    response_content = format_analysis_response(comprehensive_data)

                    # Create visualizations
                    visualizations = {}
                    if "cost_breakdown" in comprehensive_data:
                        visualizations["cost_chart"] = create_cost_visualization(
                            comprehensive_data["cost_breakdown"]
                        )

                    if "repair_timeline" in comprehensive_data:
                        visualizations["timeline_chart"] = create_timeline_chart(
                            comprehensive_data["repair_timeline"]
                        )

                    # Add assistant response
                    assistant_message = {
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.now(),
                        "visualizations": visualizations,
                    }

                else:
                    # Simple text response
                    response_content = """
🎯 **정확한 피해 분석을 위한 필수 정보**

정밀한 건물 피해 분석을 위해 다음 정보를 제공해주세요:

📸 **건물 피해 이미지**
- 손상된 부위가 명확히 보이는 고화질 사진
- 가능하면 여러 각도에서 촬영한 이미지

📏 **피해 면적 정보**
- 손상된 영역의 정확한 면적 (m² 단위)
- 대략적인 추정치도 가능합니다

---

💡 **제공되는 분석 서비스**

🔍 **AI 피해 진단**
- 9가지 피해 유형 자동 분류
- 5단계 심각도 평가
- 영향 받은 구조 부위 식별

💰 **정밀 비용 산정**
- 건설 표준 단가 기반 계산
- 자재비, 인건비, 장비비 상세 분석
- 면적별 단가 및 총 복구 비용

📅 **체계적 복구 계획**
- 단계별 작업 일정 수립
- 필요 인력 및 장비 산정
- 우선순위 기반 작업 순서

🏗️ **표준 기준 적용**
- 국가 건설 표준 시방서 준수
- 공종별 표준 품셈 적용
- 안전 기준 및 품질 관리

위의 정보를 입력하고 **🚀 분석 시작** 버튼을 클릭해주세요!
                    """

                    assistant_message = {
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.now(),
                    }

                st.session_state.messages.append(assistant_message)

            except Exception as e:
                error_message = {
                    "role": "assistant",
                    "content": f"❌ 분석 중 오류가 발생했습니다: {str(e)}\n\n다시 시도해주시거나 다른 이미지를 업로드해주세요.",
                    "timestamp": datetime.now(),
                }
                st.session_state.messages.append(error_message)

        # Rerun to show new messages
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🏢 건물 피해 분석 AI • 정확하고 신뢰할 수 있는 AI 진단 서비스</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
