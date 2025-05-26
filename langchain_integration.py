"""
LangChain integration for building damage analysis - Performance Optimized
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import json
from datetime import datetime
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

from langchain.chains.base import Chain
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr

from models import BuildingDamageAnalysisModel
from config import DAMAGE_CATEGORIES, MODELS_DIR
import cv2

logger = logging.getLogger(__name__)

# Performance optimization: Thread pool for async operations
_thread_pool = ThreadPoolExecutor(max_workers=2)

# Cache for model predictions
_prediction_cache = {}


def performance_timer(func):
    """Decorator to measure function execution time"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result

    return wrapper


class DamageAnalysisOutput(BaseModel):
    """Enhanced structured output for damage analysis"""

    damage_id: str = Field(description="Unique identifier for this damage analysis")
    image_metadata: Dict[str, Any] = Field(description="Image metadata information")
    damage_analysis: Dict[str, Any] = Field(
        description="Detailed damage analysis results"
    )
    recommendations: Dict[str, Any] = Field(
        description="Repair recommendations and actions"
    )
    cost_analysis: Dict[str, Any] = Field(
        description="Detailed cost breakdown and estimates"
    )
    repair_specifications: Dict[str, Any] = Field(
        description="Detailed repair specifications and methods"
    )
    confidence_score: float = Field(description="Overall confidence score (0-1)")
    timestamp: str = Field(description="Analysis timestamp")


class DamageAnalysisOutputParser(BaseOutputParser[DamageAnalysisOutput]):
    """Enhanced parser for damage analysis output"""

    @performance_timer
    def parse(self, text: str) -> DamageAnalysisOutput:
        """Parse the LLM output into structured format"""
        try:
            # Try to parse as JSON first
            if text.strip().startswith("{"):
                data = json.loads(text)
                return DamageAnalysisOutput(**data)

            # If not JSON, create structured output from text
            return self._parse_text_output(text)

        except Exception as e:
            logger.error(f"Error parsing output: {e}")
            return self._create_default_output(text)

    def _parse_text_output(self, text: str) -> DamageAnalysisOutput:
        """Parse text output into enhanced structured format"""

        # Extract key information using simple text parsing
        lines = text.split("\n")

        damage_analysis = {
            "primary_damage_type": "Unknown",
            "damage_types": [],
            "affected_areas": [],
            "severity_score": 1,
            "confidence_level": 0.5,
            "detailed_findings": text,
            "structural_impact": "미확인",
            "safety_risk_level": "보통",
        }

        recommendations = {
            "immediate_actions": [],
            "repair_priority": "medium",
            "safety_concerns": [],
            "repair_timeline": "1-2주",
            "professional_consultation": False,
        }

        cost_analysis = {
            "material_cost": 0,
            "labor_cost": 0,
            "equipment_cost": 0,
            "total_cost": 0,
            "cost_per_sqm": 0,
            "cost_breakdown": {},
        }

        repair_specifications = {
            "repair_methods": [],
            "required_materials": [],
            "required_equipment": [],
            "labor_requirements": {},
            "construction_standards": "건설공사 표준품셈 2024",
            "quality_standards": [],
        }

        # Simple keyword extraction
        text_lower = text.lower()

        # Extract severity
        if any(
            word in text_lower
            for word in ["심각", "위험", "critical", "severe", "긴급"]
        ):
            damage_analysis["severity_score"] = 4
            damage_analysis["safety_risk_level"] = "높음"
            recommendations["professional_consultation"] = True
        elif any(word in text_lower for word in ["보통", "moderate", "중간"]):
            damage_analysis["severity_score"] = 3
            damage_analysis["safety_risk_level"] = "보통"
        elif any(word in text_lower for word in ["경미", "minor", "가벼운"]):
            damage_analysis["severity_score"] = 2
            damage_analysis["safety_risk_level"] = "낮음"

        # Extract damage types
        for damage_type in DAMAGE_CATEGORIES["damage_types"]:
            keywords = damage_type.lower().split()
            if any(keyword in text_lower for keyword in keywords):
                damage_analysis["primary_damage_type"] = damage_type
                damage_analysis["damage_types"].append(damage_type)

        # Extract affected areas
        for area in DAMAGE_CATEGORIES["affected_areas"]:
            if area.lower() in text_lower:
                damage_analysis["affected_areas"].append(area)

        return DamageAnalysisOutput(
            damage_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            image_metadata={"source": "uploaded_image"},
            damage_analysis=damage_analysis,
            recommendations=recommendations,
            cost_analysis=cost_analysis,
            repair_specifications=repair_specifications,
            confidence_score=0.7,
            timestamp=datetime.now().isoformat(),
        )

    def _create_default_output(self, text: str) -> DamageAnalysisOutput:
        """Create enhanced default output when parsing fails"""

        return DamageAnalysisOutput(
            damage_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            image_metadata={"source": "uploaded_image"},
            damage_analysis={
                "primary_damage_type": "Unknown",
                "damage_types": [],
                "affected_areas": [],
                "severity_score": 1,
                "confidence_level": 0.3,
                "detailed_findings": text,
                "structural_impact": "미확인",
                "safety_risk_level": "미확인",
            },
            recommendations={
                "immediate_actions": ["전문가 검토 필요"],
                "repair_priority": "medium",
                "safety_concerns": ["정확한 분석을 위해 추가 검사 필요"],
                "repair_timeline": "전문가 상담 후 결정",
                "professional_consultation": True,
            },
            cost_analysis={
                "material_cost": 0,
                "labor_cost": 0,
                "equipment_cost": 0,
                "total_cost": 0,
                "cost_per_sqm": 0,
                "cost_breakdown": {"분석 실패": "비용 산정 불가"},
            },
            repair_specifications={
                "repair_methods": ["전문가 진단 필요"],
                "required_materials": [],
                "required_equipment": [],
                "labor_requirements": {},
                "construction_standards": "건설공사 표준품셈 2024",
                "quality_standards": ["전문가 검토 필요"],
            },
            confidence_score=0.3,
            timestamp=datetime.now().isoformat(),
        )


class BuildingDamageLLM(LLM):
    """Enhanced custom LLM wrapper for building damage analysis model"""

    # Use PrivateAttr to avoid LangChain field validation
    _model: Any = PrivateAttr(default=None)
    _device: str = PrivateAttr(default="cpu")
    _current_image: Optional[Image.Image] = PrivateAttr(default=None)
    _model_path: Optional[Path] = PrivateAttr(default=None)

    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        super().__init__()
        self._device = device
        self._model_path = model_path
        self._load_model()

    @performance_timer
    def _load_model(self):
        """Load the damage analysis model with performance optimization"""
        try:
            if self._model_path and self._model_path.exists():
                logger.info(f"Loading model from {self._model_path}")
                self._model = BuildingDamageAnalysisModel.load_model(
                    self._model_path, self._device
                )
            else:
                logger.info("Creating new model instance")
                self._model = BuildingDamageAnalysisModel()
                self._model.to(self._device)

            self._model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None

    def set_image(self, image_path: Union[str, Path]):
        """Set image for analysis with caching"""
        try:
            # Check cache first
            cache_key = str(image_path)
            if cache_key in _prediction_cache:
                logger.info("Using cached image")
                return

            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            self._current_image = image
            logger.info(f"Image set successfully: {image.size}")

        except Exception as e:
            logger.error(f"Failed to set image: {e}")
            self._current_image = None

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input with optimization"""
        try:
            # Resize image for faster processing while maintaining quality
            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to tensor
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array).float()

            # Normalize
            image_tensor = image_tensor / 255.0

            # Rearrange dimensions: (H, W, C) -> (C, H, W)
            image_tensor = image_tensor.permute(2, 0, 1)

            # Add batch dimension: (C, H, W) -> (1, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor.to(self._device)

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "building_damage_analysis_enhanced"

    @performance_timer
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate enhanced response for the given prompt"""

        if self._current_image is None:
            return "오류: 분석할 이미지가 설정되지 않았습니다. set_image() 메서드를 사용하여 이미지를 설정해주세요."

        try:
            # Check cache first
            cache_key = f"{id(self._current_image)}_{hash(prompt)}"
            if cache_key in _prediction_cache:
                logger.info("Using cached prediction")
                return _prediction_cache[cache_key]

            # Preprocess image
            image_tensor = self._preprocess_image(self._current_image)

            # Get model predictions
            with torch.no_grad():
                predictions = self._model.get_damage_predictions(
                    image_tensor, [prompt], threshold=0.3
                )

            if predictions:
                prediction = predictions[0]

                # Format the enhanced response
                response = self._format_enhanced_analysis_response(prediction, prompt)

                # Cache the result
                _prediction_cache[cache_key] = response

                return response
            else:
                return "분석 결과를 생성할 수 없습니다."

        except Exception as e:
            logger.error(f"Error in damage analysis: {e}")
            return f"분석 중 오류가 발생했습니다: {str(e)}"

    def _format_enhanced_analysis_response(
        self, prediction: Dict[str, Any], prompt: str
    ) -> str:
        """Format the model prediction into an enhanced readable response"""

        severity_level = prediction["severity_level"]
        severity_desc = prediction["severity_description"]
        damage_types = prediction["damage_types"]
        affected_areas = prediction["affected_areas"]
        confidence = prediction["severity_confidence"]

        # Enhanced response with more detailed information
        response = f"""
# 🏗️ 건물 피해 분석 상세 보고서

## 📊 분석 개요
- **분석 ID**: {datetime.now().strftime('ANA-%Y%m%d-%H%M%S')}
- **분석 시간**: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}
- **신뢰도**: {confidence:.1%}

## 🔍 피해 현황 분석

### 🚨 피해 심각도
- **등급**: {severity_level}/5
- **설명**: {severity_desc}
- **구조적 영향**: {self._assess_structural_impact(severity_level)}

### 🏠 피해 유형 및 영역
{self._format_damage_details(damage_types, affected_areas)}

## 🔧 복구 방법 및 사양

### 📋 권장 복구 방법
{self._generate_enhanced_recommendations(severity_level, damage_types)}

### 🛠️ 필요 자재 및 장비
{self._generate_material_equipment_list(damage_types)}

### 👷 인력 구성
{self._generate_labor_requirements(severity_level, damage_types)}

## 💰 비용 분석
{self._generate_cost_analysis(severity_level, damage_types)}

## ⚠️ 안전 및 품질 관리
{self._generate_enhanced_safety_warnings(severity_level)}

## 📋 적용 기준
- **건설공사 표준품셈**: 2024년 기준
- **건축법**: 현행 건축법 및 시행령
- **KS 기준**: 해당 자재 및 공법 관련 KS 기준

---
*분석 기준: {prompt}*
*분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*시스템: Tumblr AI v2.0 Enhanced*
        """.strip()

        return response

    def _assess_structural_impact(self, severity_level: int) -> str:
        """Assess structural impact based on severity"""
        impact_levels = {
            1: "구조적 영향 없음",
            2: "경미한 구조적 영향",
            3: "보통 수준의 구조적 영향",
            4: "심각한 구조적 영향 가능성",
            5: "매우 심각한 구조적 위험",
        }
        return impact_levels.get(severity_level, "구조적 영향 미확인")

    def _format_damage_details(
        self, damage_types: List[str], affected_areas: List[str]
    ) -> str:
        """Format detailed damage information"""
        details = []

        if damage_types:
            details.append("**피해 유형:**")
            for i, damage_type in enumerate(damage_types, 1):
                details.append(f"  {i}. {damage_type}")
        else:
            details.append("**피해 유형:** 특정 피해 유형을 식별할 수 없습니다.")

        if affected_areas:
            details.append("\n**영향 받은 영역:**")
            for i, area in enumerate(affected_areas, 1):
                details.append(f"  {i}. {area}")
        else:
            details.append("\n**영향 받은 영역:** 특정 영역을 식별할 수 없습니다.")

        return "\n".join(details)

    def _generate_enhanced_recommendations(
        self, severity_level: int, damage_types: List[str]
    ) -> str:
        """Generate enhanced repair recommendations"""
        recommendations = []

        # Base recommendations by damage type
        repair_methods = {
            "균열": [
                "균열 부위 정밀 조사 및 원인 분석",
                "균열 폭 및 깊이 측정",
                "에폭시 수지 주입 또는 실링 처리",
                "표면 마감 복구",
            ],
            "누수": [
                "누수 경로 추적 및 원인 파악",
                "기존 방수층 상태 점검",
                "방수층 보수 또는 재시공",
                "배수 시설 점검 및 개선",
            ],
            "화재": [
                "화재 손상 범위 정밀 조사",
                "구조 안전성 검토",
                "손상 부재 교체 또는 보강",
                "내화 성능 복구",
            ],
        }

        primary_damage = damage_types[0] if damage_types else "일반"
        methods = repair_methods.get(primary_damage, ["전문가 진단 후 결정"])

        for i, method in enumerate(methods, 1):
            recommendations.append(f"{i}. {method}")

        return "\n".join(recommendations)

    def _generate_material_equipment_list(self, damage_types: List[str]) -> str:
        """Generate detailed material and equipment list"""
        materials = {
            "균열": ["에폭시 수지", "프라이머", "실링재", "보수 모르타르"],
            "누수": ["방수 시트", "우레탄 방수재", "실리콘 실란트", "배수재"],
            "화재": ["내화재", "단열재", "구조용 강재", "내화 도료"],
        }

        equipment = {
            "균열": ["주입기", "압축기", "그라인더", "청소 장비"],
            "누수": ["토치", "롤러", "압착기", "건조 장비"],
            "화재": ["절단기", "용접기", "크레인", "안전 장비"],
        }

        primary_damage = damage_types[0] if damage_types else "일반"

        result = "**주요 자재:**\n"
        for i, material in enumerate(
            materials.get(primary_damage, ["전문가 상담 필요"]), 1
        ):
            result += f"  {i}. {material}\n"

        result += "\n**필요 장비:**\n"
        for i, equip in enumerate(
            equipment.get(primary_damage, ["전문가 상담 필요"]), 1
        ):
            result += f"  {i}. {equip}\n"

        return result

    def _generate_labor_requirements(
        self, severity_level: int, damage_types: List[str]
    ) -> str:
        """Generate detailed labor requirements"""
        base_labor = {"특급기능사": 1, "고급기능사": 1, "보통인부": 2}

        # Adjust based on severity
        if severity_level >= 4:
            base_labor["특급기능사"] += 1
            base_labor["고급기능사"] += 1

        result = "**인력 구성:**\n"
        for job_type, count in base_labor.items():
            result += f"  - {job_type}: {count}명\n"

        return result

    def _generate_cost_analysis(
        self, severity_level: int, damage_types: List[str]
    ) -> str:
        """Generate detailed cost analysis"""
        base_costs = {
            "균열": {"material": 25000, "labor": 15000},
            "누수": {"material": 35000, "labor": 20000},
            "화재": {"material": 80000, "labor": 40000},
        }

        primary_damage = damage_types[0] if damage_types else "일반"
        costs = base_costs.get(primary_damage, {"material": 30000, "labor": 18000})

        # Adjust based on severity
        multiplier = 1 + (severity_level - 1) * 0.2

        material_cost = costs["material"] * multiplier
        labor_cost = costs["labor"] * multiplier
        total_cost = material_cost + labor_cost

        result = f"""**비용 구성 (m²당):**
  - 자재비: {material_cost:,.0f}원
  - 노무비: {labor_cost:,.0f}원
  - 총 단가: {total_cost:,.0f}원

**비용 산정 기준:**
  - 건설공사 표준품셈 2024년 기준
  - 일반적인 시장 단가 적용
  - 현장 여건에 따라 ±20% 변동 가능"""

        return result

    def _generate_enhanced_safety_warnings(self, severity_level: int) -> str:
        """Generate enhanced safety warnings"""
        warnings = []

        if severity_level >= 4:
            warnings.extend(
                [
                    "🚨 **즉시 조치 필요**",
                    "  - 해당 영역 출입 금지",
                    "  - 구조 엔지니어 긴급 진단",
                    "  - 임시 보강 조치 검토",
                    "",
                    "⚠️ **안전 관리**",
                    "  - 작업자 안전교육 필수",
                    "  - 개인보호구 착용 의무",
                    "  - 안전관리자 상주",
                ]
            )
        elif severity_level >= 3:
            warnings.extend(
                [
                    "⚠️ **주의 깊은 관리**",
                    "  - 정기적 안전 점검",
                    "  - 작업 중 안전 확보",
                    "  - 진행 상황 모니터링",
                    "",
                    "🔍 **품질 관리**",
                    "  - 시공 품질 검사",
                    "  - 자재 품질 확인",
                    "  - 완료 후 성능 검증",
                ]
            )
        else:
            warnings.extend(
                [
                    "✅ **일반 안전 관리**",
                    "  - 기본 안전수칙 준수",
                    "  - 정기 점검 실시",
                    "  - 예방적 유지관리",
                ]
            )

        return "\n".join(warnings)


class ImageAnalysisChain(Chain):
    """LangChain for image-based damage analysis"""

    # Use PrivateAttr to avoid LangChain field validation
    _llm: Any = PrivateAttr(default=None)
    _prompt_template: Any = PrivateAttr(default=None)
    _output_parser: Any = PrivateAttr(default=None)

    @property
    def input_keys(self) -> List[str]:
        return ["image_path", "query"]

    @property
    def output_keys(self) -> List[str]:
        return ["analysis_result"]

    def __init__(
        self, model_path: Optional[Path] = None, device: str = "cpu", **kwargs
    ):
        super().__init__(**kwargs)

        # Initialize the custom LLM
        self._llm = BuildingDamageLLM(model_path, device)

        # Create prompt template
        self._prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""
건물 피해 분석을 수행해주세요.

분석 요청: {query}

이미지를 기반으로 다음 사항들을 분석해주세요:
1. 피해의 심각도 (1-5 등급)
2. 피해 유형 식별
3. 영향을 받은 건물 영역
4. 권장 조치사항
5. 안전 주의사항

상세하고 전문적인 분석을 제공해주세요.
            """.strip(),
        )

        # Output parser
        self._output_parser = DamageAnalysisOutputParser()

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the chain"""

        image_path = inputs["image_path"]
        query = inputs.get("query", "건물의 피해 상황을 분석해주세요.")

        try:
            # Set the image for analysis
            self._llm.set_image(image_path)

            # Format the prompt
            formatted_prompt = self._prompt_template.format(query=query)

            # Get analysis result
            analysis_text = self._llm(formatted_prompt)

            # Parse the output
            parsed_result = self._output_parser.parse(analysis_text)

            return {"analysis_result": parsed_result}

        except Exception as e:
            logger.error(f"Error in ImageAnalysisChain: {e}")

            # Return error result
            error_result = DamageAnalysisOutput(
                damage_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                image_metadata={"source": str(image_path), "error": str(e)},
                damage_analysis={
                    "primary_damage_type": "Analysis Failed",
                    "affected_areas": [],
                    "severity_score": 0,
                    "confidence_level": 0.0,
                    "detailed_findings": f"분석 실패: {str(e)}",
                },
                recommendations={
                    "immediate_actions": ["수동 검사 필요"],
                    "repair_priority": "unknown",
                    "safety_concerns": ["분석 시스템 오류로 인한 수동 검사 필요"],
                },
                confidence_score=0.0,
                timestamp=datetime.now().isoformat(),
            )

            return {"analysis_result": error_result}


class ReportGenerationChain(Chain):
    """Chain for generating comprehensive damage reports"""

    # Use PrivateAttr to avoid LangChain field validation
    _report_template: str = PrivateAttr(default="")

    @property
    def input_keys(self) -> List[str]:
        return ["analysis_result", "additional_info"]

    @property
    def output_keys(self) -> List[str]:
        return ["report"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._report_template = """
# 건물 피해 분석 종합 보고서

## 분석 개요
- **분석 ID**: {damage_id}
- **분석 일시**: {timestamp}
- **전체 신뢰도**: {confidence_score:.1%}

## 피해 현황 요약
### 주요 피해 유형
{primary_damage_type}

### 피해 심각도
- **등급**: {severity_score}/5
- **설명**: {severity_description}

### 영향 받은 영역
{affected_areas}

## 상세 분석 결과
{detailed_findings}

## 권장 조치사항
### 즉시 조치사항
{immediate_actions}

### 보수 우선순위
{repair_priority}

### 안전 주의사항
{safety_concerns}

## 추가 정보
{additional_info}

---
*본 보고서는 AI 기반 자동 분석 시스템에 의해 생성되었습니다.*
*정확한 진단을 위해서는 전문가의 현장 검사가 필요합니다.*
        """

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report"""

        analysis_result = inputs["analysis_result"]
        additional_info = inputs.get("additional_info", "추가 정보 없음")

        if isinstance(analysis_result, DamageAnalysisOutput):
            # Extract data from structured output
            damage_analysis = analysis_result.damage_analysis
            recommendations = analysis_result.recommendations

            # Format the report
            report = self._report_template.format(
                damage_id=analysis_result.damage_id,
                timestamp=analysis_result.timestamp,
                confidence_score=analysis_result.confidence_score,
                primary_damage_type=damage_analysis.get(
                    "primary_damage_type", "Unknown"
                ),
                severity_score=damage_analysis.get("severity_score", 0),
                severity_description=f"{damage_analysis.get('severity_score', 0)}/5 등급",
                affected_areas=self._format_list(
                    damage_analysis.get("affected_areas", [])
                ),
                detailed_findings=damage_analysis.get(
                    "detailed_findings", "상세 분석 정보 없음"
                ),
                immediate_actions=self._format_list(
                    recommendations.get("immediate_actions", [])
                ),
                repair_priority=recommendations.get("repair_priority", "unknown"),
                safety_concerns=self._format_list(
                    recommendations.get("safety_concerns", [])
                ),
                additional_info=additional_info,
            )
        else:
            # Handle string input
            report = f"""
# 건물 피해 분석 보고서

## 분석 결과
{analysis_result}

## 추가 정보
{additional_info}

---
*분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
            """

        return {"report": report}

    def _format_list(self, items: List[str]) -> str:
        """Format list items"""
        if not items:
            return "- 해당 없음"
        return "\n".join([f"- {item}" for item in items])


class ValidationChain(Chain):
    """Chain for validating analysis results"""

    @property
    def input_keys(self) -> List[str]:
        return ["analysis_result"]

    @property
    def output_keys(self) -> List[str]:
        return ["validation_result", "confidence_adjustment"]

    def __init__(self):
        super().__init__()

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis results"""

        analysis_result = inputs["analysis_result"]

        validation_checks = {
            "severity_consistency": True,
            "damage_type_validity": True,
            "area_consistency": True,
            "recommendation_appropriateness": True,
        }

        confidence_adjustment = 1.0
        issues = []

        if isinstance(analysis_result, DamageAnalysisOutput):
            damage_analysis = analysis_result.damage_analysis

            # Check severity consistency
            severity = damage_analysis.get("severity_score", 0)
            if severity < 1 or severity > 5:
                validation_checks["severity_consistency"] = False
                issues.append("심각도 등급이 유효 범위(1-5)를 벗어남")
                confidence_adjustment *= 0.7

            # Check damage types
            damage_types = damage_analysis.get("damage_types", [])
            valid_types = DAMAGE_CATEGORIES["damage_types"]

            for damage_type in damage_types:
                if damage_type not in valid_types:
                    validation_checks["damage_type_validity"] = False
                    issues.append(f"알 수 없는 피해 유형: {damage_type}")
                    confidence_adjustment *= 0.8

        validation_result = {
            "is_valid": all(validation_checks.values()),
            "checks": validation_checks,
            "issues": issues,
            "overall_confidence": analysis_result.confidence_score
            * confidence_adjustment,
        }

        return {
            "validation_result": validation_result,
            "confidence_adjustment": confidence_adjustment,
        }


def create_damage_analysis_pipeline(
    model_path: Optional[Path] = None, device: str = "cpu"
) -> Dict[str, Chain]:
    """Create complete damage analysis pipeline"""

    # Create individual chains
    image_analysis_chain = ImageAnalysisChain(model_path, device)
    report_generation_chain = ReportGenerationChain()
    validation_chain = ValidationChain()

    return {
        "image_analysis": image_analysis_chain,
        "report_generation": report_generation_chain,
        "validation": validation_chain,
    }


def analyze_building_damage(
    image_path: Union[str, Path],
    query: str = "건물의 피해 상황을 분석해주세요.",
    model_path: Optional[Path] = None,
    device: str = "cpu",
    generate_report: bool = True,
) -> Dict[str, Any]:
    """
    Complete building damage analysis workflow

    Args:
        image_path: Path to the image to analyze
        query: Analysis query/request
        model_path: Path to trained model (optional)
        device: Device to run analysis on
        generate_report: Whether to generate comprehensive report

    Returns:
        Dictionary with analysis results
    """

    # Create pipeline
    pipeline = create_damage_analysis_pipeline(model_path, device)

    # Step 1: Image Analysis
    logger.info("Starting image analysis...")
    analysis_inputs = {"image_path": str(image_path), "query": query}

    analysis_output = pipeline["image_analysis"](analysis_inputs)
    analysis_result = analysis_output["analysis_result"]

    # Step 2: Validation
    logger.info("Validating analysis results...")
    validation_output = pipeline["validation"]({"analysis_result": analysis_result})
    validation_result = validation_output["validation_result"]

    # Step 3: Report Generation (if requested)
    report = None
    if generate_report:
        logger.info("Generating comprehensive report...")
        report_inputs = {
            "analysis_result": analysis_result,
            "additional_info": f"검증 결과: {validation_result}",
        }
        report_output = pipeline["report_generation"](report_inputs)
        report = report_output["report"]

    return {
        "analysis_result": analysis_result,
        "validation_result": validation_result,
        "report": report,
        "success": validation_result["is_valid"],
    }


if __name__ == "__main__":
    # Test the LangChain integration
    import logging

    logging.basicConfig(level=logging.INFO)

    # Test with a sample image (if available)
    sample_image_path = Path("learning_pictures/1.jpg")

    if sample_image_path.exists():
        result = analyze_building_damage(
            image_path=sample_image_path,
            query="이 건물의 피해 상황을 자세히 분석해주세요.",
            device="cpu",
        )

        print("Analysis completed!")
        print(f"Success: {result['success']}")
        if result["report"]:
            print("\nGenerated Report:")
            print(result["report"])
    else:
        print(f"Sample image not found: {sample_image_path}")
        print("Please provide a valid image path for testing.")
