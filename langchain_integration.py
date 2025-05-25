"""
LangChain integration for building damage analysis
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import json
from datetime import datetime

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


class DamageAnalysisOutput(BaseModel):
    """Structured output for damage analysis"""

    damage_id: str = Field(description="Unique identifier for this damage analysis")
    image_metadata: Dict[str, Any] = Field(description="Image metadata information")
    damage_analysis: Dict[str, Any] = Field(
        description="Detailed damage analysis results"
    )
    recommendations: Dict[str, Any] = Field(
        description="Repair recommendations and actions"
    )
    confidence_score: float = Field(description="Overall confidence score (0-1)")
    timestamp: str = Field(description="Analysis timestamp")


class DamageAnalysisOutputParser(BaseOutputParser[DamageAnalysisOutput]):
    """Parser for damage analysis output"""

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
        """Parse text output into structured format"""

        # Extract key information using simple text parsing
        lines = text.split("\n")

        damage_analysis = {
            "primary_damage_type": "Unknown",
            "affected_areas": [],
            "severity_score": 1,
            "confidence_level": 0.5,
            "detailed_findings": text,
        }

        recommendations = {
            "immediate_actions": [],
            "repair_priority": "medium",
            "safety_concerns": [],
        }

        # Simple keyword extraction
        text_lower = text.lower()

        # Extract severity
        if any(word in text_lower for word in ["심각", "위험", "critical", "severe"]):
            damage_analysis["severity_score"] = 4
        elif any(word in text_lower for word in ["보통", "moderate"]):
            damage_analysis["severity_score"] = 3
        elif any(word in text_lower for word in ["경미", "minor"]):
            damage_analysis["severity_score"] = 2

        # Extract damage types
        for damage_type in DAMAGE_CATEGORIES["damage_types"]:
            keywords = damage_type.lower().split()
            if any(keyword in text_lower for keyword in keywords):
                damage_analysis["primary_damage_type"] = damage_type
                break

        return DamageAnalysisOutput(
            damage_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            image_metadata={"source": "uploaded_image"},
            damage_analysis=damage_analysis,
            recommendations=recommendations,
            confidence_score=0.7,
            timestamp=datetime.now().isoformat(),
        )

    def _create_default_output(self, text: str) -> DamageAnalysisOutput:
        """Create default output when parsing fails"""

        return DamageAnalysisOutput(
            damage_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            image_metadata={"source": "uploaded_image"},
            damage_analysis={
                "primary_damage_type": "Unknown",
                "affected_areas": [],
                "severity_score": 1,
                "confidence_level": 0.3,
                "detailed_findings": text,
            },
            recommendations={
                "immediate_actions": ["전문가 검토 필요"],
                "repair_priority": "medium",
                "safety_concerns": ["정확한 분석을 위해 추가 검사 필요"],
            },
            confidence_score=0.3,
            timestamp=datetime.now().isoformat(),
        )


class BuildingDamageLLM(LLM):
    """Custom LLM wrapper for building damage analysis model"""

    # Use PrivateAttr to avoid LangChain field validation
    _device: str = PrivateAttr(default="cpu")
    _model: Any = PrivateAttr(default=None)
    _current_image: Any = PrivateAttr(default=None)

    def __init__(
        self, model_path: Optional[Path] = None, device: str = "cpu", **kwargs
    ):
        super().__init__(**kwargs)

        self._device = device

        # Load the trained model
        if model_path is None:
            # Try to find the best model
            model_path = self._find_best_model()

        if model_path and model_path.exists():
            self._model = BuildingDamageAnalysisModel.load_model(model_path, device)
            logger.info(f"Loaded model from {model_path}")
        else:
            # Create a new model if no trained model is found
            from models import create_model

            self._model = create_model(device)
            logger.warning("No trained model found. Using untrained model.")

        self._model.eval()
        self._current_image = None

    def _find_best_model(self) -> Optional[Path]:
        """Find the best trained model"""

        # Look for best_model.pt in training directories
        training_dirs = [
            d
            for d in MODELS_DIR.iterdir()
            if d.is_dir() and d.name.startswith("training_")
        ]

        for training_dir in sorted(training_dirs, reverse=True):  # Most recent first
            best_model_path = training_dir / "best_model.pt"
            if best_model_path.exists():
                return best_model_path

        return None

    def set_image(self, image: Union[str, Path, Image.Image, np.ndarray]):
        """Set the current image for analysis"""

        if isinstance(image, (str, Path)):
            # Load from file path
            image_path = Path(image)
            if image_path.exists():
                self._current_image = Image.open(image_path).convert("RGB")
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")

        elif isinstance(image, Image.Image):
            self._current_image = image.convert("RGB")

        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            self._current_image = Image.fromarray(image)

        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        logger.info("Image set for analysis")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""

        # Resize image to match CLIP model requirements
        image = image.resize((224, 224))

        # Convert to numpy array
        image_array = np.array(image)

        # Normalize
        image_array = image_array.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std

        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)

        return image_tensor.to(self._device)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate response for the given prompt"""

        if self._current_image is None:
            return "오류: 분석할 이미지가 설정되지 않았습니다. set_image() 메서드를 사용하여 이미지를 설정해주세요."

        try:
            # Preprocess image
            image_tensor = self._preprocess_image(self._current_image)

            # Get model predictions
            with torch.no_grad():
                predictions = self._model.get_damage_predictions(
                    image_tensor, [prompt], threshold=0.3
                )

            if predictions:
                prediction = predictions[0]

                # Format the response
                response = self._format_analysis_response(prediction, prompt)
                return response
            else:
                return "분석 결과를 생성할 수 없습니다."

        except Exception as e:
            logger.error(f"Error in damage analysis: {e}")
            return f"분석 중 오류가 발생했습니다: {str(e)}"

    def _format_analysis_response(self, prediction: Dict[str, Any], prompt: str) -> str:
        """Format the model prediction into a readable response"""

        severity_level = prediction["severity_level"]
        severity_desc = prediction["severity_description"]
        damage_types = prediction["damage_types"]
        affected_areas = prediction["affected_areas"]
        confidence = prediction["severity_confidence"]

        response = f"""
# 건물 피해 분석 결과

## 피해 심각도
- **등급**: {severity_level}/5
- **설명**: {severity_desc}
- **신뢰도**: {confidence:.2%}

## 피해 유형
{self._format_list(damage_types) if damage_types else "- 특정 피해 유형을 식별할 수 없습니다."}

## 영향 받은 영역
{self._format_list(affected_areas) if affected_areas else "- 특정 영역을 식별할 수 없습니다."}

## 권장 조치사항
{self._generate_recommendations(severity_level, damage_types)}

## 안전 주의사항
{self._generate_safety_warnings(severity_level)}

---
*분석 기준: {prompt}*
*분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """.strip()

        return response

    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as markdown list"""
        return "\n".join([f"- {item}" for item in items])

    def _generate_recommendations(
        self, severity_level: int, damage_types: List[str]
    ) -> str:
        """Generate recommendations based on severity and damage types"""

        recommendations = []

        if severity_level >= 4:
            recommendations.extend(
                [
                    "즉시 전문가 검사 필요",
                    "건물 사용 중단 고려",
                    "긴급 보수 작업 계획 수립",
                ]
            )
        elif severity_level >= 3:
            recommendations.extend(
                [
                    "전문가 상세 검사 권장",
                    "보수 작업 계획 수립",
                    "정기적인 모니터링 실시",
                ]
            )
        else:
            recommendations.extend(
                ["정기적인 점검 실시", "예방적 보수 고려", "상황 모니터링"]
            )

        # Add specific recommendations based on damage types
        for damage_type in damage_types:
            if "균열" in damage_type or "Crack" in damage_type:
                recommendations.append("균열 진행 상황 모니터링")
            elif "수해" in damage_type or "Water" in damage_type:
                recommendations.append("습기 제거 및 방수 처리")
            elif "화재" in damage_type or "Fire" in damage_type:
                recommendations.append("구조적 안전성 검사 필수")

        return self._format_list(recommendations)

    def _generate_safety_warnings(self, severity_level: int) -> str:
        """Generate safety warnings based on severity"""

        warnings = []

        if severity_level >= 4:
            warnings.extend(
                ["⚠️ 즉시 대피 고려", "⚠️ 건물 출입 제한", "⚠️ 응급 상황 대비"]
            )
        elif severity_level >= 3:
            warnings.extend(
                ["⚠️ 주의 깊은 사용", "⚠️ 정기적인 안전 점검", "⚠️ 악화 징후 모니터링"]
            )
        else:
            warnings.extend(["일반적인 안전 수칙 준수", "변화 상황 주시"])

        return self._format_list(warnings)

    @property
    def _llm_type(self) -> str:
        return "building_damage_analysis"


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
