"""
보고서 포맷터 모듈
분석 결과를 종합 보고서로 포맷팅
"""

from datetime import datetime
from typing import Dict, Any
from .data_processor import get_standard_repair_data


def get_severity_description(severity_score: int) -> str:
    """심각도 점수에 따른 설명 반환"""
    severity_descriptions = {
        1: "경미한 피해",
        2: "가벼운 피해",
        3: "보통 피해",
        4: "심각한 피해",
        5: "매우 심각한 피해",
    }
    return severity_descriptions.get(severity_score, "보통 피해")


def get_repair_methods(primary_damage: str) -> list:
    """피해 유형별 복구 방법 반환"""
    repair_methods = {
        "균열": [
            "균열 부위 청소 및 이물질 제거",
            "에폭시 수지 주입을 통한 균열 보수",
            "표면 마감 및 방수 처리",
        ],
        "누수": [
            "누수 원인 파악 및 차단",
            "기존 방수층 제거",
            "신규 방수층 시공",
            "마감재 복구",
        ],
        "화재": ["손상 부재 철거", "구조 보강", "내화재 시공", "마감재 복구"],
        "부식": ["부식 부위 제거", "방청 처리", "보강재 설치", "보호 도장"],
    }

    damage_key = next(
        (key for key in repair_methods.keys() if key in primary_damage), "균열"
    )
    return repair_methods.get(damage_key, repair_methods["균열"])


def format_basic_info_section(area: float, confidence_level: float) -> str:
    """기본 정보 섹션 포맷"""
    return f"""# 건물 피해 분석 종합 보고서

## 기본 정보
| 항목 | 내용 |
|------|------|
| **분석 일시** | {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')} |
| **분석 면적** | {area:,.1f} m² |
| **분석 신뢰도** | {confidence_level:.1%} |
| **보고서 ID** | RPT-{datetime.now().strftime('%Y%m%d%H%M%S')} |

---"""


def format_damage_analysis_section(
    damage_types: list, affected_areas: list, severity_score: int, area: float
) -> str:
    """피해 현황 분석 섹션 포맷"""
    primary_damage = damage_types[0] if damage_types else "일반 피해"
    severity_desc = get_severity_description(severity_score)

    return f"""## 피해 현황 분석

### 피해 부위
- **주요 피해 영역**: {', '.join(affected_areas)}
- **피해 범위**: {area:,.1f} m²

### 피해 유형  
- **주요 피해**: {primary_damage}
- **세부 피해 유형**: {', '.join(damage_types)}
- **피해 심각도**: {severity_score}/5 ({severity_desc})

---"""


def format_repair_methods_section(primary_damage: str) -> str:
    """복구 방법 섹션 포맷"""
    methods = get_repair_methods(primary_damage)
    methods_text = "\n".join([f"{i}. {method}" for i, method in enumerate(methods, 1)])

    return f"""## 복구 방법 및 공종

### 복구 방법
{methods_text}

### 복구 공종
- **주요 공종**: {primary_damage} 보수공사
- **세부 공종**: 
  - 철거공사
  - 보수공사  
  - 방수공사
  - 마감공사

### 공종명 (건설공사 표준품셈 기준)
- **{primary_damage} 보수**: 표준품셈 기준 적용
- **적용 기준**: 국토교통부 건설공사 표준품셈 2024년 기준

---"""


def format_materials_equipment_section(repair_data: Dict[str, Any]) -> str:
    """재료 및 장비 섹션 포맷"""
    materials_text = "\n".join(
        [f"{i}. {material}" for i, material in enumerate(repair_data["materials"], 1)]
    )
    equipment_text = "\n".join(
        [f"{i}. {equipment}" for i, equipment in enumerate(repair_data["equipment"], 1)]
    )

    return f"""## 복구 재료 및 장비

### 주요 자재
{materials_text}

### 필요 장비
{equipment_text}

---"""


def format_labor_section(repair_data: Dict[str, Any]) -> str:
    """인력 구성 섹션 포맷"""
    roles = {
        "특급기능사": "현장 총괄, 기술 지도",
        "고급기능사": "전문 작업 수행",
        "보통인부": "보조 작업, 자재 운반",
    }

    labor_rows = []
    for job_type, count in repair_data["labor"].items():
        role = roles.get(job_type, "작업 수행")
        labor_rows.append(f"| {job_type} | {count}명 | {role} |")

    labor_table = "\n".join(labor_rows)

    return f"""## 인력 구성

### 소요 인력
| 직종 | 인원 | 역할 |
|------|------|------|
{labor_table}

---"""


def format_schedule_section(repair_data: Dict[str, Any]) -> str:
    """공사 기간 섹션 포맷"""
    duration_days = repair_data["duration_days"]
    main_work_days = max(1, duration_days - 2)

    return f"""## 복구 기간

### 예상 공사 기간
- **총 공사 기간**: {duration_days}일
- **작업 단계별 기간**:
  - 준비 및 철거: 1일
  - 주요 보수 작업: {main_work_days}일  
  - 마감 및 정리: 1일

---"""


def format_cost_section(repair_data: Dict[str, Any], area: float) -> str:
    """비용 산정 섹션 포맷"""
    total_material_cost = repair_data["material_cost"]
    total_labor_cost = repair_data["labor_cost"]
    total_cost = total_material_cost + total_labor_cost

    return f"""## 비용 산정

### 자재비 단가
| 구분 | 단가 | 수량 | 금액 |
|------|------|------|------|
| 자재비 | {total_material_cost/area:,.0f}원/m² | {area:,.1f}m² | {total_material_cost:,.0f}원 |

### 노무비  
| 구분 | 단가 | 수량 | 금액 |
|------|------|------|------|
| 노무비 | {total_labor_cost/area:,.0f}원/m² | {area:,.1f}m² | {total_labor_cost:,.0f}원 |

### 총 비용 요약
| 항목 | 금액 | 비율 |
|------|------|------|
| **자재비** | {total_material_cost:,.0f}원 | {(total_material_cost/total_cost)*100:.1f}% |
| **노무비** | {total_labor_cost:,.0f}원 | {(total_labor_cost/total_cost)*100:.1f}% |
| **총 공사비** | **{total_cost:,.0f}원** | **100%** |
| **m²당 단가** | {total_cost/area:,.0f}원/m² | - |

---"""


def format_priority_safety_section(severity_score: int) -> str:
    """우선순위 및 안전 주의사항 섹션 포맷"""
    # 우선순위 결정
    if severity_score >= 4:
        priority_section = """
- **등급**: **긴급 (1순위)**
- **조치 기한**: 즉시 (24시간 이내)
- **권장 조치**: 안전상의 이유로 즉시 전문가 상담 및 응급 조치 필요"""

        safety_section = """
- **즉시 대피 고려**: 구조적 안전성 문제 가능성
- **출입 제한**: 해당 영역 사용 금지
- **전문가 진단**: 구조 엔지니어 정밀 진단 필수
- **응급 조치**: 임시 보강 또는 차단 조치 필요"""

    elif severity_score == 3:
        priority_section = """
- **등급**: **높음 (2순위)**  
- **조치 기한**: 2주 이내
- **권장 조치**: 빠른 시일 내에 수리를 진행하여 피해 확산 방지"""

        safety_section = """
- **주의 깊은 사용**: 하중 제한 및 진동 방지
- **정기 점검**: 주 1회 이상 상태 확인
- **확산 방지**: 방수 처리 등 추가 피해 방지 조치
- **모니터링**: 균열 진행 상황 지속 관찰"""

    else:
        priority_section = """
- **등급**: **보통 (3순위)**
- **조치 기한**: 1개월 이내  
- **권장 조치**: 계획적으로 수리를 진행하되 정기적 점검 실시"""

        safety_section = """
- **일반 안전수칙 준수**: 기본적인 건물 사용 수칙 준수
- **정기 유지보수**: 월 1회 이상 점검 실시
- **예방 조치**: 습도 관리 및 환기 등 예방 관리"""

    return f"""## 수리 우선순위 및 권장사항

### 우선순위{priority_section}

### 안전 주의사항{safety_section}

---"""


def format_additional_info_section() -> str:
    """추가 안내 섹션 포맷"""
    return f"""## 추가 안내

### 정밀 진단 권장
- 본 분석은 AI 기반 1차 진단 결과입니다
- 정확한 진단을 위해서는 전문가의 현장 조사가 필요합니다
- 구조적 안전성 검토가 필요한 경우 구조 엔지니어 상담을 권장합니다

### 표준품셈 적용 안내  
- 본 비용 산정은 건설공사 표준품셈을 기준으로 합니다
- 실제 시공 시 현장 여건에 따라 비용이 변동될 수 있습니다
- 정확한 견적은 전문 시공업체 상담을 통해 확인하시기 바랍니다

### 추가 문의
더 자세한 분석이나 전문가 상담이 필요하시면 언제든지 문의해주세요!

---
"""


def format_comprehensive_analysis_response(
    damage_analysis: Dict[str, Any], area: float
) -> str:
    """종합 분석 응답 포맷 (메인 함수)"""

    # Extract key information
    damage_types = damage_analysis.get("damage_types", ["일반 피해"])
    primary_damage = damage_types[0] if damage_types else "일반 피해"
    severity_score = damage_analysis.get("severity_score", 3)
    affected_areas = damage_analysis.get("affected_areas", ["전체 영역"])
    confidence_level = damage_analysis.get("confidence_level", 0.8)

    # Get standard repair data
    repair_data = get_standard_repair_data(primary_damage, area)

    # Build the comprehensive report by combining all sections
    response_parts = [
        format_basic_info_section(area, confidence_level),
        format_damage_analysis_section(
            damage_types, affected_areas, severity_score, area
        ),
        format_repair_methods_section(primary_damage),
        format_materials_equipment_section(repair_data),
        format_labor_section(repair_data),
        format_schedule_section(repair_data),
        format_cost_section(repair_data, area),
        format_priority_safety_section(severity_score),
        format_additional_info_section(),
    ]

    return "\n".join(response_parts)
