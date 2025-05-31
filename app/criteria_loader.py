"""
건축 기준 데이터 로더 및 분석 모듈
criteria 폴더의 엑셀 파일들을 활용하여 건물 손상 분석에 필요한 기준 정보 제공
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class CriteriaDataManager:
    """건축 기준 데이터 관리자"""

    def __init__(self, criteria_path: str = "criteria"):
        self.criteria_path = Path(criteria_path)
        self.data_cache = {}
        self.kcs_standards = {}
        self.inspection_data = {}
        self.damage_criteria = {}

        self._load_all_criteria()

    def _load_all_criteria(self):
        """모든 기준 데이터 로드"""
        try:
            # KCS 건설기준 파일들
            kcs_files = {
                "building_construction": "KCS41_10_00_Building_Construction.xlsx",
                "building_concrete": "KCS_41_30_01_BuildingConcrete construction.xlsx",
                "cast_in_place": "KCS41_30_02_Cast-in-place construction.xlsx",
                "high_durability": "KCS41_30_03_High-durability concrete construction.xlsx",
                "freeze_thaw": "KCS41_30_04_Concrete work subject to freeze-thaw.xlsx",
                "simple_concrete": "KCS41_30_05_Simple Concrete Construction.xlsx",
                "nuclear_concrete": "KCS_41_30_06_Nuclear Power Plant Concrete Construction.xlsx",
                "kcs_main": "KCS_41_30_10.xlsx",
            }

            # 검사 및 평가 기준 파일들
            inspection_files = {
                "inspection_manual": "building_inspection_manual.xlsx",
                "damage_risk_criteria": "damage_risk_index_criteria.xlsx",
                "damage_risk_index": "by_damage_risk_index.xlsx",
            }

            # 기타 기준 파일들
            other_files = {
                "building_standards": "rules_about_building_standards.xlsx",
                "construction_types": "reason_of_selection_type_of_construction.xlsx",
                "instructions": "instructions.xlsx",
                "disaster_response": "disaster_response.xlsx",
            }

            # KCS 기준 로드
            for key, filename in kcs_files.items():
                self._load_excel_file(key, filename, self.kcs_standards)

            # 검사 기준 로드
            for key, filename in inspection_files.items():
                self._load_excel_file(key, filename, self.inspection_data)

            # 기타 기준 로드
            for key, filename in other_files.items():
                self._load_excel_file(key, filename, self.data_cache)

            logger.info(
                f"기준 데이터 로드 완료: KCS {len(self.kcs_standards)}개, 검사 {len(self.inspection_data)}개, 기타 {len(self.data_cache)}개"
            )

        except Exception as e:
            logger.error(f"기준 데이터 로드 오류: {e}")

    def _load_excel_file(self, key: str, filename: str, target_dict: Dict):
        """개별 엑셀 파일 로드"""
        file_path = self.criteria_path / filename
        if file_path.exists():
            try:
                # 여러 시트가 있을 수 있으므로 모든 시트 로드
                excel_data = pd.read_excel(file_path, sheet_name=None)
                target_dict[key] = excel_data
                logger.info(f"{filename} 로드 완료 ({len(excel_data)} 시트)")
            except Exception as e:
                logger.warning(f"{filename} 로드 실패: {e}")
        else:
            logger.warning(f"파일 없음: {filename}")

    def get_damage_assessment_criteria(self, damage_type: str) -> Dict[str, Any]:
        """피해 유형별 평가 기준 조회"""
        try:
            criteria = {}

            # 손상 위험 지수 기준에서 조회
            if "damage_risk_criteria" in self.inspection_data:
                risk_data = self.inspection_data["damage_risk_criteria"]

                # 첫 번째 시트 사용 (보통 메인 데이터)
                if isinstance(risk_data, dict):
                    main_sheet = list(risk_data.values())[0]
                else:
                    main_sheet = risk_data

                # 피해 유형과 매칭되는 행 찾기
                damage_keywords = {
                    "crack damage": ["균열", "크랙", "갈라짐"],
                    "water damage": ["수해", "침수", "누수", "물"],
                    "fire damage": ["화재", "불", "연소"],
                    "roof damage": ["지붕", "옥상", "루프"],
                    "window damage": ["창문", "유리", "윈도우"],
                    "door damage": ["문", "도어", "출입구"],
                    "foundation damage": ["기초", "파운데이션", "토대"],
                    "structural deformation": ["구조", "변형", "틀어짐"],
                    "facade damage": ["외벽", "파사드", "외관"],
                }

                keywords = damage_keywords.get(damage_type, [damage_type])

                for keyword in keywords:
                    matching_rows = main_sheet[
                        main_sheet.astype(str)
                        .apply(lambda x: x.str.contains(keyword, case=False, na=False))
                        .any(axis=1)
                    ]

                    if not matching_rows.empty:
                        row = matching_rows.iloc[0]
                        criteria.update(
                            {
                                "severity_levels": self._extract_severity_levels(row),
                                "assessment_method": str(
                                    row.get("평가방법", "육안검사")
                                ),
                                "risk_factors": self._extract_risk_factors(row),
                                "source": "damage_risk_criteria",
                            }
                        )
                        break

            # 건물 검사 매뉴얼에서 추가 정보
            if "inspection_manual" in self.inspection_data:
                manual_data = self.inspection_data["inspection_manual"]

                if isinstance(manual_data, dict):
                    main_sheet = list(manual_data.values())[0]
                else:
                    main_sheet = manual_data

                # 수리 방법 및 비용 정보 추가
                repair_info = self._get_repair_specifications(damage_type, main_sheet)
                criteria.update(repair_info)

            return criteria if criteria else self._get_default_criteria(damage_type)

        except Exception as e:
            logger.error(f"피해 평가 기준 조회 오류: {e}")
            return self._get_default_criteria(damage_type)

    def _extract_severity_levels(self, row: pd.Series) -> Dict[str, str]:
        """심각도 레벨 추출"""
        severity_map = {}

        # 컬럼명에서 심각도 관련 정보 찾기
        for col in row.index:
            col_str = str(col).lower()
            if any(
                keyword in col_str for keyword in ["등급", "단계", "레벨", "심각도"]
            ):
                severity_map[col] = str(row[col])

        return (
            severity_map
            if severity_map
            else {
                "1등급": "경미한 손상",
                "2등급": "보통 손상",
                "3등급": "심각한 손상",
                "4등급": "위험 손상",
                "5등급": "즉시 조치 필요",
            }
        )

    def _extract_risk_factors(self, row: pd.Series) -> List[str]:
        """위험 요소 추출"""
        risk_factors = []

        for col in row.index:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ["위험", "주의", "요인", "원인"]):
                value = str(row[col])
                if value and value != "nan":
                    risk_factors.append(value)

        return risk_factors if risk_factors else ["구조적 안전성 검토 필요"]

    def _get_repair_specifications(
        self, damage_type: str, manual_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """수리 사양 정보 추출"""
        try:
            # 피해 유형별 한국어 키워드
            damage_keywords = {
                "crack damage": "균열",
                "water damage": "수해",
                "fire damage": "화재",
                "roof damage": "지붕",
                "window damage": "창문",
                "door damage": "문",
                "foundation damage": "기초",
                "structural deformation": "구조",
                "facade damage": "외벽",
            }

            keyword = damage_keywords.get(damage_type, damage_type)

            # 키워드와 매칭되는 행 찾기
            matching_rows = manual_df[
                manual_df.astype(str)
                .apply(lambda x: x.str.contains(keyword, case=False, na=False))
                .any(axis=1)
            ]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]

                return {
                    "repair_method": str(row.get("수리방법", f"{keyword} 전문 복구")),
                    "materials": str(row.get("필요자재", "상황별 맞춤 자재")),
                    "cost_estimate": str(row.get("예상비용", "현장 견적 필요")),
                    "duration": str(row.get("소요기간", "1-3주")),
                    "contractor_type": str(row.get("시공업체", "전문 건설업체")),
                    "safety_measures": str(row.get("안전조치", "작업 중 안전 확보")),
                }

            return self._get_default_repair_specs(damage_type)

        except Exception as e:
            logger.error(f"수리 사양 추출 오류: {e}")
            return self._get_default_repair_specs(damage_type)

    def _get_default_criteria(self, damage_type: str) -> Dict[str, Any]:
        """기본 평가 기준"""
        return {
            "severity_levels": {
                "1등급": "경미한 손상",
                "2등급": "보통 손상",
                "3등급": "심각한 손상",
                "4등급": "위험 손상",
                "5등급": "즉시 조치 필요",
            },
            "assessment_method": "육안검사 및 전문가 진단",
            "risk_factors": ["구조적 안전성 검토 필요"],
            "repair_method": f"{damage_type} 전문 복구",
            "materials": "상황별 맞춤 자재",
            "cost_estimate": "현장 견적 필요",
            "duration": "1-3주",
            "source": "default",
        }

    def _get_default_repair_specs(self, damage_type: str) -> Dict[str, Any]:
        """기본 수리 사양"""
        return {
            "repair_method": f"{damage_type} 전문 복구",
            "materials": "상황별 맞춤 자재",
            "cost_estimate": "현장 견적 필요",
            "duration": "1-3주",
            "contractor_type": "전문 건설업체",
            "safety_measures": "작업 중 안전 확보",
        }

    def get_kcs_standards(
        self, construction_type: str = "building_concrete"
    ) -> Dict[str, Any]:
        """KCS 건설기준 조회"""
        try:
            if construction_type in self.kcs_standards:
                kcs_data = self.kcs_standards[construction_type]

                # 첫 번째 시트의 주요 정보 추출
                if isinstance(kcs_data, dict):
                    main_sheet = list(kcs_data.values())[0]
                else:
                    main_sheet = kcs_data

                return {
                    "standard_name": construction_type,
                    "total_rows": len(main_sheet),
                    "columns": list(main_sheet.columns),
                    "sample_data": main_sheet.head(3).to_dict("records"),
                    "applicable_scope": "건축 콘크리트 공사",
                }

            return {"error": f"KCS 기준 '{construction_type}' 없음"}

        except Exception as e:
            logger.error(f"KCS 기준 조회 오류: {e}")
            return {"error": str(e)}

    def get_disaster_response_guidelines(self) -> Dict[str, Any]:
        """재해 대응 지침 조회"""
        try:
            if "disaster_response" in self.data_cache:
                disaster_data = self.data_cache["disaster_response"]

                if isinstance(disaster_data, dict):
                    main_sheet = list(disaster_data.values())[0]
                else:
                    main_sheet = disaster_data

                # 재해 유형별 대응 방안 추출
                response_guidelines = {}

                for _, row in main_sheet.iterrows():
                    disaster_type = str(row.get("재해유형", "일반"))
                    response = str(row.get("대응방안", "전문가 상담"))
                    priority = str(row.get("우선순위", "보통"))

                    response_guidelines[disaster_type] = {
                        "response_method": response,
                        "priority": priority,
                        "immediate_actions": str(row.get("즉시조치", "안전 확보")),
                    }

                return response_guidelines

            return {"error": "재해 대응 데이터 없음"}

        except Exception as e:
            logger.error(f"재해 대응 지침 조회 오류: {e}")
            return {"error": str(e)}

    def generate_comprehensive_report(self, damage_analysis: Dict[str, Any]) -> str:
        """종합 분석 보고서 생성 (pandas 기반)"""
        try:
            report_sections = []

            # 1. 피해 현황 요약
            damage_areas = damage_analysis.get("damage_areas", [])
            total_damages = len(
                [d for d in damage_areas if d.get("damage_type") != "normal building"]
            )

            report_sections.append(
                f"""
피해 현황 요약
- **총 감지 영역**: {len(damage_areas)}개
- **피해 영역**: {total_damages}개
- **정상 영역**: {len(damage_areas) - total_damages}개
"""
            )

            # 2. 피해 유형별 상세 분석
            if total_damages > 0:
                report_sections.append("피해 유형별 상세 분석")

                for i, damage in enumerate(damage_areas):
                    if damage.get("damage_type") != "normal building":
                        damage_type = damage.get("damage_type", "unknown")
                        confidence = damage.get("confidence", 0.0)

                        # 기준 데이터에서 평가 기준 조회
                        criteria = self.get_damage_assessment_criteria(damage_type)

                        # 심각도 계산 (confidence 기반)
                        severity_level = min(5, max(1, int(confidence * 5) + 1))
                        severity_desc = (
                            list(criteria.get("severity_levels", {}).values())[
                                severity_level - 1
                            ]
                            if criteria.get("severity_levels")
                            else "보통 손상"
                        )

                        report_sections.append(
                            f"""
피해 영역 {i+1}: {damage.get("damage_type_kr", damage_type)}
- **신뢰도**: {confidence:.2f} ({confidence*100:.1f}%)
- **심각도**: {severity_level}등급 - {severity_desc}
- **평가 방법**: {criteria.get("assessment_method", "육안검사")}
- **수리 방법**: {criteria.get("repair_method", "전문가 상담")}
- **예상 기간**: {criteria.get("duration", "1-3주")}
- **예상 비용**: {criteria.get("cost_estimate", "견적 필요")}
"""
                        )

            # 3. KCS 건설기준 적용
            kcs_info = self.get_kcs_standards("building_concrete")
            if "error" not in kcs_info:
                report_sections.append(
                    f"""
적용 건설기준 (KCS)
- **기준명**: {kcs_info.get("standard_name", "건축 콘크리트")}
- **적용 범위**: {kcs_info.get("applicable_scope", "건축 공사")}
- **기준 항목**: {kcs_info.get("total_rows", 0)}개 조항
"""
                )

            # 4. 재해 대응 지침
            disaster_guidelines = self.get_disaster_response_guidelines()
            if "error" not in disaster_guidelines:
                report_sections.append("재해 대응 지침")
                for disaster_type, guideline in list(disaster_guidelines.items())[
                    :3
                ]:  # 상위 3개만
                    report_sections.append(
                        f"""
{disaster_type}
- **대응 방법**: {guideline.get("response_method", "전문가 상담")}
- **우선순위**: {guideline.get("priority", "보통")}
- **즉시 조치**: {guideline.get("immediate_actions", "안전 확보")}
"""
                    )

            # 5. 종합 권고사항
            report_sections.append(
                f"""
종합 권고사항

즉시 조치 사항
- 피해 영역 {total_damages}개소에 대한 안전 점검 실시
- 구조적 안전성 확보를 위한 임시 조치
- 전문가 정밀 진단 의뢰

복구 계획
1. **1단계**: 안전 확보 및 응급 조치 (1주)
2. **2단계**: 상세 설계 및 자재 준비 (2주) 
3. **3단계**: 본격 복구 공사 실시 (4-8주)

참고 기준
- KCS 건설기준 적용
- 건물 검사 매뉴얼 준수
- 재해 대응 지침 반영

*본 보고서는 AI 분석 결과이며, 최종 판단은 전문가 검토를 통해 확정하시기 바랍니다.*
"""
            )

            return "\n".join(report_sections)

        except Exception as e:
            logger.error(f"종합 보고서 생성 오류: {e}")
            return f"보고서 생성 중 오류 발생: {str(e)}"

    def get_summary_stats(self) -> Dict[str, Any]:
        """로드된 기준 데이터 요약 통계"""
        return {
            "kcs_standards_count": len(self.kcs_standards),
            "inspection_data_count": len(self.inspection_data),
            "other_criteria_count": len(self.data_cache),
            "total_files_loaded": len(self.kcs_standards)
            + len(self.inspection_data)
            + len(self.data_cache),
            "kcs_files": list(self.kcs_standards.keys()),
            "inspection_files": list(self.inspection_data.keys()),
            "other_files": list(self.data_cache.keys()),
        }


# 전역 인스턴스 (싱글톤 패턴)
_criteria_manager = None


def get_criteria_manager() -> CriteriaDataManager:
    """기준 데이터 매니저 인스턴스 반환"""
    global _criteria_manager
    if _criteria_manager is None:
        _criteria_manager = CriteriaDataManager()
    return _criteria_manager
