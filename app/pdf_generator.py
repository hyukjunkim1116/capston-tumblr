"""
건물 손상 분석 PDF 보고서 생성 모듈

이 모듈은 AI 분석 결과를 전문적인 PDF 보고서로 생성합니다.
"""

import io
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# PDF 생성 라이브러리
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        Table,
        TableStyle,
        PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.pdfgen import canvas
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class BuildingDamageReportGenerator:
    """건물 손상 분석 PDF 보고서 생성기"""

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab 라이브러리가 설치되지 않았습니다. 'pip install reportlab' 실행하세요."
            )

        # 페이지 설정
        self.page_size = A4
        self.margin = 2 * cm

        # 한글 폰트 설정
        self._setup_korean_fonts()

        # 스타일 초기화
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        logger.info("PDF 보고서 생성기 초기화 완료")

    def _setup_korean_fonts(self):
        """한글 폰트 설정"""
        try:
            # 프로젝트 내 폰트 우선 사용
            project_fonts = [
                "font/Pretendard-Regular.ttf",  # 프로젝트 내 Pretendard 폰트
            ]

            # 시스템 폰트 (폴백용)
            system_fonts = [
                # macOS
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/Library/Fonts/AppleSDGothicNeo.ttc",
                "/System/Library/Fonts/Helvetica.ttc",
                # Windows
                "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/gulim.ttc",
                # Linux
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]

            all_fonts = project_fonts + system_fonts

            self.korean_font = None
            self.korean_font_bold = None

            for font_path in all_fonts:
                if os.path.exists(font_path):
                    try:
                        # Regular 폰트 등록
                        pdfmetrics.registerFont(TTFont("KoreanFont", font_path))
                        self.korean_font = "KoreanFont"

                        # Bold 폰트 등록 (같은 폰트를 Bold로도 사용)
                        pdfmetrics.registerFont(TTFont("KoreanFont-Bold", font_path))
                        self.korean_font_bold = "KoreanFont-Bold"

                        logger.info(f"한글 폰트 등록 완료: {font_path}")

                        # Pretendard 폰트를 찾았으면 우선적으로 사용
                        if "Pretendard" in font_path:
                            logger.info("Pretendard 폰트 사용 - 한글 지원 최적화")
                            break
                        # 프로젝트 폰트를 찾았으면 시스템 폰트보다 우선
                        elif font_path in project_fonts:
                            break

                    except Exception as e:
                        logger.warning(f"폰트 등록 실패 {font_path}: {e}")
                        continue

            if not self.korean_font:
                # 폴백: 기본 폰트 사용
                self.korean_font = "Helvetica"
                self.korean_font_bold = "Helvetica-Bold"
                logger.warning("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

        except Exception as e:
            logger.error(f"폰트 설정 중 오류: {e}")
            self.korean_font = "Helvetica"
            self.korean_font_bold = "Helvetica-Bold"

    def _setup_custom_styles(self):
        """커스텀 스타일 설정"""
        # 제목 스타일
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=20,
                textColor=colors.HexColor("#1f2937"),
                alignment=TA_CENTER,
                spaceAfter=20,
                fontName=self.korean_font_bold,
            )
        )

        # 섹션 제목 스타일
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading1"],
                fontSize=14,
                textColor=colors.HexColor("#374151"),
                alignment=TA_LEFT,
                spaceAfter=12,
                spaceBefore=20,
                fontName=self.korean_font_bold,
            )
        )

        # 서브섹션 제목 스타일
        self.styles.add(
            ParagraphStyle(
                name="SubHeader",
                parent=self.styles["Heading2"],
                fontSize=12,
                textColor=colors.HexColor("#4b5563"),
                alignment=TA_LEFT,
                spaceAfter=8,
                spaceBefore=12,
                fontName=self.korean_font_bold,
            )
        )

        # 본문 스타일
        self.styles.add(
            ParagraphStyle(
                name="CustomBody",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#374151"),
                alignment=TA_JUSTIFY,
                spaceAfter=6,
                fontName=self.korean_font,
            )
        )

        # 경고 박스 스타일
        self.styles.add(
            ParagraphStyle(
                name="WarningText",
                parent=self.styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#dc2626"),
                alignment=TA_LEFT,
                spaceAfter=6,
                fontName=self.korean_font_bold,
            )
        )

    def generate_report(
        self,
        analysis_result: str,
        image_path: Optional[str] = None,
        area: float = 0.0,
        user_message: str = "",
        damage_areas: Optional[list] = None,
        output_path: Optional[str] = None,
    ) -> io.BytesIO:
        """
        건물 손상 분석 PDF 보고서 생성

        Args:
            analysis_result: AI 분석 결과 텍스트
            image_path: 분석된 이미지 경로
            area: 분석 면적
            user_message: 사용자 메시지
            damage_areas: 피해 영역 데이터
            output_path: PDF 저장 경로 (None이면 BytesIO 반환)

        Returns:
            BytesIO: PDF 데이터 (output_path가 None인 경우)
        """
        try:
            # PDF 버퍼 생성
            buffer = io.BytesIO()

            # 문서 생성
            doc = SimpleDocTemplate(
                buffer,
                pagesize=self.page_size,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin,
                title="건물 손상 분석 보고서",
            )

            # 문서 내용 구성
            story = []

            # 1. 헤더
            story.extend(self._create_header())

            # 2. 기본 정보
            story.extend(self._create_basic_info(area, user_message))

            # 3. 현장사진 (이미지가 있는 경우)
            if image_path and os.path.exists(image_path):
                story.extend(self._create_site_photo_section(image_path))

            # 4. 피해 현황
            story.extend(self._create_damage_status_section(damage_areas or []))

            # 5. 복구 근거
            story.extend(self._create_recovery_basis_section(damage_areas or []))

            # 6. 공정명
            story.extend(self._create_process_name_section(damage_areas or []))

            # 7. 복구 예상 자재
            story.extend(self._create_materials_section(damage_areas or []))

            # 8. 안전 주의사항
            story.extend(self._create_safety_section())

            # 9. 푸터
            story.extend(self._create_footer())

            # PDF 생성
            doc.build(
                story,
                onFirstPage=self._create_page_header,
                onLaterPages=self._create_page_header,
            )

            # 버퍼 포인터 리셋
            buffer.seek(0)

            # 파일로 저장 (요청된 경우)
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(buffer.getvalue())
                logger.info(f"PDF 보고서 저장 완료: {output_path}")
                buffer.seek(0)  # 다시 리셋

            logger.info("PDF 보고서 생성 완료")
            return buffer

        except Exception as e:
            logger.error(f"PDF 보고서 생성 실패: {e}")
            raise

    def _create_header(self) -> list:
        """보고서 헤더 생성"""
        elements = []

        # 메인 제목
        title = Paragraph("건물 손상 분석 보고서", self.styles["CustomTitle"])
        elements.append(title)
        elements.append(Spacer(1, 0.3 * inch))

        # 부제목
        subtitle = Paragraph(
            "AI 기반 건물 구조 안전성 진단", self.styles["SectionHeader"]
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_basic_info(self, area: float, user_message: str) -> list:
        """기본 정보 섹션 생성"""
        elements = []

        elements.append(Paragraph("1. 분석 개요", self.styles["SectionHeader"]))

        # 분석 정보 테이블
        data = [
            ["항목", "내용"],
            ["분석 일시", datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")],
            ["분석 면적", f"{area:.1f} m²"],
            ["분석 요청", user_message if user_message else "건물 손상 분석"],
            ["분석 시스템", "Tumblr AI - YOLOv8 + GPT-4 + CLIP"],
        ]

        table = Table(data, colWidths=[3 * cm, 10 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#374151")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    # 한글 폰트 적용
                    ("FONTNAME", (0, 0), (-1, 0), self.korean_font_bold),  # 헤더는 Bold
                    ("FONTNAME", (0, 1), (-1, -1), self.korean_font),  # 내용은 Regular
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f9fafb")],
                    ),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_site_photo_section(self, image_path: str) -> list:
        """현장사진 섹션 생성"""
        elements = []

        elements.append(Paragraph("2. 현장사진", self.styles["SectionHeader"]))

        try:
            # 이미지 크기 조정
            if PIL_AVAILABLE:
                with PILImage.open(image_path) as img:
                    original_width, original_height = img.size
                    max_width = 12 * cm
                    max_height = 8 * cm

                    # 비율 유지하면서 크기 조정
                    ratio = min(
                        max_width / original_width, max_height / original_height
                    )
                    new_width = original_width * ratio
                    new_height = original_height * ratio
            else:
                new_width, new_height = 12 * cm, 8 * cm

            # 이미지 삽입
            img = Image(image_path, width=new_width, height=new_height)
            elements.append(img)
            elements.append(Spacer(1, 0.2 * inch))

            elements.append(Spacer(1, 0.3 * inch))

        except Exception as e:
            logger.warning(f"이미지 추가 실패: {e}")
            error_text = Paragraph(
                f"이미지 로드 실패: {image_path}", self.styles["WarningText"]
            )
            elements.append(error_text)
            elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_damage_status_section(self, damage_areas: list) -> list:
        """피해 현황 섹션 생성"""
        elements = []

        elements.append(Paragraph("3. 피해 현황", self.styles["SectionHeader"]))

        # 피해 영역 테이블
        data = [
            ["영역", "피해 정도"],
        ]

        for area in damage_areas:
            data.append([area["name"], area["description"]])

        table = Table(data, colWidths=[3 * cm, 10 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#374151")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    # 한글 폰트 적용
                    ("FONTNAME", (0, 0), (-1, 0), self.korean_font_bold),  # 헤더는 Bold
                    ("FONTNAME", (0, 1), (-1, -1), self.korean_font),  # 내용은 Regular
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f9fafb")],
                    ),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_recovery_basis_section(self, damage_areas: list) -> list:
        """복구 근거 섹션 생성"""
        elements = []

        elements.append(Paragraph("4. 복구 근거", self.styles["SectionHeader"]))

        # 복구 근거 텍스트
        recovery_basis = "\n".join([area["basis"] for area in damage_areas])
        recovery_basis_para = Paragraph(recovery_basis, self.styles["CustomBody"])
        elements.append(recovery_basis_para)

        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_process_name_section(self, damage_areas: list) -> list:
        """공정명 섹션 생성"""
        elements = []

        elements.append(Paragraph("5. 공정명", self.styles["SectionHeader"]))

        # 공정명 텍스트
        process_names = "\n".join([area["process"] for area in damage_areas])
        process_names_para = Paragraph(process_names, self.styles["CustomBody"])
        elements.append(process_names_para)

        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_materials_section(self, damage_areas: list) -> list:
        """복구 예상 자재 섹션 생성"""
        elements = []

        elements.append(Paragraph("6. 복구 예상 자재", self.styles["SectionHeader"]))

        # 자재 테이블
        data = [
            ["자재", "용도"],
        ]

        for area in damage_areas:
            for material in area["materials"]:
                data.append([material["name"], material["usage"]])

        table = Table(data, colWidths=[3 * cm, 10 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#374151")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    # 한글 폰트 적용
                    ("FONTNAME", (0, 0), (-1, 0), self.korean_font_bold),  # 헤더는 Bold
                    ("FONTNAME", (0, 1), (-1, -1), self.korean_font),  # 내용은 Regular
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f9fafb")],
                    ),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_safety_section(self) -> list:
        """안전 주의사항 섹션 생성"""
        elements = []

        elements.append(
            Paragraph("7. 중요 안전 주의사항", self.styles["SectionHeader"])
        )

        safety_warnings = [
            "이 분석 결과는 AI 기반 예비 진단으로, 최종 안전성 판단은 전문가의 현장 조사가 필요합니다.",
            "구조적 손상이 의심되는 경우 즉시 건축구조기술사 또는 안전진단전문기관에 정밀 조사를 의뢰하세요.",
            "긴급 안전 조치가 필요한 경우 해당 구역의 출입을 금지하고 관계 당국에 신고하세요.",
            "수리 및 보강 작업은 관련 법규와 기준을 준수하여 시행하세요.",
            "이 보고서는 참고용으로만 사용하며, 실제 공사는 전문 업체에 의뢰하세요.",
        ]

        for warning in safety_warnings:
            warning_para = Paragraph(f"※ {warning}", self.styles["WarningText"])
            elements.append(warning_para)
            elements.append(Spacer(1, 0.05 * inch))

        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_footer(self) -> list:
        """보고서 푸터 생성"""
        elements = []

        elements.append(Spacer(1, 0.3 * inch))

        # 면책 조항
        disclaimer = """
        본 보고서는 AI 기술을 활용한 예비 분석 결과로, 참고용으로만 사용해야 합니다. 
        최종적인 구조 안전성 판단 및 보수 방법 결정은 반드시 전문가의 현장 조사와 정밀 진단을 통해 수행되어야 합니다.
        
        Tumblr AI는 이 분석 결과의 정확성이나 완전성에 대해 보증하지 않으며, 
        이 보고서를 근거로 한 어떠한 조치나 결정에 대해서도 책임을 지지 않습니다.
        """

        disclaimer_para = Paragraph(disclaimer, self.styles["CustomBody"])
        elements.append(disclaimer_para)

        # 생성 정보
        generation_info = f"보고서 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Tumblr AI v1.0"
        info_para = Paragraph(generation_info, self.styles["CustomBody"])
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(info_para)

        return elements

    def _create_page_header(self, canvas, doc):
        """페이지 헤더 생성"""
        canvas.saveState()

        # 헤더 라인
        canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
        canvas.setLineWidth(1)
        canvas.line(
            doc.leftMargin,
            doc.height + doc.topMargin - 0.5 * cm,
            doc.width + doc.leftMargin,
            doc.height + doc.topMargin - 0.5 * cm,
        )

        # 페이지 번호
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        page_text = f"페이지 {doc.page}"
        canvas.drawRightString(
            doc.width + doc.leftMargin, doc.height + doc.topMargin - 0.3 * cm, page_text
        )

        canvas.restoreState()


def create_damage_report_pdf(
    analysis_result: str,
    image_path: Optional[str] = None,
    area: float = 0.0,
    user_message: str = "",
    damage_areas: Optional[list] = None,
    output_path: Optional[str] = None,
) -> io.BytesIO:
    """
    건물 손상 분석 PDF 보고서 생성 (편의 함수)

    Args:
        analysis_result: AI 분석 결과
        image_path: 분석 이미지 경로
        area: 분석 면적
        user_message: 사용자 메시지
        damage_areas: 피해 영역 데이터
        output_path: 저장 경로 (None이면 BytesIO 반환)

    Returns:
        BytesIO: PDF 데이터
    """
    generator = BuildingDamageReportGenerator()
    return generator.generate_report(
        analysis_result=analysis_result,
        image_path=image_path,
        area=area,
        user_message=user_message,
        damage_areas=damage_areas,
        output_path=output_path,
    )
