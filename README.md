# 건물 손상 분석 AI 시스템

YOLOv8 + CLIP + Pandas + GPT-4를 활용한 건물 피해 자동 분석 및 보고서 생성 시스템

## 🏗️ 프로젝트 구조

```
tumblr/
├── app/                    # 핵심 애플리케이션
│   ├── analysis_engine.py  # AI 분석 엔진 (YOLOv8 + CLIP)
│   ├── criteria_loader.py  # 건축 기준 데이터 매니저
│   ├── yolo_trainer.py     # YOLO 모델 훈련
│   └── clip_trainer.py     # CLIP 모델 파인튜닝
├── train/                  # 훈련 관련 모든 파일
│   ├── scripts/           # 훈련 스크립트
│   ├── models/            # 훈련된 모델
│   ├── datasets/          # 훈련 데이터셋
│   ├── runs/              # 훈련 실행 결과
│   ├── logs/              # 훈련 로그
│   └── configs/           # 설정 파일
├── criteria/              # 건축 기준 데이터 (Excel)
│   ├── KCS 건설기준 파일들
│   ├── 건물 검사 매뉴얼
│   ├── 손상 위험 지수 기준
│   └── 재해 대응 지침
├── ui/                    # 사용자 인터페이스
├── utils/                 # 유틸리티 함수들
└── streamlit_app.py       # 메인 웹 애플리케이션
```

## 🚀 주요 기능

### 1. AI 기반 건물 손상 분석

- **YOLOv8**: 건물 이미지에서 손상 영역 자동 감지
- **CLIP**: 감지된 영역의 손상 유형 분류 (균열, 수해, 화재 등)
- **커스텀 훈련**: 건물 손상 데이터로 특화 훈련된 모델 사용

### 2. 건축 기준 데이터 활용

- **KCS 건설기준**: 한국건설기준 자동 적용
- **검사 매뉴얼**: 건물 검사 매뉴얼 기반 평가
- **손상 위험 지수**: 체계적인 위험도 평가
- **재해 대응**: 재해별 맞춤 대응 지침

### 3. 종합 보고서 생성

- **Pandas 기반**: Excel 데이터 직접 처리 및 분석
- **자동 매핑**: 손상 유형별 수리 방법 및 비용 산정
- **시각화**: 손상 영역 표시된 이미지 생성
- **전문 보고서**: 건축 전문가 수준의 상세 분석 보고서

## 📋 기술 스택

### AI/ML

- **YOLOv8**: 객체 감지 (ultralytics)
- **CLIP**: 이미지-텍스트 매칭 (OpenAI)
- **PyTorch**: 딥러닝 프레임워크
- **OpenCV**: 컴퓨터 비전

### 데이터 처리

- **Pandas**: Excel 데이터 처리 및 분석
- **NumPy**: 수치 계산
- **openpyxl**: Excel 파일 읽기/쓰기

### 웹 프레임워크

- **Streamlit**: 웹 애플리케이션 프레임워크
- **PIL/Pillow**: 이미지 처리

### 기타

- **LangChain**: LLM 체인 관리 (선택사항)
- **OpenAI API**: GPT-4 보고서 생성 (선택사항)

## 🛠️ 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd tumblr

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 모델 훈련 (선택사항)

```bash
# 훈련 스크립트 실행
cd train/scripts
python train_models.py
```

### 3. 웹 애플리케이션 실행

```bash
# 메인 디렉토리에서
streamlit run streamlit_app.py
```

## 📊 사용 방법

### 1. 이미지 업로드

- 건물 손상 이미지를 업로드
- 지원 형식: JPG, PNG, WebP, AVIF

### 2. 분석 실행

- AI가 자동으로 손상 영역 감지 및 분류
- 건축 기준 데이터와 자동 매핑

### 3. 보고서 확인

- 상세한 손상 분석 보고서 생성
- 수리 방법, 비용, 기간 등 포함
- 시각화된 손상 영역 이미지 제공

## 🔧 커스터마이징

### 모델 재훈련

```bash
# 새로운 데이터로 모델 훈련
cd train/scripts
python train_models.py
```

### 기준 데이터 추가

- `criteria/` 폴더에 새로운 Excel 파일 추가
- `app/criteria_loader.py`에서 파일 매핑 설정

### UI 수정

- `streamlit_app.py`: 메인 인터페이스
- `ui/` 폴더: 추가 UI 컴포넌트

## 📁 주요 파일 설명

### 핵심 모듈

- `app/analysis_engine.py`: AI 분석 파이프라인
- `app/criteria_loader.py`: 건축 기준 데이터 매니저
- `streamlit_app.py`: 웹 애플리케이션 메인

### 훈련 관련

- `train/scripts/train_models.py`: 통합 훈련 스크립트
- `train/models/`: 훈련된 모델 파일들
- `train/datasets/`: 훈련 데이터셋

### 기준 데이터

- `criteria/`: 건축 기준 Excel 파일들
- KCS 건설기준, 검사 매뉴얼, 재해 대응 지침 등

## 🔍 분석 결과 예시

### 감지 가능한 손상 유형

- **균열 손상**: 벽체, 기초 균열
- **수해 손상**: 침수, 누수 피해
- **화재 손상**: 화재로 인한 손상
- **지붕 손상**: 지붕재 파손, 누수
- **창문/문 손상**: 개구부 파손
- **구조 변형**: 건물 구조적 변형
- **외벽 손상**: 외벽 마감재 손상

### 제공 정보

- **손상 위치**: 바운딩 박스로 정확한 위치 표시
- **심각도**: 1-5등급 위험도 평가
- **수리 방법**: 구체적인 복구 방안
- **예상 비용**: 수리 비용 범위
- **소요 기간**: 복구 예상 기간
- **안전 조치**: 즉시 필요한 안전 조치

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

_본 시스템은 AI 기반 분석 결과를 제공하며, 최종 판단은 반드시 건축 전문가의 검토를 통해 확정하시기 바랍니다._
