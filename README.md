# 🏠 건물 피해 분석 AI 시스템 (최적화 버전)

## 📋 개요

최적화된 건물 피해 분석 AI 시스템입니다. 사용자가 업로드한 건물 피해 이미지를 분석하여 상세한 피해 보고서를 생성합니다.

## ✨ 주요 기능

- 🔍 **AI 기반 피해 분석**: 최적화된 딥러닝 모델을 통한 정확한 피해 유형 및 심각도 분석
- 📊 **기준 데이터 기반 평가**: 건설공사 표준품셈 기준 복구 방법 및 비용 산정
- 💰 **비용 추정**: 피해 면적과 유형에 따른 수리 비용 자동 계산
- 📋 **상세 보고서**: 우선순위, 안전 주의사항, 권장 조치 포함
- 🚀 **FAISS 벡터 스토어**: 빠르고 안정적인 기준 데이터 검색
- 💬 **ChatGPT 스타일 UI**: 직관적인 대화형 인터페이스

## 🏗️ 시스템 아키텍처

```
사용자 이미지 업로드
        ↓
AI 모델 피해 분석 (최적화된 모델)
        ↓
FAISS 벡터 스토어 → 기준 데이터 검색
        ↓
종합 분석 보고서 생성
```

## 🛠️ 기술 스택

- **Frontend**: Streamlit (ChatGPT 스타일 UI)
- **Backend**: Python, FastAPI
- **AI/ML**: PyTorch, LangChain, HuggingFace Transformers (최적화)
- **Vector Store**: FAISS (배포 최적화)
- **Embeddings**: sentence-transformers
- **Data Processing**: Pandas, OpenCV

## 📦 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd tumblr

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.streamlit/secrets.toml` 파일을 생성하고 다음 내용을 추가:

```toml
[general]
OPENAI_API_KEY = "your-openai-api-key"
HUGGINGFACE_TOKEN = "your-huggingface-token"
```

### 3. FAISS 인덱스 빌드 (선택사항)

```bash
# 기준 데이터로부터 FAISS 인덱스 빌드
python build_index.py
```

### 4. 애플리케이션 실행

```bash
# Streamlit 앱 실행
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 사용할 수 있습니다.

## 📁 프로젝트 구조 (최적화)

```
tumblr/
├── streamlit_app.py          # 메인 애플리케이션
├── build_index.py            # FAISS 인덱스 빌드 스크립트
├── requirements.txt          # 의존성 목록
├── README.md                 # 프로젝트 문서
├── DEPLOYMENT_GUIDE.md       # 배포 가이드
├──
├── analysis/                 # 분석 모듈 (LangChain 통합)
│   ├── __init__.py
│   └── integration.py        # LangChain 체인 및 분석 로직
├──
├── faiss/                    # FAISS 벡터 스토어 모듈
│   ├── __init__.py
│   ├── vector_store.py       # FAISS 벡터 스토어 구현
│   └── index_builder.py      # 인덱스 빌드 로직
├──
├── ml/                       # 머신러닝 모듈 (최적화)
│   ├── __init__.py
│   ├── models.py             # 최적화된 AI 모델 정의
│   ├── trainer.py            # 모델 훈련 스크립트
│   └── data_loader.py        # 데이터 로딩 유틸리티
├──
├── utils/                    # 유틸리티 모듈
│   ├── __init__.py
│   ├── config.py             # 최적화된 설정 파일
│   └── env_checker.py        # 환경 체크 유틸리티
├──
├── ui/                       # UI 컴포넌트 모듈
│   ├── __init__.py
│   └── ui_components.py      # Streamlit UI 컴포넌트
├──
├── models/                   # 훈련된 모델 저장소
│   └── best_model/           # 최적 성능 모델
│       ├── best_model.pt     # 최적 모델 파일
│       ├── final_model.pt    # 최종 모델 파일
│       └── training_info.json # 훈련 정보
├──
├── criteria/                 # 기준 데이터 (기존 standard 대체)
│   ├── main_data.xlsx        # 주요 기준 데이터
│   ├── faiss_index/          # FAISS 인덱스 파일들
│   ├── damage_risk_index/    # 피해 위험도 지수
│   ├── damage_status_recovery/ # 피해 상태별 복구 방법
│   ├── location_scores/      # 위치별 점수
│   ├── unit_prices/          # 단가 정보
│   ├── labor_costs/          # 인건비 정보
│   └── work_types/           # 공종별 정보
├──
├── learning_data/            # 학습 데이터
│   ├── learning_texts.xlsx   # 학습용 텍스트 데이터
│   └── learning_pictures/    # 학습용 이미지 데이터
├──
├── cache/                    # 캐시 디렉토리
├── logs/                     # 로그 파일
└── .streamlit/               # Streamlit 설정
    └── secrets.toml          # 환경 변수 설정
```

## 🚀 성능 최적화

### 모델 최적화

- **경량화된 모델**: CLIP-ViT-Base-Patch32 사용 (기존 Large 대비 50% 크기 감소)
- **한국어 특화**: KLUE-BERT-Base 사용으로 한국어 처리 성능 향상
- **메모리 효율성**: 배치 크기 및 차원 최적화

### 데이터 처리 최적화

- **이미지 크기 최적화**: 224x224로 표준화
- **캐싱 시스템**: Streamlit 캐싱으로 반복 처리 최적화
- **FAISS 인덱스**: 빠른 벡터 검색을 위한 최적화된 인덱스

### 코드 최적화

- **불필요한 임포트 제거**: 메모리 사용량 최소화
- **모듈화**: 기능별 명확한 분리로 유지보수성 향상
- **에러 처리**: 강화된 예외 처리 및 폴백 메커니즘

## 🔧 주요 개선사항

### 1. 프로젝트 구조 개선

- 기능별 모듈 분리 (analysis, faiss, ml, utils, ui)
- 명확한 의존성 관리
- 표준화된 폴더 구조

### 2. 모델 성능 최적화

- 최적 성능 모델만 유지 (best_model)
- 경량화된 아키텍처
- 효율적인 메모리 사용

### 3. 데이터 관리 개선

- standard → criteria 폴더로 명확한 명명
- 체계적인 데이터 분류
- 효율적인 벡터 스토어 관리

## 📊 시스템 요구사항

### 최소 요구사항

- **RAM**: 4GB 이상
- **저장공간**: 10GB 이상
- **Python**: 3.8 이상

### 권장 요구사항

- **RAM**: 8GB 이상
- **저장공간**: 20GB 이상
- **GPU**: CUDA 지원 (선택사항)

## 🔍 사용 방법

1. **이미지 업로드**: 분석할 건물 피해 이미지를 업로드
2. **면적 입력**: 피해 면적을 m² 단위로 입력
3. **분석 요청**: 구체적인 분석 요청사항 입력
4. **결과 확인**: AI가 생성한 상세 분석 보고서 확인

## 🛡️ 보안 및 개인정보

- 업로드된 이미지는 분석 후 자동 삭제
- 개인정보는 수집하지 않음
- 모든 데이터는 로컬에서 처리

## 📞 지원 및 문의

- **이슈 리포트**: GitHub Issues 활용
- **기능 요청**: Pull Request 환영
- **문의사항**: 프로젝트 관리자에게 연락

---

_📅 최종 업데이트: 2025년 5월 31일_  
_🤖 시스템 버전: Tumblr AI v2.0 Optimized_
