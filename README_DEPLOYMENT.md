# 🏢 건물 피해 분석 AI 시스템

> AI 기반 건물 손상 진단 및 복구 계획 수립 시스템

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)

## 🎯 프로젝트 개요

건물 피해 이미지를 업로드하면 AI가 자동으로 분석하여 피해 유형, 심각도, 복구 비용, 작업 일정을 제공하는 종합 분석 시스템입니다.

### ✨ 주요 기능

- 🔍 **AI 피해 진단**: 9가지 피해 유형 자동 분류 및 5단계 심각도 평가
- 💰 **정밀 비용 산정**: 건설 표준 단가 기반 자재비, 인건비, 장비비 계산
- 📅 **체계적 복구 계획**: 단계별 작업 일정 및 우선순위 수립
- 📊 **시각화**: 비용 분석 차트 및 간트 차트 제공
- 🏗️ **표준 기준 적용**: 국가 건설 표준 시방서 및 품셈 적용

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/building-damage-analysis.git
cd building-damage-analysis
```

### 2. 가상환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 설정

```bash
cp env_example.txt .env
# .env 파일을 편집하여 필요한 설정 추가
```

### 5. 애플리케이션 실행

```bash
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`로 접속하세요.

## 📁 프로젝트 구조

```
building-damage-analysis/
├── 📱 streamlit_app.py          # Streamlit 웹 애플리케이션
├── 🤖 models.py                 # AI 모델 정의 (665M 파라미터)
├── 🔗 langchain_integration.py  # LangChain 체인 구현
├── 🗄️ vector_store.py           # 벡터스토어 및 표준 데이터 검색
├── 📊 data_loader.py            # 데이터 로딩 및 전처리
├── 🎯 trainer.py                # 모델 학습 스크립트
├── ⚙️ config.py                 # 설정 파일
├── 📋 requirements.txt          # Python 의존성
├── 📖 README.md                 # 프로젝트 문서
├── 📚 standard/                 # 표준 데이터 (Excel, PDF)
│   ├── main_data.xlsx
│   ├── damage_risk_index/
│   ├── damage_status_recovery/
│   ├── unit_prices/
│   └── ...
└── 🖼️ learning_pictures/        # 학습 이미지 데이터 (488개)
```

## 🧠 AI 모델 아키텍처

### 멀티모달 구조

- **VisionEncoder**: CLIP (openai/clip-vit-large-patch14)
- **TextEncoder**: DialoGPT (microsoft/DialoGPT-medium)
- **MultimodalProjection**: 비전-텍스트 임베딩 정렬
- **DamageClassifier**: 심각도, 피해유형, 영향영역 분류
- **CrossModalAttention**: 멀티모달 특성 융합

### 출력 구조

- **심각도**: 1-5 등급 (경미한 손상 ~ 완전 파괴)
- **피해 유형**: 균열, 수해, 화재, 지붕, 창문/문, 기초침하, 구조변형, 외벽, 전기/기계
- **영향 영역**: 외벽, 지붕, 기초, 창문, 문, 발코니, 계단, 기타

## 📊 데이터 및 표준

### 학습 데이터

- **이미지**: 488개 건물 피해 사진
- **텍스트**: Excel 기반 매칭 데이터

### 표준 데이터 (벡터스토어)

- **피해위험지수**: 피해 유형별 위험도 기준
- **복구 근거**: 표준 복구 방법 및 시방서
- **단가 정보**: 건설 표준 시장 단가
- **노무비**: 건설업 임금 실태 조사
- **공종명**: 건설공사 표준 품셈

## 🛠️ 기술 스택

### Backend

- **Python 3.11+**: 메인 언어
- **PyTorch**: AI 모델 프레임워크
- **LangChain**: AI 체인 구성
- **ChromaDB**: 벡터 데이터베이스
- **HuggingFace**: 사전 훈련 모델

### Frontend

- **Streamlit**: 웹 애플리케이션 프레임워크
- **Plotly**: 인터랙티브 차트
- **CSS3**: 모던 UI 스타일링

### Data Processing

- **Pandas**: 데이터 처리
- **OpenCV**: 이미지 전처리
- **Albumentations**: 데이터 증강

## 🔧 설정 및 환경변수

### 필수 설정

```bash
# .env 파일
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_TOKEN=your_hf_token
DEVICE=cpu  # 또는 cuda
```

### 선택적 설정

```bash
CACHE_DIR=./cache
LOGS_DIR=./logs
MODEL_PATH=./models/training_20250525_230241/best_model.pt
```

## 📈 성능 지표

### 모델 성능

- **파라미터 수**: 665M (362M 학습가능)
- **추론 시간**: ~30초 (CPU), ~5초 (GPU)
- **정확도**: 피해 유형별 80-95%

### 시스템 성능

- **벡터스토어**: 2536 문서 청크
- **검색 속도**: <1초
- **동시 사용자**: 100+ 지원

## 🚀 배포 옵션

### 1. Streamlit Cloud

```bash
# GitHub 연동 후 자동 배포
# streamlit.io에서 앱 생성
```

### 2. Docker 배포

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

### 3. Heroku 배포

```bash
# Procfile 생성
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

- **이메일**: your-email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)
- **프로젝트 링크**: [https://github.com/your-username/building-damage-analysis](https://github.com/your-username/building-damage-analysis)

## 🙏 감사의 말

- [OpenAI CLIP](https://github.com/openai/CLIP) - 비전 인코더
- [Microsoft DialoGPT](https://github.com/microsoft/DialoGPT) - 텍스트 인코더
- [LangChain](https://github.com/langchain-ai/langchain) - AI 체인 프레임워크
- [Streamlit](https://github.com/streamlit/streamlit) - 웹 애플리케이션 프레임워크

---

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!
