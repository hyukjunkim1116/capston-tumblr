# 🏠 건물 피해 분석 AI 시스템 (FAISS 기반)

## 📋 개요

FAISS 기반 벡터 스토어를 사용하는 건물 피해 분석 AI 시스템입니다. 사용자가 업로드한 건물 피해 이미지를 분석하여 상세한 피해 보고서를 생성합니다.

## ✨ 주요 기능

- 🔍 **AI 기반 피해 분석**: 딥러닝 모델을 통한 정확한 피해 유형 및 심각도 분석
- 📊 **표준 데이터 기반 평가**: 건설공사 표준품셈 기준 복구 방법 및 비용 산정
- 💰 **비용 추정**: 피해 면적과 유형에 따른 수리 비용 자동 계산
- 📋 **상세 보고서**: 우선순위, 안전 주의사항, 권장 조치 포함
- 🚀 **FAISS 벡터 스토어**: 빠르고 안정적인 표준 데이터 검색
- 💬 **ChatGPT 스타일 UI**: 직관적인 대화형 인터페이스

## 🏗️ 시스템 아키텍처

```
사용자 이미지 업로드
        ↓
AI 모델 피해 분석
        ↓
FAISS 벡터 스토어 → 표준 데이터 검색
        ↓
종합 분석 보고서 생성
```

## 🛠️ 기술 스택

- **Frontend**: Streamlit (ChatGPT 스타일 UI)
- **Backend**: Python, FastAPI
- **AI/ML**: PyTorch, LangChain, HuggingFace Transformers
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

### 2. 환경변수 설정

```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

### 3. FAISS 인덱스 빌드

```bash
# 표준 데이터 인덱싱
python build_faiss_index.py
```

### 4. 애플리케이션 실행

```bash
# Streamlit 앱 실행
streamlit run streamlit_app.py
```

## 🚀 배포

### Streamlit Cloud

1. **requirements.txt 확인**
2. **환경변수 설정** (OPENAI_API_KEY)
3. **FAISS 인덱스 포함** (자동으로 Git에 포함됨)

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python build_faiss_index.py

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📊 성능 지표

- **FAISS 인덱스 크기**: ~7MB
- **문서 청크 수**: 2,536개
- **로딩 시간**: 3-5초
- **검색 속도**: ~100ms
- **메모리 사용량**: ~200MB

## 📁 프로젝트 구조

```
tumblr/
├── streamlit_app.py          # 메인 애플리케이션
├── vector_store_faiss.py     # FAISS 벡터 스토어
├── build_faiss_index.py      # 인덱스 빌드 스크립트
├── langchain_integration.py  # LangChain 통합
├── models.py                 # AI 모델 정의
├── config.py                 # 설정 파일
├── ui/
│   └── ui_components.py      # UI 컴포넌트
├── standard/                 # 표준 데이터
│   ├── faiss_index/         # FAISS 인덱스 (7MB)
│   ├── metadata.json        # 메타데이터 (445KB)
│   ├── main_data.xlsx       # 메인 데이터
│   └── */                   # 각종 표준 데이터
└── requirements.txt         # 의존성 목록
```

## 🔧 사용법

1. **이미지 업로드**: 건물 피해 사진 업로드
2. **면적 입력**: 피해 면적을 m² 단위로 입력
3. **질문 입력**: 피해에 대한 구체적인 질문
4. **분석 결과**: AI가 생성한 상세 분석 보고서 확인

## 📋 분석 결과 포함 항목

- **피해 유형 분석**: 균열, 누수, 화재 등
- **심각도 평가**: 1-5 단계 평가
- **수리 우선순위**: 긴급/높음/보통
- **예상 수리 비용**: 면적 기반 자동 계산
- **안전 주의사항**: 즉시 조치 필요 사항
- **표준 근거**: 건설공사 표준품셈 기준

## 🔍 Vector Store 상태 확인

UI 사이드바에서 실시간 상태 확인:

- ✅ **FAISS 연결됨**: 정상 동작
- ⚠️ **기본 모드**: 제한된 기능

## 🆘 문제 해결

### FAISS 인덱스 오류

```bash
# 인덱스 재빌드
python build_faiss_index.py
```

### 메모리 부족

```python
# 배치 크기 조정
encode_kwargs={"batch_size": 8}
```

### 임베딩 모델 오류

```python
# 더 작은 모델 사용
model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

## 📚 참고 문서

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - 상세 배포 가이드
- [표준 데이터 구조](standard/) - 건설공사 표준품셈 데이터

## 🤝 기여

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**💡 참고**: 이 시스템은 FAISS 기반으로 SQLite 의존성 없이 안정적으로 배포할 수 있습니다.
