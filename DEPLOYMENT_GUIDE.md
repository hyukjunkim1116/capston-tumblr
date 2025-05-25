# 🚀 FAISS 기반 건물 피해 분석 시스템 배포 가이드

## 📋 개요

이 가이드는 FAISS 기반 건물 피해 분석 AI 시스템을 배포 환경에서 안정적으로 운영하기 위한 설정 방법을 설명합니다.

## 🏗️ 아키텍처

### Vector Store 전략

- **FAISS**: 메인 벡터 스토어 (SQLite 의존성 없음, 배포 최적화)
- **Fallback Mode**: Vector Store 실패 시 기본 기능 제공

### 시스템 구조

```
FAISS Vector Store → 성공 → 정상 동작
                  → 실패 → Fallback Mode (제한된 기능)
```

## 🛠️ 배포 전 준비

### 1. FAISS 인덱스 빌드

```bash
# 로컬에서 FAISS 인덱스 미리 빌드
python build_faiss_index.py

# 빌드 결과 확인
ls -la standard/faiss_index/
ls -la standard/metadata.json
```

### 2. 파일 구조 확인

```
standard/
├── faiss_index/          # FAISS 벡터 인덱스 (7MB)
├── metadata.json         # 문서 메타데이터 (445KB)
├── main_data.xlsx        # 메인 데이터
├── damage_risk_index/    # 피해 위험 지수
├── damage_status_recovery/ # 피해 복구 기준
├── location_scores/      # 위치 점수
├── unit_prices/         # 단가 정보
├── labor_costs/         # 노무비
└── work_types/          # 공종 정보
```

## 🌐 배포 환경별 설정

### Streamlit Cloud

1. **requirements.txt 확인**

```txt
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
streamlit>=1.28.0
```

2. **환경변수 설정**

```bash
OPENAI_API_KEY=your_api_key
DEVICE=cpu
```

### Heroku

1. **Procfile 생성**

```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **runtime.txt 설정**

```
python-3.11.0
```

### Docker

```dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 앱 파일 복사
COPY . .

# FAISS 인덱스 빌드 (선택사항)
RUN python build_faiss_index.py

# 포트 노출
EXPOSE 8501

# 앱 실행
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔧 Vector Store 상태 모니터링

### UI에서 확인

사이드바에서 현재 Vector Store 상태를 확인할 수 있습니다:

- ✅ **FAISS 연결됨** - 정상 동작
- ⚠️ **기본 모드** - 제한된 기능

### 프로그래밍 방식 확인

```python
from vector_store_faiss import create_faiss_vector_store

vector_store = create_faiss_vector_store()
if vector_store and vector_store.vectorstore:
    print("✅ FAISS vector store available")
    print(f"📊 Documents: {len(vector_store.documents_metadata)}")
else:
    print("❌ FAISS vector store not available")
```

## 🚨 문제 해결

### 1. FAISS 인덱스 로딩 실패

**증상**: "FAISS index not found" 오류

**해결책**:

```bash
# 로컬에서 인덱스 재빌드
python build_faiss_index.py

# Git에 커밋
git add standard/faiss_index/ standard/metadata.json
git commit -m "Add FAISS index for deployment"
```

### 2. 임베딩 모델 로딩 실패

**증상**: HuggingFace 모델 다운로드 실패

**해결책**:

```python
# 더 작은 모델 사용
model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

### 3. 메모리 부족

**증상**: 임베딩 모델 로딩 실패

**해결책**:

```python
# 배치 크기 줄이기
encode_kwargs={"batch_size": 8, "show_progress_bar": False}
```

### 4. Fallback Mode 동작

**증상**: "기본 모드 (제한된 기능)" 표시

**원인**: FAISS 초기화 실패

**확인 방법**:

```python
# 로그 확인
import logging
logging.basicConfig(level=logging.DEBUG)

# Vector Store 상태 확인
from vector_store_faiss import create_faiss_vector_store
vector_store = create_faiss_vector_store()
```

## 📊 성능 최적화

### 1. 인덱스 크기 최적화

```python
# build_faiss_index.py에서 청크 크기 조정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # 더 작은 청크
    chunk_overlap=100, # 더 적은 오버랩
)
```

### 2. 임베딩 모델 최적화

```python
# CPU 최적화
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'batch_size': 16, 'show_progress_bar': False}
)
```

### 3. Streamlit 캐싱 활용

```python
# 임베딩 모델 캐싱
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(...)

# Vector Store 캐싱
@st.cache_resource
def load_vector_store():
    return create_faiss_vector_store()
```

## 🔄 업데이트 프로세스

### 표준 데이터 업데이트

1. **로컬에서 데이터 수정**
2. **FAISS 인덱스 재빌드**
   ```bash
   python build_faiss_index.py
   ```
3. **Git 커밋 및 배포**
   ```bash
   git add standard/
   git commit -m "Update standard data and rebuild FAISS index"
   git push
   ```

### Vector Store 코드 업데이트

1. **vector_store_faiss.py 수정**
2. **로컬 테스트**
   ```bash
   python vector_store_faiss.py
   ```
3. **인덱스 재빌드 (필요시)**
4. **배포**

## 📈 모니터링 및 로깅

### 로그 레벨 설정

```python
# 배포 환경
logging.basicConfig(level=logging.WARNING)

# 개발 환경
logging.basicConfig(level=logging.DEBUG)
```

### 성능 메트릭

- **인덱스 크기**: ~7MB (FAISS)
- **로딩 시간**: ~3-5초
- **검색 속도**: ~100ms
- **메모리 사용량**: ~200MB

## 🎯 베스트 프랙티스

1. **배포 전 체크리스트**

   - [ ] FAISS 인덱스 빌드 완료
   - [ ] requirements.txt 업데이트
   - [ ] 환경변수 설정
   - [ ] 로컬 테스트 완료

2. **모니터링**

   - Vector Store 상태 정기 확인
   - 로그 모니터링
   - 성능 메트릭 추적

3. **백업 전략**
   - 표준 데이터 정기 백업
   - FAISS 인덱스 버전 관리
   - 설정 파일 백업

## 🆘 지원

문제가 발생하면 다음을 확인하세요:

1. **로그 확인**: Streamlit Cloud 로그 또는 터미널 출력
2. **Vector Store 상태**: UI 사이드바에서 확인
3. **파일 존재 여부**: `standard/faiss_index/`, `standard/metadata.json`
4. **의존성 설치**: requirements.txt

---

**📝 참고**: 이 시스템은 자동 fallback을 지원하므로, FAISS가 실패해도 기본 기능은 계속 작동합니다.
