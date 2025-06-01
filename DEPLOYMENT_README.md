# 🚀 배포 가이드 - 최신 오류 해결

## 📋 최신 해결 내용 (2025-06-01)

### ✅ **새로 해결된 배포 오류**

- **YOLOv8 import 오류**: 패키지 순서 및 의존성 최적화
- **LangChain LLMChain deprecated**: 최신 invoke 패턴으로 업데이트
- **Streamlit config 무효 옵션**: 잘못된 runner 설정 제거
- **asyncio 오류**: 헬스체크 및 비동기 처리 개선

### ✅ **기존 해결된 배포 오류**

- **OpenCV libGL.so.1 오류**: `cv2` import 제거로 해결
- **LangChain deprecation 경고**: 최신 패키지로 업데이트
- **CLIP 모델 로딩 오류**: 안정적인 fallback 처리

## 🔧 배포 전 체크리스트

### 1. **환경변수 설정**

```bash
# env.template를 .env로 복사
cp env.template .env

# .env 파일에서 실제 값 설정
OPENAI_API_KEY=your_actual_api_key
ENVIRONMENT=production
DEVICE=cpu
```

### 2. **requirements.txt 확인**

```txt
# Streamlit Cloud 최적화된 패키지 순서
streamlit>=1.28.0
openai>=1.0.0
torch>=2.0.0
ultralytics>=8.0.0
opencv-python-headless>=4.8.0  # ✅ GUI 없는 환경 호환
langchain-openai>=0.1.0         # ✅ 최신 패키지
```

### 3. **Streamlit 설정 최적화**

```toml
# .streamlit/config.toml (유효한 옵션만)
[global]
developmentMode = false

[server]
headless = true
maxUploadSize = 200

[client]
showErrorDetails = false

[runner]
magicEnabled = false  # ✅ 유효한 옵션만 사용
```

## 🌐 플랫폼별 배포 방법

### **Streamlit Cloud**

1. GitHub 연결
2. 환경변수 설정:
   ```
   OPENAI_API_KEY = your_key
   ENVIRONMENT = production
   DEVICE = cpu
   TOKENIZERS_PARALLELISM = false
   TRANSFORMERS_VERBOSITY = error
   ```
3. 자동 배포 완료

### **Heroku**

```bash
# Procfile 생성
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# 환경변수 설정
heroku config:set OPENAI_API_KEY=your_key
heroku config:set ENVIRONMENT=production
heroku config:set TOKENIZERS_PARALLELISM=false
```

### **Railway**

```bash
# railway.toml 생성
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run streamlit_app.py"

[env]
TOKENIZERS_PARALLELISM = "false"
TRANSFORMERS_VERBOSITY = "error"
```

### **Docker**

```dockerfile
FROM python:3.11-slim

# 환경변수 설정
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_VERBOSITY=error
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
```

## 🐛 최신 오류 해결

### **1. YOLOv8 import 오류**

```python
# ❌ 문제: 패키지 로딩 순서
WARNING:root:YOLOv8 not available. Install with: pip install ultralytics

# ✅ 해결: 패키지 순서 최적화 + 상세 로깅
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLOv8 패키지 로드 성공")
except ImportError as e:
    YOLO_AVAILABLE = False
    logger.warning(f"YOLOv8 not available: {e}")
except Exception as e:
    YOLO_AVAILABLE = False
    logger.warning(f"YOLOv8 로딩 오류: {e}")
```

### **2. LangChain LLMChain deprecated**

```python
# ❌ 이전 (deprecated)
from langchain.chains import LLMChain
self.llm_chain = LLMChain(llm=llm, prompt=prompt)
response = self.llm_chain.run(data)

# ✅ 수정 (최신 패턴)
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
formatted_prompt = self.prompt.format(data)
response = self.llm.invoke(formatted_prompt)
```

### **3. Streamlit Config 무효 옵션**

```toml
# ❌ 무효한 옵션들
[runner]
installTracer = false  # 제거됨
fixMatplotlib = false  # 제거됨

# ✅ 유효한 옵션만
[runner]
magicEnabled = false  # ✅ 유효
```

### **4. asyncio 오류**

```python
# 환경변수로 비동기 처리 최적화
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### **5. 헬스체크 실패**

```bash
# ❌ 오류
dial tcp 127.0.0.1:8501: connect: connection refused

# ✅ 해결: 서버 주소 설정
streamlit run streamlit_app.py --server.address=0.0.0.0
```

## 📊 성능 모니터링

### **배포 로그 확인**

```python
# 상세 로깅으로 문제 추적
logger.info("YOLOv8 패키지 로드 성공")
logger.warning("Fine-tuned 모델 없음, 기본 CLIP 사용")
logger.info("기본 CLIP 모델 로드 완료")
```

### **메모리 최적화**

```python
# CPU 전용 PyTorch 설정
torch>=2.0.0
# 메모리 할당 제한
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### **응답 시간**

- **초기 로딩**: ~60초 (모델 다운로드)
- **이후 분석**: ~30초
- **보고서 생성**: ~10초

## 🚀 배포 후 확인사항

### **1. 로그 체크**

- [ ] YOLOv8 패키지 로드 성공
- [ ] CLIP 모델 로드 완료
- [ ] GPT-4 보고서 생성기 초기화 완료
- [ ] LangChain deprecation 경고 없음

### **2. 기능 테스트**

- [ ] 이미지 업로드 정상 작동
- [ ] AI 분석 결과 출력
- [ ] PDF 보고서 생성
- [ ] 오류 없이 완료

### **3. 성능 확인**

- [ ] 응답 시간 60초 이내
- [ ] 메모리 사용량 안정
- [ ] 헬스체크 통과

## 🔧 유지보수

### **정기 업데이트**

```bash
# 패키지 업데이트 (주의: 호환성 확인)
pip install -U streamlit ultralytics langchain-openai

# 배포 전 로컬 테스트
streamlit run streamlit_app.py
```

### **모니터링 체크리스트**

- 일일 에러 로그 확인
- 주간 성능 지표 검토
- 월간 패키지 보안 업데이트

## 📞 지원

배포 중 문제 발생시:

1. **로그 확인**: Streamlit Cloud > Manage app > Logs
2. **환경변수**: Settings > Secrets 재확인
3. **의존성**: requirements.txt 패키지 순서 확인
4. **재배포**: GitHub 커밋 후 자동 재배포

---

**✅ 2025-06-01 최신 오류까지 모두 해결되었습니다!**
