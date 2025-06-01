# 🚀 배포 가이드 - OpenCV 오류 해결

## 📋 주요 해결 내용

### ✅ **해결된 배포 오류**

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
# 배포 환경에서 안전한 패키지들
opencv-python-headless>=4.8.0  # ✅ GUI 없는 환경 호환
langchain-openai>=0.1.0         # ✅ 최신 패키지
torch>=2.0.0                    # ✅ CPU 모드 지원
```

### 3. **Streamlit 설정 최적화**

```toml
# .streamlit/config.toml
[global]
developmentMode = false

[server]
headless = true
maxUploadSize = 200

[runner]
magicEnabled = false
installTracer = false
fixMatplotlib = false
```

## 🌐 플랫폼별 배포 방법

### **Streamlit Cloud**

1. GitHub 연결
2. 환경변수 설정:
   ```
   OPENAI_API_KEY = your_key
   ENVIRONMENT = production
   DEVICE = cpu
   ```
3. 자동 배포 완료

### **Heroku**

```bash
# Procfile 생성
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# 환경변수 설정
heroku config:set OPENAI_API_KEY=your_key
heroku config:set ENVIRONMENT=production
```

### **Railway**

```bash
# railway.toml 생성
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "streamlit run streamlit_app.py"
```

### **Docker**

```dockerfile
FROM python:3.11-slim

# 시스템 패키지 설치 (OpenCV 의존성 제거됨)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🐛 오류 해결

### **1. OpenCV libGL.so.1 오류**

```bash
# ❌ 오류 원인
import cv2  # GUI 라이브러리 필요

# ✅ 해결 방법
# cv2 import 제거 (코드에서 실제로 사용하지 않음)
# opencv-python-headless 사용
```

### **2. LangChain 경고**

```python
# ❌ 이전 (deprecated)
from langchain.llms import OpenAI

# ✅ 수정 (최신)
from langchain_openai import OpenAI
```

### **3. 메모리 부족**

```bash
# 환경변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
```

### **4. 모델 파일 없음**

```python
# 자동 fallback 처리됨
if not Path(model_path).exists():
    logger.warning("모델 파일 없음, 기본 모델 사용")
    model = YOLO("yolov8n.pt")  # 자동 다운로드
```

## 📊 성능 모니터링

### **로그 확인**

```python
# 배포 환경에서 로그 레벨 조정
logging.basicConfig(level=logging.WARNING)
```

### **메모리 사용량**

```python
# Streamlit Cloud: 1GB 제한
# Heroku: 512MB 제한
# Railway: 8GB 제한
```

### **응답 시간**

- **이미지 분석**: ~30초
- **보고서 생성**: ~10초
- **모델 로딩**: 초기 1회만 ~20초

## 🚀 배포 후 확인사항

### **1. 기능 테스트**

- [ ] 이미지 업로드 정상 작동
- [ ] AI 분석 결과 출력
- [ ] PDF 보고서 생성
- [ ] 오류 없이 완료

### **2. 성능 확인**

- [ ] 응답 시간 30초 이내
- [ ] 메모리 사용량 안정
- [ ] 오류 로그 없음

### **3. UI/UX 확인**

- [ ] 모바일 반응형 디자인
- [ ] 다크 테마 적용
- [ ] 로딩 스피너 표시

## 🔧 유지보수

### **정기 업데이트**

```bash
# 패키지 업데이트
pip install -U streamlit ultralytics openai

# 보안 패치
pip audit --fix
```

### **모니터링**

- 일일 사용량 확인
- 오류 로그 검토
- 성능 지표 추적

## 📞 지원

배포 중 문제 발생시:

1. **로그 확인**: 배포 플랫폼 로그 검토
2. **환경변수**: API 키 및 설정 재확인
3. **의존성**: requirements.txt 패키지 버전 확인

---

**✅ 이제 안정적으로 배포됩니다!**
