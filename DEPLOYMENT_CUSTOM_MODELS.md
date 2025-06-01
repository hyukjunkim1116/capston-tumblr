# 🚀 배포 환경에서 커스텀 모델 사용 가이드

## 📋 개요

**중요 변경사항**: 이제 모든 환경(로컬/배포)에서 **커스텀 YOLOv8 모델이 필수**입니다.

- 기본 모델 사용 불가
- 배포 환경과 로컬 환경 완전 동일화
- 커스텀 모델 없으면 시스템 중단

## 🎯 지원하는 방법들

### 1. **GitHub Releases (권장)**

- ✅ 가장 안정적이고 신뢰할 수 있는 방법
- ✅ 버전 관리 가능
- ✅ 대용량 파일 지원 (최대 2GB)

### 2. **Google Drive**

- ✅ 간단한 업로드
- ⚠️ 공유 링크 설정 필요
- ⚠️ 다운로드 제한 있을 수 있음

### 3. **Hugging Face Hub**

- ✅ AI 모델에 특화된 플랫폼
- ✅ 무료 호스팅
- ✅ 모델 카드 및 설명 지원

## 🔧 설정 방법

### 방법 1: GitHub Releases (권장)

#### 1단계: GitHub 저장소에 Release 생성

```bash
# 1. 현재 커스텀 모델 확인
ls -la train/models/
# custom_yolo_damage.pt (6.0MB)

# 2. GitHub에 커밋 (코드만, 모델 파일 제외)
git add .
git commit -m "Add custom model deployment support"
git push origin main

# 3. GitHub에서 새 Release 생성
# - GitHub 저장소 → Releases → Create a new release
# - Tag version: v1.0.0
# - Release title: "Custom Models v1.0.0"
# - 파일 첨부: custom_yolo_damage.pt 업로드
```

#### 2단계: 환경변수 설정

**Streamlit Cloud:**

```
CUSTOM_YOLO_URL = https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/custom_yolo_damage.pt
```

**Heroku:**

```bash
heroku config:set CUSTOM_YOLO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/custom_yolo_damage.pt"
```

**Railway:**

```bash
# railway.toml 또는 웹 대시보드에서
CUSTOM_YOLO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/custom_yolo_damage.pt"
```

### 방법 2: Google Drive

#### 1단계: Google Drive에 모델 업로드

```bash
# 1. Google Drive에 파일 업로드
# - drive.google.com 접속
# - custom_yolo_damage.pt 파일 업로드
# - 파일 우클릭 → 공유 → 링크 복사

# 2. 공유 링크를 다운로드 직링크로 변환
# 원본: https://drive.google.com/file/d/1ABC123DEF456GHI/view?usp=sharing
# 변환: https://drive.google.com/uc?id=1ABC123DEF456GHI
```

#### 2단계: 환경변수 설정

```
CUSTOM_YOLO_URL = https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID
```

### 방법 3: Hugging Face Hub

#### 1단계: Hugging Face에 모델 업로드

```bash
# 1. Hugging Face CLI 설치 및 로그인
pip install huggingface_hub
huggingface-cli login

# 2. 새 모델 저장소 생성
# https://huggingface.co/new 에서 새 모델 생성
# 이름: building-damage-yolo

# 3. 모델 파일 업로드
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="train/models/custom_yolo_damage.pt",
    path_in_repo="custom_yolo_damage.pt",
    repo_id="YOUR_USERNAME/building-damage-yolo",
    repo_type="model"
)
```

#### 2단계: 환경변수 설정

```
CUSTOM_YOLO_URL = https://huggingface.co/YOUR_USERNAME/building-damage-yolo/resolve/main/custom_yolo_damage.pt
```

## 🔄 실제 사용 예시

### 환경변수 설정 예시

#### Streamlit Cloud

```
# Streamlit Cloud → Manage app → Settings → Secrets
OPENAI_API_KEY = "your_openai_api_key"
CUSTOM_YOLO_URL = "https://github.com/your-username/tumblr/releases/download/v1.0.0/custom_yolo_damage.pt"
CUSTOM_CLIP_URL = "https://huggingface.co/your-username/building-damage-clip/resolve/main/clip_finetuned.pt"
```

#### .env 파일 (로컬 테스트용)

```bash
# .env
OPENAI_API_KEY=your_openai_api_key
CUSTOM_YOLO_URL=https://github.com/your-username/tumblr/releases/download/v1.0.0/custom_yolo_damage.pt
CUSTOM_CLIP_URL=https://huggingface.co/your-username/building-damage-clip/resolve/main/clip_finetuned.pt
```

## 📊 모델 로딩 순서

시스템은 다음 순서로 모델을 찾습니다:

### YOLO 모델

1. **로컬 커스텀 모델**: `train/models/custom_yolo_damage.pt`
2. **캐시된 커스텀 모델**: `cache/models/custom_yolo_damage.pt`
3. **환경변수 URL**: `CUSTOM_YOLO_URL`에서 다운로드
4. **기본 모델**: YOLOv8n (자동 다운로드)

### CLIP 모델

1. **로컬 커스텀 모델**: `train/models/clip_finetuned.pt`
2. **캐시된 커스텀 모델**: `cache/models/clip_finetuned.pt`
3. **환경변수 URL**: `CUSTOM_CLIP_URL`에서 다운로드
4. **기본 모델**: ViT-B/32

## 🚨 주의사항

### 파일 크기 제한

- **GitHub Releases**: 최대 2GB
- **Google Drive**: 무제한 (하지만 다운로드 제한)
- **Hugging Face**: 무제한 (하지만 50GB 이상은 Git LFS 필요)

### 보안

- **공개 저장소**: 모델이 공개됨
- **비공개 저장소**: 인증 토큰 필요할 수 있음
- **민감한 모델**: 비공개 저장소 사용 권장

### 성능

- **첫 실행**: 모델 다운로드로 시간 소요 (1-3분)
- **이후 실행**: 캐시된 모델 사용으로 빠름
- **메모리**: 배포 환경 메모리 제한 고려

## 🔍 디버깅

### 로그 확인

```python
# 모델 로딩 과정 로그 확인
# Streamlit Cloud → Manage app → Logs에서 확인 가능

# 성공 로그 예시:
# ✅ YOLOv8 다운로드된 커스텀 모델 로드 완료
# ✅ CLIP 다운로드된 커스텀 모델 로드 완료

# 실패 로그 예시:
# ⚠️ 커스텀 모델 다운로드 실패, 기본 모델 사용
# 📝 CUSTOM_YOLO_URL 환경변수 없음, 기본 모델 사용
```

### 문제 해결

```bash
# 1. URL 접속 확인
curl -I "https://github.com/your-username/your-repo/releases/download/v1.0.0/custom_yolo_damage.pt"

# 2. 파일 크기 확인
curl -sI "URL" | grep -i content-length

# 3. 로컬 테스트
CUSTOM_YOLO_URL="your_url" streamlit run streamlit_app.py
```

## 📈 성능 비교

| 환경           | 모델            | 로딩 시간 | 추론 속도 | 정확도 |
| -------------- | --------------- | --------- | --------- | ------ |
| 로컬           | 커스텀          | ~1초      | 빠름      | 높음   |
| 배포 (첫 실행) | 다운로드+커스텀 | ~60초     | 보통      | 높음   |
| 배포 (이후)    | 캐시된 커스텀   | ~5초      | 보통      | 높음   |
| 배포 (기본)    | YOLOv8n         | ~10초     | 보통      | 중간   |

## ✅ 체크리스트

### 배포 전

- [ ] 커스텀 모델이 정상 동작하는지 로컬 테스트
- [ ] 모델 파일을 외부 저장소에 업로드
- [ ] 다운로드 URL이 정상 작동하는지 확인
- [ ] 환경변수 설정

### 배포 후

- [ ] 첫 실행 시 모델 다운로드 로그 확인
- [ ] 커스텀 모델 로딩 성공 로그 확인
- [ ] 분석 결과가 로컬과 유사한지 확인

---

**💡 팁**: GitHub Releases를 사용하는 것이 가장 안정적이고 관리하기 쉽습니다!
