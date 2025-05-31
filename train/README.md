# Train Directory Structure

이 폴더는 건물 손상 분석 시스템의 모든 훈련 관련 컴포넌트들을 포함합니다.

## 📁 Directory Structure

```
train/
├── scripts/           # 훈련 스크립트들
│   └── train_models.py
├── models/           # 훈련된 모델들
│   ├── custom_yolo_damage.pt
│   └── clip_finetuned.pt
├── datasets/         # 훈련 데이터셋들
│   ├── yolo_dataset/
│   │   ├── train/
│   │   ├── val/
│   │   └── dataset.yaml
│   └── learning_data/
│       ├── learning_pictures/
│       └── learning_texts.xlsx
├── runs/            # 훈련 실행 결과들
│   └── detect/
├── logs/            # 훈련 로그들
│   └── training.log
└── configs/         # 설정 파일들
    └── yolov8n.pt   # 기본 YOLO 모델
```

## 🚀 Usage

### 모델 훈련

```bash
cd train/scripts
python train_models.py
```

### 데이터셋 구조

- `datasets/yolo_dataset/`: YOLO 객체 탐지용 데이터셋
- `datasets/learning_data/`: 일반 학습 데이터 및 텍스트

### 모델 파일

- `models/custom_yolo_damage.pt`: 손상 탐지용 커스텀 YOLO 모델
- `models/clip_finetuned.pt`: 파인튜닝된 CLIP 모델

## 📊 Training Results

훈련 결과는 `runs/` 폴더에 저장되며, 로그는 `logs/` 폴더에서 확인할 수 있습니다.
