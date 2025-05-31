"""
YOLOv8 건물 피해 감지 모델 커스텀 훈련
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class YOLODatasetBuilder:
    """YOLOv8용 데이터셋 빌더"""

    def __init__(
        self,
        source_images_dir="train/datasets/learning_data/learning_pictures",
        source_labels_file="train/datasets/learning_data/learning_texts.xlsx",
        output_dir="train/datasets/yolo_dataset",
    ):
        self.source_images_dir = Path(source_images_dir)
        self.source_labels_file = Path(source_labels_file)
        self.output_dir = Path(output_dir)

        # YOLOv8 피해 클래스 정의
        self.damage_classes = {
            0: "crack",  # 균열
            1: "water_damage",  # 수해
            2: "fire_damage",  # 화재
            3: "roof_damage",  # 지붕
            4: "window_damage",  # 창문
            5: "door_damage",  # 문
            6: "foundation_damage",  # 기초
            7: "structural_deformation",  # 구조변형
            8: "facade_damage",  # 외벽
        }

    def build_dataset(self):
        """YOLO 형식 데이터셋 빌드"""
        logger.info("YOLOv8 데이터셋 빌드 시작...")

        # 디렉토리 생성
        self._create_directories()

        # 라벨 데이터 로드
        df = pd.read_excel(self.source_labels_file)

        # 이미지-라벨 매핑
        self._process_images_and_labels(df)

        # YAML 설정 파일 생성
        self._create_yaml_config()

        logger.info(f"데이터셋 빌드 완료: {self.output_dir}")

    def _create_directories(self):
        """YOLOv8 디렉토리 구조 생성"""
        directories = [
            self.output_dir / "train" / "images",
            self.output_dir / "train" / "labels",
            self.output_dir / "val" / "images",
            self.output_dir / "val" / "labels",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _process_images_and_labels(self, df):
        """이미지와 라벨 처리"""
        train_split = 0.8
        image_files = list(self.source_images_dir.glob("*"))

        # 이미지 파일 필터링 (유효한 확장자만)
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
        image_files = [f for f in image_files if f.suffix.lower() in valid_extensions]

        train_count = int(len(image_files) * train_split)

        for i, image_file in enumerate(image_files):
            try:
                # train/val 분할
                is_train = i < train_count
                split_dir = "train" if is_train else "val"

                # 이미지 복사
                dest_image = (
                    self.output_dir / split_dir / "images" / f"{image_file.stem}.jpg"
                )
                self._copy_and_convert_image(image_file, dest_image)

                # 라벨 생성 (임시로 전체 이미지를 하나의 피해 영역으로)
                label_file = (
                    self.output_dir / split_dir / "labels" / f"{image_file.stem}.txt"
                )
                self._create_dummy_label(label_file, image_file)

            except Exception as e:
                logger.warning(f"이미지 처리 실패 {image_file}: {e}")

    def _copy_and_convert_image(self, source, destination):
        """이미지 복사 및 JPG 변환"""
        try:
            with Image.open(source) as img:
                # RGB 변환 (AVIF, WebP 호환성)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(destination, "JPEG", quality=90)
        except Exception as e:
            logger.error(f"이미지 변환 실패 {source}: {e}")

    def _create_dummy_label(self, label_file, image_file):
        """임시 라벨 생성 (실제 어노테이션 없이)"""
        try:
            # 전체 이미지를 하나의 피해 영역으로 가정
            # YOLO 형식: class_id center_x center_y width height (0~1 정규화)
            with open(label_file, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")  # 클래스 0 (crack), 전체 이미지
        except Exception as e:
            logger.error(f"라벨 생성 실패 {label_file}: {e}")

    def _create_yaml_config(self):
        """YOLOv8 설정 YAML 파일 생성"""
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(self.damage_classes),
            "names": list(self.damage_classes.values()),
        }

        yaml_file = self.output_dir / "dataset.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"YAML 설정 파일 생성: {yaml_file}")


class YOLOCustomTrainer:
    """YOLOv8 커스텀 훈련"""

    def __init__(self, dataset_yaml="train/datasets/yolo_dataset/dataset.yaml"):
        self.dataset_yaml = dataset_yaml
        self.model = None

    def train(self, epochs=50, batch_size=16, img_size=640):
        """모델 훈련"""
        logger.info("YOLOv8 커스텀 훈련 시작...")

        # 사전 훈련된 모델 로드
        self.model = YOLO("train/configs/yolov8n.pt")  # nano 버전으로 빠른 훈련

        # 훈련 실행
        results = self.model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device="cpu",  # CPU 사용으로 수정
            save=True,
            project="train/runs/detect",
            name="building_damage",
            exist_ok=True,
        )

        logger.info("훈련 완료!")
        return results

    def save_model(self, save_path="train/models/custom_yolo_damage.pt"):
        """훈련된 모델 저장"""
        if self.model:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            logger.info(f"모델 저장: {save_path}")


def train_custom_yolo():
    """YOLOv8 커스텀 훈련 실행"""

    # 1단계: 데이터셋 빌드
    dataset_builder = YOLODatasetBuilder()
    dataset_builder.build_dataset()

    # 2단계: 모델 훈련
    trainer = YOLOCustomTrainer()
    results = trainer.train(epochs=30, batch_size=8)  # 적은 epochs로 테스트

    # 3단계: 모델 저장
    trainer.save_model()

    return results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("🚀 YOLOv8 건물 피해 감지 모델 훈련 시작...")
    results = train_custom_yolo()
    print("✅ 훈련 완료!")
