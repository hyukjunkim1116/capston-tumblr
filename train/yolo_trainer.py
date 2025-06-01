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
        """이미지와 라벨 처리 - 실제 라벨 데이터 활용"""
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

                # 실제 라벨 데이터 기반 라벨 생성
                label_file = (
                    self.output_dir / split_dir / "labels" / f"{image_file.stem}.txt"
                )
                self._create_accurate_label(label_file, image_file, df)

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

    def _create_accurate_label(self, label_file, image_file, df):
        """실제 라벨 데이터 기반 정확한 라벨 생성"""
        try:
            # 이미지 번호 추출 (예: 301.png -> 301)
            image_num = image_file.stem

            # 해당 이미지에 대한 라벨 데이터 찾기
            matching_rows = df[df["순번"].astype(str).str.contains(image_num, na=False)]

            # 이미지 크기 정보 가져오기
            with Image.open(image_file) as img:
                img_width, img_height = img.size

            labels = []

            if not matching_rows.empty:
                for _, row in matching_rows.iterrows():
                    # 피해 부위와 피해현황 정보로 클래스 결정
                    damage_part = str(row.get("피해 부위", "")).lower()
                    damage_status = str(row.get("피해현황", "")).lower()

                    # 피해 유형 매핑
                    class_id = self._map_damage_to_class(damage_part, damage_status)

                    # 피해 부위에 따른 바운딩 박스 위치 추정
                    bbox = self._estimate_bbox_from_damage_part(
                        damage_part, img_width, img_height
                    )

                    # YOLO 형식으로 변환 (center_x, center_y, width, height - 정규화)
                    center_x = (bbox[0] + bbox[2]) / 2 / img_width
                    center_y = (bbox[1] + bbox[3]) / 2 / img_height
                    width = (bbox[2] - bbox[0]) / img_width
                    height = (bbox[3] - bbox[1]) / img_height

                    labels.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    )

            # 라벨이 없으면 기본 라벨 생성
            if not labels:
                labels.append("0 0.5 0.5 1.0 1.0")  # 전체 이미지를 균열로 가정

            # 라벨 파일 저장
            with open(label_file, "w") as f:
                f.write("\n".join(labels) + "\n")

        except Exception as e:
            logger.error(f"정확한 라벨 생성 실패 {label_file}: {e}")
            # 폴백: 기본 라벨 생성
            with open(label_file, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

    def _map_damage_to_class(self, damage_part, damage_status):
        """피해 부위와 현황을 클래스 ID로 매핑"""
        # 피해 유형 키워드 매핑
        mapping_rules = {
            0: ["균열", "크랙", "갈라짐", "틈"],  # crack
            1: ["수해", "침수", "누수", "물", "습기"],  # water_damage
            2: ["화재", "불", "연소", "탄화"],  # fire_damage
            3: ["지붕", "옥상", "루프", "처마"],  # roof_damage
            4: ["창문", "유리", "윈도우", "창호"],  # window_damage
            5: ["문", "도어", "출입구", "현관"],  # door_damage
            6: ["기초", "파운데이션", "토대", "밑바닥"],  # foundation_damage
            7: ["구조", "변형", "틀어짐", "처짐", "기울어짐"],  # structural_deformation
            8: ["외벽", "파사드", "외관", "벽면"],  # facade_damage
        }

        # 피해 부위와 현황을 결합한 텍스트에서 키워드 검색
        combined_text = f"{damage_part} {damage_status}".lower()

        for class_id, keywords in mapping_rules.items():
            if any(keyword in combined_text for keyword in keywords):
                return class_id

        return 0  # 기본값: 균열

    def _estimate_bbox_from_damage_part(self, damage_part, img_width, img_height):
        """피해 부위에 따른 바운딩 박스 위치 추정"""
        damage_part = damage_part.lower()

        # 부위별 대략적인 위치 매핑 (x1, y1, x2, y2)
        position_mapping = {
            "지붕": (0.1, 0.0, 0.9, 0.3),  # 상단
            "옥상": (0.1, 0.0, 0.9, 0.3),
            "처마": (0.0, 0.0, 1.0, 0.4),
            "외벽": (0.0, 0.2, 1.0, 0.8),  # 중앙
            "벽면": (0.0, 0.2, 1.0, 0.8),
            "파사드": (0.0, 0.2, 1.0, 0.8),
            "창문": (0.2, 0.3, 0.8, 0.7),  # 중앙 작은 영역
            "창호": (0.2, 0.3, 0.8, 0.7),
            "유리": (0.2, 0.3, 0.8, 0.7),
            "문": (0.3, 0.4, 0.7, 0.9),  # 하단 중앙
            "도어": (0.3, 0.4, 0.7, 0.9),
            "출입구": (0.3, 0.4, 0.7, 0.9),
            "기초": (0.0, 0.7, 1.0, 1.0),  # 하단
            "파운데이션": (0.0, 0.7, 1.0, 1.0),
            "토대": (0.0, 0.7, 1.0, 1.0),
        }

        # 매칭되는 부위 찾기
        for part, (x1_ratio, y1_ratio, x2_ratio, y2_ratio) in position_mapping.items():
            if part in damage_part:
                x1 = int(x1_ratio * img_width)
                y1 = int(y1_ratio * img_height)
                x2 = int(x2_ratio * img_width)
                y2 = int(y2_ratio * img_height)
                return (x1, y1, x2, y2)

        # 기본값: 중앙 영역 (50% 크기)
        margin_x = int(img_width * 0.25)
        margin_y = int(img_height * 0.25)
        return (margin_x, margin_y, img_width - margin_x, img_height - margin_y)

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
