"""
CLIP 건설 도메인 Fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import clip
import pandas as pd
from PIL import Image
from pathlib import Path
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class BuildingDamageDataset(Dataset):
    """건물 피해 이미지-텍스트 데이터셋"""

    def __init__(
        self,
        images_dir="train/datasets/learning_data/learning_pictures",
        labels_file="train/datasets/learning_data/learning_texts.xlsx",
        clip_model=None,
    ):
        self.images_dir = Path(images_dir)
        self.clip_model = clip_model

        # 라벨 데이터 로드
        self.df = pd.read_excel(labels_file)

        # 이미지 파일 목록
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
        self.image_files = [
            f for f in self.images_dir.glob("*") if f.suffix.lower() in valid_extensions
        ]

        # 피해 유형 매핑
        self.damage_descriptions = self._create_damage_descriptions()

    def _create_damage_descriptions(self) -> List[str]:
        """피해 설명 텍스트 생성"""
        descriptions = []

        # learning_texts.xlsx의 피해현황, 피해부위 컬럼 활용
        for _, row in self.df.iterrows():
            try:
                part = str(row.get("피해 부위", "건물"))
                status = str(row.get("피해현황", "손상"))
                description = f"{part}에 {status} 피해"
                descriptions.append(description)
            except:
                descriptions.append("건물 피해")

        # 기본 피해 설명들도 추가
        base_descriptions = [
            "건물 외벽 균열 피해",
            "지붕 누수 피해",
            "창문 파손 피해",
            "콘크리트 박리 피해",
            "철근 노출 피해",
            "화재 손상 피해",
            "구조적 변형 피해",
            "정상적인 건물",
        ]
        descriptions.extend(base_descriptions)

        return list(set(descriptions))  # 중복 제거

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]

        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_file)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # CLIP 전처리
            if self.clip_model:
                image_input = self.clip_model[1](image).unsqueeze(0)
            else:
                image_input = image

            # 랜덤하게 피해 설명 선택 (실제로는 이미지와 매칭되어야 함)
            description_idx = idx % len(self.damage_descriptions)
            description = self.damage_descriptions[description_idx]

            return image_input.squeeze(0), description

        except Exception as e:
            logger.error(f"데이터 로드 오류 {image_file}: {e}")
            # 폴백: 기본 데이터 반환
            default_image = torch.zeros(3, 224, 224)
            return default_image, "건물 피해"


class CLIPFineTuner:
    """CLIP 모델 Fine-tuning"""

    def __init__(self, model_name="ViT-B/32", device="auto"):
        self.device = (
            "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
        )

        # CLIP 모델 로드
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        # 훈련 가능하도록 설정
        for param in self.model.parameters():
            param.requires_grad = True

        logger.info(f"CLIP 모델 로드 완료: {model_name}, Device: {self.device}")

    def create_data_loader(self, batch_size=16, num_workers=2):
        """데이터 로더 생성"""
        dataset = BuildingDamageDataset(clip_model=(self.model, self.preprocess))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

        return dataloader

    def _collate_fn(self, batch):
        """배치 데이터 처리"""
        images, texts = zip(*batch)

        # 이미지 스택
        images = torch.stack(images)

        # 텍스트 토큰화
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)

        return images.to(self.device), text_tokens

    def train(self, epochs=10, learning_rate=1e-5, batch_size=8):
        """CLIP Fine-tuning 훈련"""
        logger.info("CLIP Fine-tuning 시작...")

        # 데이터 로더
        dataloader = self.create_data_loader(batch_size=batch_size)

        # 옵티마이저 (작은 learning rate 사용)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 손실 함수 (CLIP 원본과 동일)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, (images, text_tokens) in enumerate(dataloader):
                try:
                    # Forward pass
                    logits_per_image, logits_per_text = self.model(images, text_tokens)

                    # CLIP 손실 계산
                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=self.device)

                    loss_img = loss_fn(logits_per_image, labels)
                    loss_text = loss_fn(logits_per_text, labels)
                    loss = (loss_img + loss_text) / 2

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                        )

                except Exception as e:
                    logger.error(f"배치 처리 오류: {e}")
                    continue

            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1} 완료, Average Loss: {avg_loss:.4f}")

        logger.info("CLIP Fine-tuning 완료!")

    def save_model(self, save_path="train/models/clip_finetuned.pt"):
        """Fine-tuned 모델 저장"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # 모델 상태만 저장 (CLIP 구조는 유지)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": "ViT-B/32",
            },
            save_path,
        )

        logger.info(f"Fine-tuned CLIP 모델 저장: {save_path}")

    def load_finetuned_model(self, load_path="train/models/clip_finetuned.pt"):
        """Fine-tuned 모델 로드"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Fine-tuned CLIP 모델 로드: {load_path}")


def train_clip_finetuning():
    """CLIP Fine-tuning 실행"""

    # Fine-tuner 초기화
    fine_tuner = CLIPFineTuner()

    # 훈련 실행 (작은 epochs로 테스트)
    fine_tuner.train(epochs=5, learning_rate=1e-6, batch_size=4)

    # 모델 저장
    fine_tuner.save_model()

    return fine_tuner


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("🚀 CLIP 건설 도메인 Fine-tuning 시작...")
    fine_tuner = train_clip_finetuning()
    print("✅ Fine-tuning 완료!")
