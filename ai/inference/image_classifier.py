"""
Inference Script (EfficientNet-B0)
- 학습된 모델(best_model.pth)로 단일 이미지 또는 랜덤 이미지 분류
- CAPTCHA Dataset의 폴더 구조 기반 자동 라벨 매핑
"""

import os
import sys
import argparse
import random
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================
# 0) training 폴더 import 경로 추가
# ============================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # /ai/inference
TRAINING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "training"))

sys.path.append(TRAINING_DIR)

from captcha_dataset import CAPTCHADataset  # ← 이제 정상 import 됨


# ============================================
# 1) argparse
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet Inference Script")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="이미지 폴더 루트 경로 (ex: ../images)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="학습된 best_model.pth 경로")
    parser.add_argument("--image_path", type=str, default=None,
                        help="분류할 이미지 경로 (없으면 랜덤 선택)")

    return parser.parse_args()


# ============================================
# 2) 이미지 분류
# ============================================
def classify(args):
    # 장치 설정
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"사용 장치: {device}")

    # ===== Dataset 불러서 라벨 매핑 자동 생성 =====
    dataset = CAPTCHADataset(args.data_dir, transform=None)

    group_to_label = dataset.group_to_label
    label_to_group = {v: k for k, v in group_to_label.items()}

    NUM_CLASSES = len(label_to_group)
    print(f"자동 감지된 클래스 수: {NUM_CLASSES}")
    print("label_to_group:", label_to_group)

    # ===== 모델 생성 =====
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    # ===== 모델 로드 =====
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"모델 파일 없음: {args.model_path}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # ===== 전처리 =====
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    # ============================================
    # 3) 이미지 선택 (랜덤 or 지정)
    # ============================================
    if args.image_path is None:
        print("\n📌 image_path가 없어서 랜덤 이미지 선택합니다.")

        all_images = []

        # 2단계 폴더(images/group/class) 순회하며 이미지 수집
        for group in os.listdir(args.data_dir):
            group_path = os.path.join(args.data_dir, group)
            if not os.path.isdir(group_path):
                continue

            for cls in os.listdir(group_path):
                cls_path = os.path.join(group_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                for f in os.listdir(cls_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_images.append(os.path.join(cls_path, f))

        if len(all_images) == 0:
            raise RuntimeError("data_dir 안에 이미지가 없습니다.")

        img_path = random.choice(all_images)
        print(f"🎯 선택된 랜덤 이미지: {img_path}")

    else:
        img_path = args.image_path
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"이미지가 존재하지 않음: {img_path}")

        print(f"\n🎯 지정된 이미지: {img_path}")

    # ============================================
    # 4) 추론
    # ============================================
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = probs.argmax()

    predicted_group = label_to_group[pred_idx]
    confidence = probs[pred_idx]

    # ============================================
    # 5) 출력
    # ============================================
    print("\n===== 예측 결과 =====")
    print(f"예측 클래스: {predicted_group}")
    print(f"신뢰도: {confidence*100:.2f}%")
    print("\n확률 분포:")
    for i in range(NUM_CLASSES):
        print(f"  {label_to_group[i]} : {probs[i]*100:.2f}%")

    return {
        "predicted_label": predicted_group,
        "confidence": float(confidence),
        "probabilities": {label_to_group[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    }


# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    args = parse_args()
    classify(args)
