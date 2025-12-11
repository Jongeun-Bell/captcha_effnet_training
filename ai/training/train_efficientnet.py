"""
EfficientNet-B0 학습 코드 (CAPTCHA 데이터용)
- 2단계 폴더(images/group/classname) 구조 자동 지원
- 클래스 개수(NUM_CLASSES) 자동 계산
- MLflow 연동 (file:./mlruns 고정) - 추후 서버 ip 주소로 변경 필수
- argparse는 main() 내부에서만 실행되도록 펙토링
"""

import os
import time
import argparse
from datetime import datetime

import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

from captcha_dataset import CAPTCHADataset


# ============================================================
# 학습 함수
# ============================================================
def main():
    # ------------------------
    # argparse 설정
    # ------------------------
    parser = argparse.ArgumentParser(description="EfficientNet CAPTCHA Training Script")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="학습 데이터(images) 경로 (예: ./images)")
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="모델 저장 폴더")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    patience = args.patience

    # ------------------------
    # 출력 폴더 생성
    # ------------------------
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "best_model.pth")

    # ------------------------
    # MLflow 설정
    # ------------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("captcha-effnet-v2")

    run_name = f"effnet_lr{learning_rate}_bs{batch_size}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # ------------------------
    # 전처리 정의
    # ------------------------
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ============================================================
    # 데이터 로드
    # ============================================================
    print("=" * 60)
    print("데이터셋 로드 중...")
    print("=" * 60)

    full_dataset = CAPTCHADataset(data_dir, transform=None)

    # train/val 분리
    train_ratio = 0.8
    val_ratio = 0.2

    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    print(f"총 데이터: {total_size}개")
    print(f"학습 데이터: {train_size}개")
    print(f"검증 데이터: {val_size}개\n")

    # ============================================================
    # 모델 설정
    # ============================================================
    print("=" * 60)
    print("모델 설정 중...")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    print(f"사용 장치: {device}\n")

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # ------------------------
    # 클래스 개수 자동 계산
    # ------------------------
    NUM_CLASSES = len(set(full_dataset.labels))
    print(f"감지된 실제 클래스 수: {NUM_CLASSES}")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    model.to(device)

    print(f"✅ EfficientNet-B0 로드 완료 ({NUM_CLASSES}개 클래스)\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    # ============================================================
    # MLflow 기록 + 학습 시작
    # ============================================================
    print("=" * 60)
    print("학습 시작")
    print("=" * 60 + "\n")

    start_total = time.time()
    best_accuracy = 0
    patience_counter = 0

    # signature 생성용 입력 예시
    example_input = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        example_output = model(example_input)

    signature = infer_signature(
        example_input.cpu().numpy(),
        example_output.cpu().numpy()
    )

    # ----------------------------
    # MLflow RUN
    # ----------------------------
    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("run_name", run_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("patience", patience)
        mlflow.log_param("train_samples", train_size)
        mlflow.log_param("val_samples", val_size)
        mlflow.log_param("num_classes", NUM_CLASSES)

        # ============================================================
        # EPOCH LOOP
        # ============================================================
        for epoch in range(num_epochs):
            epoch_start = time.time()

            # ------------------------
            # Train
            # ------------------------
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total

            # ------------------------
            # Validation
            # ------------------------
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)

                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total

            scheduler.step()
            epoch_time = time.time() - epoch_start

            # ------------------------
            # Epoch 결과 출력
            # ------------------------
            print(f"[Epoch {epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2%}")
            print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2%}")
            print(f"  → Epoch Time: {epoch_time:.2f} sec")

            gap = train_accuracy - val_accuracy
            if gap >= 0.10:
                print(f"  ⚠️ 과적합 경고! (차이: {gap:.2%})")
            else:
                print(f"  ✓ 정상 (차이: {gap:.2%})")

            # mlflow 기록
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_accuracy, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

            # ------------------------
            # Early Stopping
            # ------------------------
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0

                torch.save(model.state_dict(), model_save_path)
                print(f"  ✅ 최고 모델 저장! (Val_Acc: {val_accuracy:.2%})")

                mlflow.pytorch.log_model(
                    model,
                    name="best_model",
                    signature=signature
                )
            else:
                patience_counter += 1
                print(f"  ⏳ Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("\n🛑 Early Stopping 발동!")
                    print(f"최고 검증 정확도: {best_accuracy:.2%}")
                    break

            print()

        # best accuracy 기록
        mlflow.log_metric("best_val_acc", best_accuracy)

    # ============================================================
    # 완료 출력
    # ============================================================
    end_total = time.time()
    print("=" * 60)
    print("학습 완료!")
    print(f"최고 검증 정확도: {best_accuracy:.2%}")
    print(f"전체 학습 시간: {end_total - start_total:.2f} sec")
    print(f"모델 저장 경로: {model_save_path}")
    print("=" * 60)


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    main()
