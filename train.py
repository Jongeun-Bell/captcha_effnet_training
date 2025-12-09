"""
개선된 학습 코드 (CAPTCHA 데이터용) + MLflow 연동
- Dataset 한 번만 생성 후 random_split로 나누고 transform 분리 적용
- MLflow: log_model(name=..) 적용, log_artifact 중복 제거
- Model Signature 추가
"""

import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import time
import os

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from datetime import datetime

from captcha_dataset import CAPTCHADataset

# ============================================
# 설정
# ============================================
data_dir = "/Users/bell/Desktop/captcha/images/"
output_dir = "./"
model_save_path = os.path.join(output_dir, "best_model.pth")

# 데이터셋 분할 비율
train_ratio = 0.8
val_ratio = 0.2

# 하이퍼파라미터
batch_size = 64
learning_rate = 0.0003
num_epochs = 20
patience = 3

# MLflow 실험 이름
mlflow.set_experiment("captcha-effnet")

# MLflow Run 이름 (LR, BS, 시간 포함)
run_name = f"effnet_lr{learning_rate}_bs{batch_size}_{datetime.now().strftime('%Y%m%d_%H%M')}"

# ============================================
# 전처리 정의
# ============================================
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================
# 데이터 로드
# ============================================
print("=" * 60)
print("데이터셋 로드 중...")
print("=" * 60)

# Dataset 한 번만 생성 (transform=None)
full_dataset = CAPTCHADataset(data_dir, transform=None)

total_size = len(full_dataset)
train_size = int(total_size * train_ratio)
val_size = total_size - train_size

# random_split → Subset 두 개 생성
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

# Subset 내부에서 참조하는 dataset(transform)을 각각 지정
train_subset.dataset.transform = train_transform
val_subset.dataset.transform = val_transform

# DataLoader 생성
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"총 데이터: {total_size}개")
print(f"학습 데이터: {train_size}개")
print(f"검증 데이터: {val_size}개")
print(f"배치 크기: {batch_size}\n")

# ============================================
# 모델 설정
# ============================================
print("=" * 60)
print("모델 설정 중...")
print("=" * 60)

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}\n")

model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# 분류층 수정 (2 클래스: Animal / Object)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.to(device)
print("✅ EfficientNet-B0 로드 완료 (2개 클래스)\n")

# ============================================
# 손실 함수 / 옵티마이저 / 스케줄러
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

# ============================================
# MLflow Run 시작
# ============================================
print("=" * 60)
print("학습 시작")
print("=" * 60 + "\n")

start_total = time.time()
best_accuracy = 0
patience_counter = 0

# Signature 준비용 예시 입력 (128x128 이미지 1개)
example_input = torch.randn(1, 3, 128, 128).to(device)
with torch.no_grad():
    example_output = model(example_input)
signature = infer_signature(example_input.cpu().numpy(),
                            example_output.cpu().numpy())

with mlflow.start_run(run_name=run_name):

    # 파라미터 로깅
    mlflow.log_param("run_name", run_name)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("patience", patience)
    mlflow.log_param("train_samples", train_size)
    mlflow.log_param("val_samples", val_size)

    # ==========================
    # Epoch Loop
    # ==========================
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # ----- 학습 -----
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

        # ----- 검증 -----
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

        # ----- 출력 -----
        print(f"[Epoch {epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2%}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2%}")
        print(f"  → Epoch Time: {epoch_time:.2f} sec")

        # ----- 과적합 체크 (참고용 출력만) -----
        overfitting_gap = train_accuracy - val_accuracy
        if overfitting_gap >= 0.10:
            print(f"  ⚠️ 과적합 경고! (차이: {overfitting_gap:.2%})")
        else:
            print(f"  ✓ 정상 (차이: {overfitting_gap:.2%})")

        # MLflow 기록
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_accuracy, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_accuracy, step=epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # ----- 🔥 표준 Early Stopping 로직 -----
        # 1) val_accuracy가 이전 best보다 크면:
        #    - best_accuracy 업데이트
        #    - patience_counter 리셋
        #    - 모델 저장 + MLflow에 log_model
        # 2) 아니라면:
        #    - patience_counter 증가
        #    - patience만큼 연속으로 개선이 없으면 Early Stopping
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

    mlflow.log_metric("best_val_acc", best_accuracy)

# ============================================
# 학습 완료 출력
# ============================================
end_total = time.time()
print("=" * 60)
print("학습 완료!")
print(f"최고 검증 정확도: {best_accuracy:.2%}")
print(f"전체 학습 시간: {end_total - start_total:.2f} sec")
print(f"모델 저장 경로: {model_save_path}")
print("=" * 60)

# ============================================
# 최종 모델 검증
# ============================================
print("\n최고 모델 로드 중...")
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

final_correct = 0
final_total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        final_total += labels.size(0)
        final_correct += (predicted == labels).sum().item()

final_accuracy = final_correct / final_total
print(f"최종 검증 정확도: {final_accuracy:.2%}")
print("✅ 모델 저장 완료:", model_save_path)
