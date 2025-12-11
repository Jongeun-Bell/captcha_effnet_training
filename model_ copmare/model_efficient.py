import torch
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
import time

# CAPTCHADataset 임포트 (같은 폴더에 있어야 함)
from captcha_dataset import CAPTCHADataset

# ============================================
# 데이터 로드
# ============================================
data_dir = "/Users/bell/Desktop/PYTHON/captcha/images/"

# 전처리 정의 (학습용 - 증강 포함)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),  # 데이터 증강
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 데이터 증강
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 전처리 정의 (검증용 - 증강 없음)
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# CAPTCHADataset으로 데이터셋 로드 (동물/사물로 그룹핑됨)
print("CAPTCHADataset 로드 중...")
full_dataset = CAPTCHADataset(data_dir, transform=train_transform)

# 학습/검증 분할
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# DataLoader 생성
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

print(f"총 데이터: {len(full_dataset)}개")
print(f"학습 데이터: {len(train_ds)}개")
print(f"검증 데이터: {len(val_ds)}개")
print(f"클래스: Animal(0), Object(1) - 2개 그룹\n")

# ============================================
# 모델 설정
# ============================================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 장치: {device}\n")

# EfficientNet-B0 로드 (사전학습 가중치)
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# 분류층 수정 (2개 클래스: 동물, 사물)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.to(device)

# ============================================
# 손실 함수, 옵티마이저, 스케줄러 설정
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

# ============================================
# 학습 루프
# ============================================
print("=" * 60)
print("학습 시작")
print("=" * 60 + "\n")

start_total = time.time()

best_accuracy = 0
patience_counter = 0
patience = 3

for epoch in range(10):
    epoch_start = time.time()
    
    # ========== 학습 단계 ==========
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        # 장치로 이동
        imgs, labels = imgs.to(device), labels.to(device)
        
        # 기울기 초기화
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        # 손실 누적
        train_loss += loss.item()
        
        # 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # 학습 에포크 평균
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    
    epoch_end = time.time()
    
    # ========== 검증 단계 ==========
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = val_correct / val_total
    
    # 학습률 감소
    scheduler.step()
    
    # ========== 결과 출력 ==========
    print(f"[Epoch {epoch+1}/10] Train Loss: {avg_train_loss:.4f}")
    print(f"  Train Acc: {train_accuracy:.2%} | Val Acc: {val_accuracy:.2%}")
    print(f"  → Epoch Time: {epoch_end - epoch_start:.2f} sec")
    
    # ========== 과적합 판단 ==========
    overfitting_gap = train_accuracy - val_accuracy
    if overfitting_gap >= 0.10:
        print(f"  ⚠️ 과적합 경고! (Train-Val 차이: {overfitting_gap:.2%})")
    
    # ========== 최고 성능 모델 저장 ==========
    if val_accuracy > best_accuracy and overfitting_gap < 0.10:
        best_accuracy = val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✅ 최고 성능 모델 저장! (Val_Acc: {val_accuracy:.2%})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n🛑 Early Stopping! 최고 검증 정확도: {best_accuracy:.2%}")
            break
    
    print()

# ============================================
# 전체 학습 시간 출력
# ============================================
end_total = time.time()
total_time = end_total - start_total

print("=" * 60)
print(f"학습 완료!")
print(f"최고 검증 정확도: {best_accuracy:.2%}")
print(f"전체 학습 시간: {total_time:.2f} sec ({total_time/60:.2f} min)")
print("=" * 60)

# ============================================
# 최고 성능 모델 로드 및 최종 검증
# ============================================
print("\n최고 성능 모델 로드 중...")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

final_correct = 0
final_total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        final_total += labels.size(0)
        final_correct += (predicted == labels).sum().item()

final_accuracy = final_correct / final_total
print(f"최종 검증 정확도: {final_accuracy:.2%}")
print(f"✅ 모델 저장됨: best_model.pth")