import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time   # ← 추가!

data_dir = "/Users/bell/Desktop/PYTHON/captcha/images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = models.resnet18(weights="IMAGENET1K_V1")
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(dataset.classes))

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 전체 학습 시간 측정 시작
# ---------------------------
start_total = time.time()

for epoch in range(3):

    epoch_start = time.time()   # ← 에폭 시작 시간

    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        pred = model(imgs)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_end = time.time()   # ← 에폭 끝

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"→ Epoch Time: {epoch_end - epoch_start:.2f} sec")

    # validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"Validation Accuracy: {acc:.2f}%\n")

# ---------------------------
# 전체 걸린 시간 출력
# ---------------------------
end_total = time.time()
print(f"Total Training Time: {end_total - start_total:.2f} sec")
