# tsne_test.py
# 실행 위치: captcha/tsne_test.py

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import models, transforms
from sklearn.manifold import TSNE


# ============================
# 1) 이미지 임베딩 모델 로드
# ============================
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 마지막 FC 제거
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        vec = model(x).squeeze().numpy()  # 2048차원
    return vec


# ============================
# 2) 이미지 로드 & 임베딩
# ============================
IMAGE_DIR = "images"   # PYTHON/captcha/images 폴더에 이미지 넣기

embeddings = []
labels = []        # 이미지 파일명 or 폴더명
files = []

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_DIR, file)
        vec = get_embedding(path)
        embeddings.append(vec)
        labels.append(file.split("_")[0])   # ex) cat_01.jpg → "cat"
        files.append(file)

embeddings = np.array(embeddings)
print("총 이미지 개수:", len(embeddings))


# ============================
# 3) t-SNE 실행
# ============================
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
result = tsne.fit_transform(embeddings)

print("t-SNE 완료")


# ============================
# 4) 그래프로 시각화
# ============================
plt.figure(figsize=(12, 10))

# 라벨을 숫자로 변환
unique_labels = list(set(labels))
color_map = {label: i for i, label in enumerate(unique_labels)}
numeric_labels = [color_map[l] for l in labels]

plt.scatter(result[:, 0], result[:, 1], c=numeric_labels, cmap="tab10")

for i, filename in enumerate(files):
    plt.text(result[i, 0], result[i, 1], filename, fontsize=7)

plt.title("t-SNE Visualization")
plt.show()
