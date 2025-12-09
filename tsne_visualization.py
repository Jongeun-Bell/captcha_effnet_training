"""
t-SNE 시각화 코드
학습된 EfficientNet-B0 모델의 특징을 2D로 시각화합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.manifold import TSNE

# ============================
# 1) 모델 준비
# ============================
print("모델 로드 중...")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 장치: {device}")

# EfficientNet-B0 로드 (구조만, 가중치는 없음)
model = models.efficientnet_b0(weights=None)  # ← weights=None (중요!)
model.classifier[1] = nn.Linear(1280, 2)  # 동물(0), 사물(1)

# 학습된 모델 가중치 로드
try:
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    print("✅ best_model.pth 로드 완료 (efficient에서 학습한 가중치)")
except FileNotFoundError:
    print("⚠️ best_model.pth를 찾을 수 없습니다.")
    print("먼저 efficient_with_captcha_dataset.py로 모델을 학습하세요.")
    exit()

# 마지막 분류층 제거 (특징 추출만)
model = nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

# 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_embedding(img_path):
    """이미지를 특징 벡터로 변환"""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        vec = model(x)  # (1, 1280, 1, 1)
        vec = nn.functional.adaptive_avg_pool2d(vec, 1)  # (1, 1280, 1, 1)
        vec = vec.view(vec.size(0), -1)  # (1, 1280)
        vec = vec.cpu().numpy()
    
    return vec.squeeze()


# ============================
# 2) 이미지 로드 & 임베딩 추출
# ============================
print("\n이미지 임베딩 추출 중...")

BASE_DIR = "/Users/bell/Desktop/captcha/images"

# 클래스 정의
CLASS_MAPPING = {
    # 동물
    'cheetah': 'animal',
    'chimpanzee': 'animal',
    'dog': 'animal',
    'gorilla': 'animal',
    'hartebeest': 'animal',
    'monkey': 'animal',
    'orangutans': 'animal',
    'otters': 'animal',
    'polar_bear': 'animal',
    
    # 사물
    'clock': 'object',
    'drawers': 'object',
    'flight': 'object',
    'gloves': 'object',
    'golf': 'object',
    'hamburger': 'object',
    'pretzel': 'object',
    'toaster': 'object',
    'violin': 'object',
}

embeddings = []
labels = []  # 그룹명: 'animal' or 'object'
filenames = []

image_count = 0

# 각 폴더 순회
for folder_name in sorted(os.listdir(BASE_DIR)):
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    if not os.path.isdir(folder_path):
        continue
    
    # 그룹 이름 가져오기
    group_name = CLASS_MAPPING.get(folder_name, None)
    if group_name is None:
        print(f"⚠️ {folder_name}은 CLASS_MAPPING에 없습니다. 건너뜀.")
        continue
    
    print(f"  {folder_name} ({group_name}) 처리 중...", end=" ")
    
    folder_count = 0
    
    # 폴더 내 이미지 처리
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(folder_path, filename)
        
        try:
            vec = get_embedding(img_path)
            embeddings.append(vec)
            labels.append(group_name)
            filenames.append(f"{folder_name}/{filename}")
            folder_count += 1
            image_count += 1
        except Exception as e:
            print(f"\n⚠️ {img_path} 처리 실패: {e}")
            continue
    
    print(f"({folder_count}개)")

embeddings = np.array(embeddings)

print(f"\n총 이미지 개수: {image_count}")
print(f"Animal: {sum(1 for l in labels if l == 'animal')}개")
print(f"Object: {sum(1 for l in labels if l == 'object')}개")


# ============================
# 3) t-SNE 실행
# ============================
print("\nt-SNE 실행 중... (시간이 걸릴 수 있습니다)")

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, max_iter=1000)
result = tsne.fit_transform(embeddings)

print("✅ t-SNE 완료")


# ============================
# 4) 그래프 시각화
# ============================
print("\n그래프 생성 중...")

plt.figure(figsize=(14, 10))

# 라벨을 숫자로 변환
unique_labels = list(set(labels))
color_map = {'animal': 0, 'object': 1}
numeric_labels = [color_map[l] for l in labels]

# 산점도 그리기
colors = ['red', 'blue']
scatter = plt.scatter(
    result[:, 0], 
    result[:, 1], 
    c=numeric_labels, 
    cmap="RdYlBu",
    s=100,
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# 범례 추가
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', edgecolor='black', label='Animal'),
    Patch(facecolor='blue', edgecolor='black', label='Object')
]
plt.legend(handles=legend_elements, fontsize=12, loc='upper right')

# 제목 및 라벨
plt.title('t-SNE Visualization of EfficientNet-B0 Features\n(Animal vs Object)', 
          fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True, alpha=0.3)

# 텍스트 추가 (선택사항 - 너무 많으면 겹칠 수 있음)
# for i, filename in enumerate(filenames):
#     plt.text(result[i, 0], result[i, 1], filename.split('/')[-1], fontsize=6, alpha=0.5)

plt.tight_layout()

# 저장
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장됨: tsne_visualization.png")

# 표시
plt.show()

print("\n완료!")