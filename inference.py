"""
추론 코드 (CAPTCHA 데이터용) - 진짜 심플 버전
- captcha_dataset.py로 이미 전처리된 데이터 사용
- class_mapping 따로 필요 없음
- 그냥 폴더를 읽고 모델로 추론만 함
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.manifold import TSNE

# ============================================
# 1단계: 모델 준비
# ============================================
print("=" * 60)
print("모델 로드 중...")
print("=" * 60)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

# 모델 경로
model_path = "./best_model.pth"

if not os.path.exists(model_path):
    print(f"⚠️ {model_path}를 찾을 수 없습니다.")
    print("먼저 training_code_improved.py로 모델을 학습하세요.")
    exit()

# EfficientNet-B0 로드 (구조만)
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)  # 2개 클래스: Animal, Object

# 학습된 가중치 로드
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ {model_path} 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

# 마지막 분류층 제거 (특징 추출만)
model = nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

print(f"✅ 모델 준비 완료 (추론 모드)\n")

# ============================================
# 2단계: 전처리 정의
# ============================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_embedding(img_path):
    """이미지를 1280차원 특징 벡터로 변환"""
    try:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            vec = model(x)  # (1, 1280, 1, 1)
            vec = nn.functional.adaptive_avg_pool2d(vec, 1)  # (1, 1280, 1, 1)
            vec = vec.view(vec.size(0), -1)  # (1, 1280)
            vec = vec.cpu().numpy()
        
        return vec.squeeze()
    except Exception as e:
        print(f"⚠️ {img_path} 처리 실패: {e}")
        return None

# ============================================
# 3단계: 데이터 로드 (그냥 폴더 순회)
# ============================================
print("=" * 60)
print("이미지 임베딩 추출 중...")
print("=" * 60)

base_dir = "/Users/bell/Desktop/PYTHON/captcha/images"

embeddings = []
labels = []  # 폴더명: 'animal' or 'object'
filenames = []

image_count = 0

# 각 폴더 순회 (그냥 폴더명만 읽기)
for folder_name in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder_name)
    
    if not os.path.isdir(folder_path):
        continue
    
    print(f"  {folder_name} 처리 중...", end=" ")
    
    folder_count = 0
    
    # 폴더 내 이미지 처리
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(folder_path, filename)
        
        vec = get_embedding(img_path)
        if vec is not None:
            embeddings.append(vec)
            labels.append(folder_name)  # 폴더명 그대로 사용
            filenames.append(f"{folder_name}/{filename}")
            folder_count += 1
            image_count += 1
    
    print(f"({folder_count}개)")

embeddings = np.array(embeddings)

print(f"\n총 이미지 개수: {image_count}\n")

# ============================================
# 4단계: t-SNE 실행
# ============================================
print("=" * 60)
print("t-SNE 실행 중... (시간이 걸릴 수 있습니다)")
print("=" * 60)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42,
    max_iter=1000,
    verbose=1
)

result = tsne.fit_transform(embeddings)

print("✅ t-SNE 완료\n")

# ============================================
# 5단계: 그래프 시각화
# ============================================
print("=" * 60)
print("그래프 생성 중...")
print("=" * 60)

# 한국어 폰트 설정 (경고 제거)
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = 'Arial'  # 또는 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 10))

# 그래프 색칠 (동물/사물 구분)
# 폴더명이 animal 관련인지 object 관련인지 판단
animal_keywords = ['cheetah', 'chimpanzee', 'dog', 'gorilla', 'hartebeest', 'monkey', 'orangutans', 'otters', 'polar_bear']
object_keywords = ['clock', 'drawers', 'flight', 'gloves', 'golf', 'hamburger', 'pretzel', 'toaster', 'violin']

numeric_labels = []
for label in labels:
    if label in animal_keywords:
        numeric_labels.append(0)  # 동물: 빨강
    elif label in object_keywords:
        numeric_labels.append(1)  # 사물: 파랑
    else:
        numeric_labels.append(2)  # 알 수 없음

# 산점도 그리기
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

# 범례 추가 (영문만 사용)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', edgecolor='black', label='Animal'),
    Patch(facecolor='blue', edgecolor='black', label='Object')
]
plt.legend(handles=legend_elements, fontsize=12, loc='upper right')

# 제목 및 라벨 (영문만 사용)
plt.title('t-SNE Visualization of EfficientNet-B0 Features\n(Animal vs Object Classification)', 
          fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# 저장
output_path = "./tsne_visualization_captcha.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 그래프 저장됨: {output_path}")

# 논블로킹 모드로 표시 (창을 띄우고 닫을 때까지 기다리지 않음)
plt.show(block=False)  # ← 중요! block=False로 설정
plt.pause(2)  # 2초 동안 표시
plt.close()  # 자동으로 창 닫기

print("✅ 그래프 표시 완료 (자동으로 닫힘)\n")

# ============================================
# 6단계: 개별 이미지 분류 (선택사항)
# ============================================
print("\n" + "=" * 60)
print("개별 이미지 분류 (랜덤 선택)")
print("=" * 60)

import random

def classify_single_image(img_path):
    """단일 이미지를 분류하고 확률 출력"""
    
    # 이미지 로드
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    # 분류 모델 로드 (분류층 포함)
    classifier = models.efficientnet_b0(weights=None)
    in_features = classifier.classifier[1].in_features
    classifier.classifier[1] = nn.Linear(in_features, 2)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    with torch.no_grad():
        outputs = classifier(x)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    # 클래스명
    class_names = ['Animal', 'Object']
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

# 랜덤 이미지 선택
if len(filenames) > 0:
    random_idx = random.randint(0, len(filenames)-1)
    test_img_path = os.path.join(base_dir, filenames[random_idx])
    predicted_class, confidence, all_probs = classify_single_image(test_img_path)
    
    print(f"테스트 이미지: {filenames[random_idx]}")
    print(f"예측 클래스: {predicted_class}")
    print(f"신뢰도: {confidence:.2%}")
    print(f"확률분포: Animal={all_probs[0]:.2%}, Object={all_probs[1]:.2%}")
else:
    print("분류할 이미지가 없습니다.")

print("\n✅ 추론 코드 완료!")