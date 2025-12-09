## 📘 CAPTCHA Image Classification with EfficientNet-B0
EfficientNet-B0 기반으로 동물(Animal) / 사물(Object) 이미지 분류를 수행하는 프로젝트이다.
ImageNet 일부를 활용해 데이터를 재구성하였으며, PyTorch, MLflow, EfficientNet을 활용해 모델 학습–평가–추론–시각화(t-SNE) 파이프라인을 구축했다.

---

## 📂 Project Structure
```
captcha/
├─ images/                 # 학습 이미지 (세부 클래스별 폴더)
├─ class_check.py          # 데이터셋 구조 점검 스크립트
├─ captcha_dataset.py      # 커스텀 Dataset (세부 클래스 → 대그룹 매핑)
├─ train.py                # EfficientNet-B0 학습 + MLflow 추적
├─ inference.py            # 학습된 모델로 단일 이미지 분류(Inference) + 선택적 임베딩 추출(t-SNE용)
├─ tsne_test.py            # ResNet50 기반 간단 t-SNE 테스트용
├─ tsne_visualization.py   # 학습된 EfficientNet 기반 t-SNE 시각화
├─ name_changer.py         # 이미지 파일명 일괄 변경 스크립트
└─ best_model.pth          # 학습된 모델 가중치(학습 후 생성)
```
---

### 1. Dataset Overview
학습 데이터는 ImageNet 기반 특정 카테고리를 추출해 다음과 같이 구성됨
```
images/
├─ cheetah/
├─ chimpanzee/
├─ dog/
├─ gorilla/
├─ hartebeest/
├─ ...
├─ clock/
├─ drawers/
├─ flight/
└─ gloves/
```
- 세부 클래스 → 대그룹 매핑 방식 사용
- 프로젝트는 “목적에 맞게 단순화된 2-Class CAPTCHA 모델”을 사용하며, 이 매핑 로직은 captcha_dataset.py 내부에 존재 

| 세부 클래스                 | 대그룹    |
| ------------------------ | ------ |
| cheetah, dog, monkey 등   | animal |
| clock, gloves, toaster 등 | object |


### 2. Model Overview
- Base Model
  - EfficientNet-B0 (ImageNet Pretrained)
  - classifier 마지막 레이어만 수정 → 2-class 출력 (후에 변경 가능)
- Loss / Optimizer / Scheduler
  - CrossEntropyLoss
  - Adam (lr=0.0003)
  - StepLR(step_size=2, gamma=0.9)
- Early Stopping
  - 검증 정확도(val_acc)가 상승하지 않는 epoch가 patience(=3) 이상이면 자동 종료
  - 과적합(gap) 여부는 참고용 메시지 출력만 하고, 모델 저장 조건에는 관여하지 않음



  ### 3. Training (train.py)

- 주요 기능
  - Dataset 1회 생성 → random_split → transform 분리 적용
  - EfficientNet-B0 pretrained 모델 사용
  - MLflow 실험 기록
  - Model Signature 자동 기록 (infer_signature)
  - Best Model 자동 저장(best_model.pth)
  - Early Stopping (patience=3)
  - 최종 검증 정확도 표시



### 4. Inference (inference.py)
학습이 끝난 EfficientNet-B0 모델을 이용해 실제 이미지가 제대로 분류되는지 검증하는 추론 코드이다.
  - 단일 이미지 분류(Classification inference)
  - 모델 특징 벡터(embedding) 추출
  - 전체 데이터셋에 대해 t-SNE 시각화 수행



### 5. Visualization (TSNE)
프로젝트는 두 종류의 t-SNE 코드 제공한다.

#### 5.1 tsne_test.py — ResNet50 기반 간단 버전
- 특징
  - ImageNet pretrained ResNet50 사용
  - 마지막 FC 제거하여 2048차원 임베딩 추출
  - images/ 폴더에 있는 파일 단일 레벨 기준
  - 파일명 앞 prefix로 라벨 처리 (ex) dog_01 → dog)
- 사용 목적
  - “데이터 자체가 어떻게 분포되어 있는가?” 빠르게 체크 가능
  - 모델 학습 없이도 t-SNE 가능

#### 5.2 tsne_visualization.py — EfficientNet 실제 학습 기반 버전
- 특징
  - 학습된 best_model.pth 기반으로 임베딩 1280차원 추출
  - CLASS_MAPPING 기반 animal/object 라벨 지정
  - 산점도 시각화
  - legend 포함 (Animal=red, Object=blue)
  - 최종 그래프를 tsne_visualization.png로 저장



### 6. Utility Scripts
#### 6.1 name_changer.py
각 폴더 내 이미지 파일을 규칙적인 형식으로 재정렬하는 역할을 하며, 이를 통해 파일명을 정규화시키고, 시각화/분석 시 파일명 일관성 유지할 수 있다. 
#### 6.2 class_check.py
Dataset을 ImageFolder처럼 인식시키기 전에 구조가 잘 구성되었는지 빠르게 검증할 수 있는 스크립트로, 총 이미지 수, 클래스 목록, 첫 10개 라벨을 출력한다.



### 7. Custom Dataset (captcha_dataset.py)
- 기능 요약
  - 폴더명(dog, gloves 등) → 대그룹(animal/object) 자동 매핑
  - 이미지 경로 리스트 구성
  - transform 적용
  - DataLoader로 학습/검증 분리 가능
- 사용 이유
  - 기본 ImageFolder는 “폴더 = 클래스”
  - → 본 프로젝트는 여러 세부 클래스를 하나의 그룹으로 합쳐 학습 필요 → 따라서 커스텀 Dataset 필수



### 8. Setup & Environment
- 추천 버전
  - Python 3.11
  - PyTorch 2.x
  - macOS: device="mps" 자동 지원
  - MLflow 필수
- 설치 
```
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
- 필요 라이브러리
```
torch
torchvision
mlflow
numpy
pillow
matplotlib
scikit-learn
```


### 9. How to Train → Evaluate → Visualize
#### (1) 데이터 정리 
```
images/
   ├─ dog/
   ├─ cheetah/
   ├─ gloves/
   ├─ toaster/
   ...
```
#### (2) 이름 정리 (Optional)
```
python name_changer.py
```
#### (3) 학습 실행
```
python train.py
```
train.py 완료 후:
- best_model.pth 생성
- MLflow에서 그래프 확인 가능
#### (4) t-SNE 시각화 (학습 기반)
```
python tsne_visualization.py
```
#### (5) 모델 추론
```
python inference.py
```

---

####  📌 Project Badges
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![MLflow](https://img.shields.io/badge/MLflow-enabled-orange)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNetB0-green)
![status](https://img.shields.io/badge/status-active-success)
![license](https://img.shields.io/badge/license-MIT-green)
