# 🧠 AI Training Module
본 디렉토리는 CAPTCHA 서비스의 AI 학습 파이프라인을 관리하는 공간이며, 아래 두 가지 범주의 모델을 포함할 수 있도록 확장성을 고려해 설계되어 있습니다.

- 행동 기반 모델
  - Isolation Forest — 이상 행동 탐지 (Ticket Slice)
  - Random Forest — 드래그 패턴 기반 행동 분류 (Drag & Drop)
    ※ 행동 기반 모델은 별도 스크립트로 추가 예정
- 이미지 기반 모델
  - EfficientNet-B0 기반 이미지 분류 모델 (Drag & Drop)

본 폴더는 이러한 모델들의 전처리, 학습, 모델 관리 기능을 통합적으로 제공하며, 확장성을 고려하여 설계되어 있습니다.

---

# ✂️ Ticket Slice (행동 기반 검증 AI)

## 🌳 Isolation Forest
→ 추가 예정

# 🖱️ Drag & Drop (인지 기반 검증 AI)

## 🏕️ Random Forest
→ 추가 예정

## 🎆 EfficientNet-B0 AI 
### 📂 Directory Structure
```
ai/
 ├── training/
 │     ├── train_efficientnet.py     # 이미지 분류 학습 메인 스크립트
 │     ├── captcha_dataset.py        # 2단계 구조 Dataset 자동 라벨링
 │     ├── name_changer.py           # 이미지 파일명 정규화 스크립트
 │     └── README.md
 │
 └── inference/
       ├── image_classifier.py       # 단일 이미지 분류 (랜덤 선택 포함)
       └── README.md

```

### 💿 Data Structure 
학습 데이터는 반드시 아래와 같은 2단계 폴더 구조여야 합니다.
```
images/
 ├── animal/
 │      ├── cheetah/
 │      ├── dog/
 │      └── ...
 └── object/
        ├── toaster/
        ├── gloves/
        └── ...
```
- Layer 1: 대그룹 (예: animal, object)
- Layer 2: 세부 클래스 폴더 (예: cheetah / toaster 등)
- 이미지 파일(.jpg/.png)은 2단계 폴더 안에 위치

---

### 🚀 Training Workflow

#### 0. 실행 전 위치 설정 
AI 학습 스크립트는 반드시 다음 디렉토리에서 실행해야 합니다:
```bash
cd /Users/bell/Desktop/captcha/ai/training/
```
이 위치에서만 `./images`, `./models`, `training/`, `inference/` 등의 상대 경로가 정상적으로 연결됩니다.


#### 1. 파일명 정규화 – name_changer.py
다양한 원본 이미지 이름을 다음처럼 규칙적으로 정리합니다.
```
e.g. 
cheetah_1.jpg
cheetah_2.jpg
...
toaster_1.jpg
toaster_2.jpg
...
```
✅ 실행 명령어
```bash
python name_changer.py --data_dir ./images
```

#### 2. Dataset 자동 라벨링 — captcha_dataset.py
대그룹 폴더명 기준으로 label 자동 생성합니다. 
```
e.g.
- Animal = 0  
- Object = 1  
```
- 전체 이미지 개수 출력
- 클래스별 이미지 개수 출력
- NUM_CLASSES 자동 계산 → 모델에 직접 반영됨

#### 3. EfficientNet 학습 — train_efficientnet.py

#### 3.1 기본(default) 파라미터 실행
✅ 실행 명령어
```bash
python train_efficientnet.py \
    --data_dir ../images \
    --output_dir ../models
```
→ batch_size, learning_rate, epochs 등은 스크립트 내부 default 사용

#### 3.2 하이퍼파라미터 직접 지정 실행
✅ 실행 명령어 
```bash
python train_efficientnet.py \
    --data_dir ./images \
    --output_dir ./models \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --epochs 30 \
    --patience 5
```

#### 4. Output
학습이 끝나면 `best_model.pth(ai/models/best_model.pth)`가 저장됩니다.
→ 이 모델은 Inference 서버(`ai/inference/`)에서 자동으로 로드됩니다.

---

### 📊 MLflow Tracking
본 학습 스크립트는 MLflow로 다음을 자동 기록합니다:
- 학습 파라미터(batch, lr, epochs, patience 등)
- train/val accuracy & loss
- best model 기록
- 모델 Signature 저장

✅ 실행 명령어 
```bash
mlflow ui --port 5000
```

`train_efficientnet.py`는 기본적으로 로컬 MLflow Tracking 서버를 사용하도록 설정되어 있습니다. 
```python
mlflow.set_tracking_uri("file:./mlruns")
```
⚠️ 하지만 **서버 환경에서는 반드시 MLflow Tracking Server 주소로 변경해야 합니다.**

실제 배포 환경에서 MLflow Dashboard를 웹으로 보려면, Tracking URI를 HTTP 주소로 변경해야 합니다. 
```python
mlflow.set_tracking_uri("http://<MLFLOW_SERVER_IP>:5000")
```

#### 1. 서버가 Public IP를 가진 경우
```
http://123.45.67.89:5000
```
#### 2. 서버가 Private IP만 있는 경우 (예: 10.x.x.x)
```
ssh -L 5000:localhost:5000 ubuntu@<BASTION_PUBLIC_IP>
```
→ 서버가 <Private IP>만 가지고 있다면 Bastion 포트포워딩을 사용해 접속 가능하며, 로컬 브라우저(`http://localhost:5000`)에서 접근할 수 있습니다.  

---

### 🎯 Summary

- 모델 학습은 NUM_CLASSES 자동 추론으로 클래스 변경에 유연함  
- MLflow Tracking URI는 서버 환경에서 꼭 `<서버 IP>:5000`로 수정해야 함  
- Dataset 매핑 로직은 `captcha_dataset.py`에서 자동 처리  
- `train_efficientnet.py`는 EfficientNet 학습 + MLflow 기록을 수행하는 메인 스크립트  

