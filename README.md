## 🚀 CAPTCHA AI Workspace

이 레포지토리는 CAPTCHA 서비스 개발을 위한 AI 모델링·실험 환경을 정리한 개인 작업 공간입니다.

현재 구조는 아래와 같이 구성되어 있으며,

- ai/ → 팀 프로젝트에 실제로 들어갈 최종 코드
- ai_old/ → 개인 로컬 환경에서 개발·실험했던 코드 백업

두 영역을 명확하게 분리하여 관리합니다.

---

## 📂 Directory Structure
```
CAPTCHA/
├── ai/                           # 팀 프로젝트용 최종 AI 코드
│   ├── images/                   # 2단계 구조의 학습용 이미지 데이터
│   ├── inference/                # 이미지 분류 추론 코드 (production-ready)
│   │   ├── image_classifier.py
│   │   └── README.md  
│   ├── models/                   # 학습된 best_model.pth 저장 위치
│   └── training/                 # EfficientNet 학습 파이프라인
│       ├── train_efficient.py
│       ├── captcha_dataset.py
│       ├── name_changer.py
│       └── README.md
│
├── ai_old/                       # 개인 실험 버전(실험 코드, 테스트, T-SNE 등)
│   ├── images/
│   ├── inference.py
│   ├── train.py
│   ├── mlflow.db
│   ├── tsne_test.py
│   ├── tsne_visualization.py
│   ├── training_summary.txt
│   └── model_compare/
│
├── output/                # 기타 출력 폴더
├── venv/                  # Python 가상환경 (업로드 제외)
├── .gitignore
├── README.md
└── requirements.txt
```
---

## 🧠 What’s Inside?
### 1. ai/ — 최종 작업물
팀 GitHub에 업로드될 AI 코드들이 정리되어 있으며, 실제 CAPTCHA 서비스에 들어갈 구조입니다.
- 포함 기능
  - EfficientNet-B0 이미지 분류 학습 (`training/train_efficient.py`)
  - Dataset 자동 매핑 (`captcha_dataset.py`)
  - 파일명 정규화 유틸리티(`name_changer.py`)
  - 이미지 분류 추론 코드(`inference/image_classifier.py`)
  - NUM_CLASSES 자동 계산
  - MLflow 기반 학습 이력 관리
  - 학습된 `best_model.pth` 보관

### 2. ai_old/ — 개인 실험 코드 보관소
학습 과정에서 테스트했던 코드들을 그대로 보존한 폴더입니다.
- 포함 기능
  - 초기 버전의 `train.py`, `inference.py`
  - T-SNE 시각화
  - class check / naming script 실험 버전
  - MLflow DB 파일
  - 모델 비교 결과 등 실험 로그
이 폴더는 레거시 참고용이며 팀 프로젝트에서는 사용되지 않습니다.

## 📝 Usage Summary
### 1. 이미지 파일명 정규화
```bash
python ai/training/name_changer.py --data_dir ./ai/images
```
### 2. 이미지 분류 모델 학습
```bash
cd ai

python training/train_efficient.py \
    --data_dir ./images \
    --output_dir ./models
```
### 3. 단일 이미지 분류 / 랜덤 추론
```bash
cd ai

python inference/image_classifier.py \
    --data_dir ./images \
    --model_path ./models/best_model.pth
```
### 4. mlflow 실행 
```bash
cd ai

mlflow ui --port 5000
```

### 📌 Notes
- ai/ 폴더가 팀 프로젝트에 실제로 포함되는 코드입니다.
- ai_old/는 개인 테스트 및 연구용이므로 팀 레포에는 포함되지 않습니다.
- .gitignore에 venv, mlruns, 캐시 파일이 설정되어 있습니다.

---

####  📌 Project Badges
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![MLflow](https://img.shields.io/badge/MLflow-enabled-orange)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNetB0-green)
![status](https://img.shields.io/badge/status-active-success)
![license](https://img.shields.io/badge/license-MIT-green)
