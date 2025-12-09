import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# ============================================
# 클래스 매핑: 세부 폴더 → 대그룹
# ============================================
CLASS_MAPPING = {
    # 동물 그룹 (0번)
    'cheetah': 'animal',
    'chimpanzee': 'animal',
    'dog': 'animal',
    'gorilla': 'animal',
    'hartebeest': 'animal',
    'monkey': 'animal',
    'orangutans': 'animal',
    'otters': 'animal',
    'polar_bear': 'animal',
    
    # 사물 그룹 (1번)
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

# 대그룹을 숫자로 변환
GROUP_TO_CLASS = {
    'animal': 0,
    'object': 1,
    # 나중에 추가 가능:
    # 'plant': 2,
}

# ============================================
# CAPTCHA Dataset 클래스
# ============================================
class CAPTCHADataset(Dataset):
    """
    CAPTCHA 이미지 데이터셋 클래스
    
    세부 폴더(dog, cheetah 등)를 대그룹(animal, object)으로 매핑합니다.
    
    Args:
        root_dir (str): images 폴더의 경로 (예: 'captcha/images/')
        transform (callable, optional): 이미지 전처리
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # root_dir 하위의 모든 폴더 순회
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            
            # 폴더인지 확인
            if not os.path.isdir(folder_path):
                continue
            
            # CLASS_MAPPING에서 그룹 이름 가져오기
            if folder_name not in CLASS_MAPPING:
                print(f"경고: {folder_name}은 CLASS_MAPPING에 없습니다. 건너뜀.")
                continue
            
            group_name = CLASS_MAPPING[folder_name]
            class_id = GROUP_TO_CLASS[group_name]
            
            # 폴더 내 모든 이미지 파일 수집
            for image_name in os.listdir(folder_path):
                # 이미지 파일만 처리 (jpg, png, jpeg)
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                image_path = os.path.join(folder_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(class_id)
        
        print(f"총 {len(self.image_paths)}개 이미지 로드됨")
        print(f"Animal: {sum(1 for l in self.labels if l == 0)}개")
        print(f"Object: {sum(1 for l in self.labels if l == 1)}개")
    
    def __len__(self):
        """데이터셋의 전체 샘플 개수"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        idx번째 샘플 반환
        
        Returns:
            image (Tensor): 처리된 이미지 (3, 128, 128)
            label (int): 클래스 번호 (0=animal, 1=object)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}")
            raise e
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================
# 사용 예시
# ============================================
if __name__ == "__main__":
    # 전처리 정의 (학습 데이터용)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(15),  # 데이터 증강: 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기/명도 조정
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 전처리 정의 (검증 데이터용 - 증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 데이터셋 생성 (절대경로 사용)
    print("=== 데이터셋 생성 ===")
    dataset = CAPTCHADataset('/Users/bell/Desktop/PYTHON/captcha/images/', transform=train_transform)
    
    # DataLoader 생성
    print("\n=== DataLoader 생성 ===")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # 배치 확인
    print("\n=== 배치 샘플 확인 ===")
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"배치 {batch_idx}")
        print(f"  이미지 shape: {images.shape}")  # (32, 3, 128, 128)
        print(f"  라벨: {labels}")  # [0, 1, 0, 0, 1, ...]
        
        # 첫 배치만 확인
        if batch_idx == 0:
            break
    
    print("\n데이터셋 준비 완료!")