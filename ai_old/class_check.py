from torchvision import datasets, transforms

dataset_path = "/Users/bell/Desktop/captcha/images"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(dataset_path, transform=transform)

print("총 이미지 개수:", len(dataset))
print("클래스 목록:", dataset.classes)

# 첫 10개의 라벨 확인
for i in range(10):
    img, label = dataset[i]
    print(i, dataset.classes[label])
