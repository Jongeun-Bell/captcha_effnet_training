import os

# images 폴더 절대경로로 바꾸기
root = "/Users/bell/Desktop/captcha/images"

for cls in os.listdir(root):
    folder = os.path.join(root, cls)

    # 폴더가 아니면 스킵
    if not os.path.isdir(folder):
        continue

    print(f"\n[처리 중] 클래스 폴더: {cls}")
    i = 1

    # 이미지 파일을 정렬해서 순서 일정하게 만들기 (중요)
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            ext = filename.split(".")[-1].lower()
            new_name = f"{cls}_{i}.{ext}"

            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)

            os.rename(old_path, new_path)

            print(f"  {filename} → {new_name}")
            i += 1

print("\n✔ 모든 폴더 변경 완료!")
