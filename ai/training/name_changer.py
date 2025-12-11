import os
import argparse

def rename_images(root: str) -> None:
    """
    2단계 폴더(images/group/classname/) 구조 지원.
    출력은 최소화하여 그룹 단위 완료 메시지만 표시.
    """

    root = os.path.abspath(root)
    print(f"[시작] 이미지 루트 경로: {root}")

    if not os.path.isdir(root):
        raise NotADirectoryError(f"유효하지 않은 폴더입니다: {root}")

    # 1단계 폴더: animal, object 등
    for group in sorted(os.listdir(root)):
        group_path = os.path.join(root, group)
        if not os.path.isdir(group_path):
            continue

        class_count = 0
        renamed_count = 0

        # 2단계 폴더: cheetah, toaster 등
        for cls in sorted(os.listdir(group_path)):
            cls_path = os.path.join(group_path, cls)
            if not os.path.isdir(cls_path):
                continue

            class_count += 1
            i = 1

            for filename in sorted(os.listdir(cls_path)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    ext = filename.split(".")[-1].lower()
                    new_name = f"{cls}_{i}.{ext}"

                    old_path = os.path.join(cls_path, filename)
                    new_path = os.path.join(cls_path, new_name)

                    if old_path != new_path:
                        os.rename(old_path, new_path)
                        renamed_count += 1

                    i += 1

        print(f"[그룹 처리 완료] {group} (클래스 {class_count}개, 변경된 이미지 {renamed_count}개)")

    print("\n✔ 전체 리네이밍 작업 완료!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2단계 이미지 폴더의 파일 이름을 정규화하는 스크립트"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="images 폴더의 최상위 경로 (예: ./images)"
    )
    args = parser.parse_args()

    rename_images(args.data_dir)
