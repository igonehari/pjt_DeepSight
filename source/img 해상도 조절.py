import cv2
import numpy as np
from pathlib import Path

# ========================================
# 경로 및 설정
# ========================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "images"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# 이미지 최대 크기 제한 (800px)
MAX_IMAGE_SIZE = 600


# ========================================
# 한글 경로 이미지 읽기 / 저장 함수
# ========================================

def read_image(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[읽기 오류] {path.name}: {e}")
        return None


def save_image(path, image):
    try:
        success, encoded = cv2.imencode(path.suffix, image)
        if success:
            encoded.tofile(str(path))
            return True
        return False
    except Exception as e:
        print(f"[저장 오류] {path.name}: {e}")
        return False


# ========================================
# 메인 실행: 800px 초과 이미지 리사이즈
# ========================================

def main():
    if not INPUT_DIR.exists():
        print(f"[오류] 입력 폴더가 존재하지 않습니다: {INPUT_DIR}")
        return

    print("=" * 60)
    print("이미지 해상도 일괄 전처리 프로그램 (최대 긴 변 800px)")
    print("=" * 60)
    print(f"대상 폴더: {INPUT_DIR}\n")

    processed_count = 0
    skipped_count = 0

    for image_path in INPUT_DIR.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # 이미 모자이크 처리된 결과물 이미지는 제외
        if image_path.stem.lower().endswith("_mosaic"):
            continue

        image = read_image(image_path)
        if image is None:
            skipped_count += 1
            continue

        h, w = image.shape[:2]
        max_dim = max(w, h)

        # 긴 변이 800px을 초과하는 경우에만 축소
        if max_dim > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized_image = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            if save_image(image_path, resized_image):
                print(f"[변환 완료] {image_path.name}: {w}x{h} -> {new_w}x{new_h}")
                processed_count += 1
            else:
                print(f"[저장 실패] {image_path.name}")
                skipped_count += 1
        else:
            print(f"[유지됨] {image_path.name}: {w}x{h} (800px 이하)")
            skipped_count += 1

    print("\n" + "=" * 60)
    print("전처리 작업 완료")
    print(f"변환된 이미지: {processed_count}개")
    print(f"유지되거나 건너뛴 이미지: {skipped_count}개")
    print("=" * 60)


if __name__ == "__main__":
    main()