
import cv2
import numpy as np
import re
from pathlib import Path


# ========================================
# 1. 경로 설정
# ========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "images"
OUTPUT_DIR = BASE_DIR / "outputs"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ========================================
# 2. 숫자 기준 정렬
# ========================================

def natural_sort_key(path):
    numbers = re.findall(r'\d+', path.stem)

    if numbers:
        return (int(numbers[-1]), path.name.lower())

    return (float("inf"), path.name.lower())


# ========================================
# 3. 얼굴 모자이크
# ========================================

def mosaic_face(image, x, y, w, h, mosaic_size=80):
    face = image[y:y + h, x:x + w]

    if face.size == 0:
        return

    small_w = max(1, w // mosaic_size)
    small_h = max(1, h // mosaic_size)

    small = cv2.resize(
        face,
        (small_w, small_h),
        interpolation=cv2.INTER_LINEAR
    )

    mosaic = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    image[y:y + h, x:x + w] = mosaic


# ========================================
# 4. 한글 경로 이미지 읽기
# ========================================

def read_image(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"[읽기 오류] {path.name}: {e}")
        return None


# ========================================
# 5. 한글 경로 이미지 저장
# ========================================

def save_image(path, image):
    try:
        success, encoded = cv2.imencode(
            path.suffix,
            image
        )

        if success:
            encoded.tofile(str(path))
            return True

        return False

    except Exception as e:
        print(f"[저장 오류] {path.name}: {e}")
        return False


# ========================================
# 6. 메인 실행
# ========================================

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("     Haar Cascade 얼굴 모자이크 프로그램")
    print("=" * 50)

    print(f"입력 폴더: {INPUT_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")

    # Haar Cascade 모델 사용
    cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("[오류] Haar Cascade 모델 로드 실패")
        return

    # 이미지 목록 숫자 기준 정렬
    image_files = sorted(
        [
            p for p in INPUT_DIR.iterdir()
            if p.is_file()
               and p.suffix.lower() in IMAGE_EXTENSIONS
               and not p.stem.lower().endswith("_mosaic")
        ],
        key=natural_sort_key
    )

    if not image_files:
        print("[안내] images 폴더에 사진이 없습니다.")
        return

    print(f"\n총 {len(image_files)}개 이미지 처리 시작\n")

    success_count = 0
    failed_count = 0

    for index, image_path in enumerate(image_files, start=1):
        print(f"[{index}/{len(image_files)}] {image_path.name}")

        image = read_image(image_path)

        if image is None:
            print("  -> 이미지 읽기 실패\n")
            failed_count += 1
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        # 오검출을 줄이도록 설정한 Haar Cascade
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 검출된 얼굴에 모자이크 적용
        for (x, y, w, h) in faces:
            mosaic_face(
                image,
                x, y, w, h,
                #모자이크 크기 수치
                mosaic_size=80
            )

        output_name = (
            f"{image_path.stem}_mosaic"
            f"{image_path.suffix}"
        )

        output_path = OUTPUT_DIR / output_name

        if save_image(output_path, image):
            print(
                f"  -> 저장 완료: {output_name}"
                f" | 검출 얼굴: {len(faces)}개\n"
            )
            success_count += 1
        else:
            print("  -> 저장 실패\n")
            failed_count += 1

    print("=" * 50)
    print("작업 완료")
    print(f"전체 이미지: {len(image_files)}개")
    print(f"성공: {success_count}개")
    print(f"실패: {failed_count}개")
    print(f"저장 위치: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
