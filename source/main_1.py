
import cv2
import numpy as np
import re
import csv
from pathlib import Path


# ========================================
# 1. 경로 및 설정
# ========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "images"
OUTPUT_DIR = BASE_DIR / "output"
CSV_PATH = OUTPUT_DIR / "face_mosaic_results.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# 검출용 이미지의 긴 변 최대 크기
MAX_DETECTION_SIZE = 1600

# 작은 얼굴 검출 기준
MIN_FACE_SIZE = 30

# 얼굴 검출 민감도
FRONTAL_NEIGHBORS = 6
PROFILE_NEIGHBORS = 5

# 얼굴 모양 필터
MIN_FACE_ASPECT = 0.60
MAX_FACE_ASPECT = 1.40

# 모자이크 영역 확장 비율
FACE_PADDING_RATIO = 0.10

# 모자이크 블록 설정
MOSAIC_SIZE = 50

# 고해상도 바운딩 박스 두께
BOX_THICKNESS_RATIO = 0.004


# ========================================
# 2. 파일명 숫자 정렬
# ========================================

def natural_sort_key(path):
    numbers = re.findall(r"\d+", path.stem)

    if numbers:
        return int(numbers[-1]), path.name.lower()

    return float("inf"), path.name.lower()


# ========================================
# 3. 한글 경로 이미지 읽기 / 저장
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
# 4. 얼굴 후보 형태 검사
# ========================================

def valid_face_box(x, y, w, h, image_width, image_height):
    if w <= 0 or h <= 0:
        return False

    # 이미지 바깥 좌표 제외
    if (
            x < 0 or y < 0
            or x + w > image_width
            or y + h > image_height
    ):
        return False

    # 너무 작은 얼굴 후보 제외
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False

    # 너비 / 높이 비율 검사
    aspect = w / h

    if not MIN_FACE_ASPECT <= aspect <= MAX_FACE_ASPECT:
        return False

    return True


# ========================================
# 5. 중복 검출 제거
# ========================================

def box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = (
            max(0, x2 - x1)
            * max(0, y2 - y1)
    )

    union = aw * ah + bw * bh - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def remove_duplicate_boxes(boxes, threshold=0.40):
    # 큰 박스를 우선 유지
    boxes = sorted(
        boxes,
        key=lambda b: b[2] * b[3],
        reverse=True
    )

    result = []

    for box in boxes:
        duplicate = any(
            box_iou(box, kept) >= threshold
            for kept in result
        )

        if not duplicate:
            result.append(box)

    return result


# ========================================
# 6. Haar Cascade 얼굴 검출
# ========================================

def detect_faces(gray, frontal_cascade, profile_cascade):
    image_height, image_width = gray.shape

    gray = cv2.equalizeHist(gray)

    min_size = max(
        24,
        int(min(image_width, image_height) * 0.018)
    )

    frontal_faces = []
    profile_faces = []

    # ------------------------------------
    # 정면 얼굴
    # ------------------------------------

    frontal_results = frontal_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=FRONTAL_NEIGHBORS,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for x, y, w, h in frontal_results:
        box = tuple(map(int, (x, y, w, h)))

        if valid_face_box(
                *box, image_width, image_height
        ):
            frontal_faces.append(box)

    # ------------------------------------
    # 측면 얼굴: 원본 방향 + 좌우 반전
    # ------------------------------------

    flipped_gray = cv2.flip(gray, 1)

    for source, flipped in (
            (gray, False),
            (flipped_gray, True)
    ):
        profile_results = profile_cascade.detectMultiScale(
            source,
            scaleFactor=1.05,
            minNeighbors=PROFILE_NEIGHBORS,
            minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for x, y, w, h in profile_results:
            x, y, w, h = map(int, (x, y, w, h))

            # 반전 이미지 좌표를 원본 좌표로 변환
            if flipped:
                x = image_width - x - w

            box = (x, y, w, h)

            if valid_face_box(
                    *box, image_width, image_height
            ):
                profile_faces.append(box)

    # 각 모델 내부 중복 제거
    frontal_faces = remove_duplicate_boxes(
        frontal_faces,
        threshold=0.40
    )

    profile_faces = remove_duplicate_boxes(
        profile_faces,
        threshold=0.40
    )

    # 두 모델의 결과 통합
    combined = frontal_faces + profile_faces

    return remove_duplicate_boxes(
        combined,
        threshold=0.40
    )


# ========================================
# 7. 검출용 이미지 축소
# ========================================

def prepare_detection_image(gray):
    height, width = gray.shape

    scale = min(
        1.0,
        MAX_DETECTION_SIZE / max(width, height)
    )

    if scale < 1.0:
        detection_gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        detection_gray = gray.copy()

    return detection_gray, scale


# ========================================
# 8. 검출 좌표를 원본 좌표로 변환
# ========================================

def restore_original_coordinates(faces, scale):
    if scale >= 1.0:
        return faces

    restored = []

    for x, y, w, h in faces:
        restored.append((
            int(round(x / scale)),
            int(round(y / scale)),
            int(round(w / scale)),
            int(round(h / scale))
        ))

    return restored


# ========================================
# 9. 얼굴 영역 확장
# ========================================

def expand_box(x, y, w, h, image_width, image_height):
    pad_x = int(w * FACE_PADDING_RATIO)
    pad_y = int(h * FACE_PADDING_RATIO)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + w + pad_x)
    y2 = min(image_height, y + h + pad_y)

    return x1, y1, x2 - x1, y2 - y1


# ========================================
# 10. 얼굴 모자이크
# ========================================

def mosaic_face(image, x, y, w, h):
    image_height, image_width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + w)
    y2 = min(image_height, y + h)

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return

    small_w = max(1, face.shape[1] // MOSAIC_SIZE)
    small_h = max(1, face.shape[0] // MOSAIC_SIZE)

    small = cv2.resize(
        face,
        (small_w, small_h),
        interpolation=cv2.INTER_LINEAR
    )

    mosaic = cv2.resize(
        small,
        (face.shape[1], face.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    image[y1:y2, x1:x2] = mosaic


# ========================================
# 11. 메인 실행
# ========================================

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Haar Cascade 얼굴 모자이크 프로그램")
    print("=" * 60)

    print(f"입력 폴더: {INPUT_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"CSV 파일: {CSV_PATH}")

    # OpenCV 기본 모델 경로
    frontal_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
    )

    profile_path = (
            cv2.data.haarcascades
            + "haarcascade_profileface.xml"
    )

    frontal_cascade = cv2.CascadeClassifier(frontal_path)
    profile_cascade = cv2.CascadeClassifier(profile_path)

    if frontal_cascade.empty():
        print("[오류] 정면 얼굴 모델 로드 실패")
        return

    if profile_cascade.empty():
        print("[오류] 측면 얼굴 모델 로드 실패")
        return

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
        print("[안내] data/images 폴더에 사진이 없습니다.")
        return

    csv_columns = [
        "image_name",
        "image_width",
        "image_height",
        "detection_scale",
        "face_index",
        "x", "y", "w", "h",
        "box_x1", "box_y1", "box_x2", "box_y2",
        "status"
    ]

    success_count = 0
    failed_count = 0
    total_faces = 0

    with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_columns
        )

        writer.writeheader()

        for index, image_path in enumerate(image_files, 1):
            print(
                f"\n[{index}/{len(image_files)}] "
                f"{image_path.name}"
            )

            image = read_image(image_path)

            if image is None:
                print("  -> 이미지 읽기 실패")
                failed_count += 1
                continue

            original_height, original_width = image.shape[:2]

            print(
                f"  원본 해상도: "
                f"{original_width} x {original_height}"
            )

            # 원본 이미지에서 회색조 변환
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # 검출용 이미지 축소
            detection_gray, scale = prepare_detection_image(gray)

            print(
                f"  검출용 해상도: "
                f"{detection_gray.shape[1]} x "
                f"{detection_gray.shape[0]}"
            )

            # 축소 이미지에서 검출
            detected_faces = detect_faces(
                detection_gray,
                frontal_cascade,
                profile_cascade
            )

            # 원본 이미지 좌표로 변환
            faces = restore_original_coordinates(
                detected_faces,
                scale
            )

            # 복원 후 범위 및 크기 재검사
            faces = [
                (x, y, w, h)
                for x, y, w, h in faces
                if valid_face_box(
                    x, y, w, h,
                    original_width,
                    original_height
                )
            ]

            faces = remove_duplicate_boxes(faces)

            faces.sort(key=lambda b: (b[1], b[0]))

            print(f"  검출 얼굴 수: {len(faces)}")

            if not faces:
                writer.writerow({
                    "image_name": image_path.name,
                    "image_width": original_width,
                    "image_height": original_height,
                    "detection_scale": round(scale, 5),
                    "face_index": "",
                    "x": "", "y": "", "w": "", "h": "",
                    "box_x1": "", "box_y1": "",
                    "box_x2": "", "box_y2": "",
                    "status": "no_face_detected"
                })

            # 원본 이미지에 모자이크 및 박스 적용
            for face_index, (x, y, w, h) in enumerate(
                    faces, start=1
            ):
                bx, by, bw, bh = expand_box(
                    x, y, w, h,
                    original_width,
                    original_height
                )

                mosaic_face(image, bx, by, bw, bh)

                thickness = max(
                    3,
                    int(
                        max(original_width, original_height)
                        * BOX_THICKNESS_RATIO
                    )
                )

                # 빨간색 바운딩 박스
                cv2.rectangle(
                    image,
                    (bx, by),
                    (bx + bw - 1, by + bh - 1),
                    (0, 0, 255),
                    thickness
                )

                font_scale = max(
                    0.7,
                    max(original_width, original_height) / 1800
                )

                cv2.putText(
                    image,
                    f"Face {face_index}",
                    (bx, max(25, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),
                    max(2, thickness // 2),
                    cv2.LINE_AA
                )

                # CSV 좌표는 원본 이미지 기준
                writer.writerow({
                    "image_name": image_path.name,
                    "image_width": original_width,
                    "image_height": original_height,
                    "detection_scale": round(scale, 5),
                    "face_index": face_index,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "box_x1": bx,
                    "box_y1": by,
                    "box_x2": bx + bw,
                    "box_y2": by + bh,
                    "status": "detected"
                })

                print(
                    f"    Face {face_index}: "
                    f"x={x}, y={y}, w={w}, h={h}"
                )

            output_path = OUTPUT_DIR / (
                f"{image_path.stem}_mosaic"
                f"{image_path.suffix}"
            )

            if save_image(output_path, image):
                print(f"  -> 저장 완료: {output_path.name}")
                success_count += 1
                total_faces += len(faces)
            else:
                print("  -> 저장 실패")
                failed_count += 1

    print("\n" + "=" * 60)
    print("작업 완료")
    print(f"전체 이미지: {len(image_files)}개")
    print(f"성공: {success_count}개")
    print(f"실패: {failed_count}개")
    print(f"총 검출 얼굴: {total_faces}개")
    print(f"결과 이미지: {OUTPUT_DIR}")
    print(f"좌표 CSV: {CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
