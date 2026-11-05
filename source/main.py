
import cv2
import numpy as np
import re
import csv

from pathlib import Path
from ultralytics import YOLO


# ========================================
# 1. 경로 및 설정
# ========================================

# 프로젝트 루트: pjt_DeepSight
BASE_DIR = Path(__file__).resolve().parent.parent

# 입력 및 출력 경로
INPUT_DIR = BASE_DIR / "data" / "images"
OUTPUT_DIR = BASE_DIR / "outputs"

CSV_PATH = OUTPUT_DIR / "face_mosaic_results.csv"
GROUND_TRUTH_PATH = BASE_DIR / "data" / "ground_truth.csv"
EVAL_CSV_PATH = OUTPUT_DIR / "evaluation_results.csv"

# YOLOv8-Face 모델 경로
YOLO_MODEL_PATH = (
        BASE_DIR / "shared" / "models" / "yolov8n-face-lindevs.pt"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ========================================
# 2. 검출 설정
# ========================================

# 최소 얼굴 크기
MIN_FACE_SIZE = 50

# Haar Cascade 민감도
FRONTAL_NEIGHBORS = 9
PROFILE_NEIGHBORS = 9

# 얼굴 형태 필터
MIN_FACE_ASPECT = 0.60
MAX_FACE_ASPECT = 1.40

# 모자이크 영역 확장
FACE_PADDING_RATIO = 0.00

# 모자이크 블록 크기
MOSAIC_SIZE = 6

# 바운딩 박스 두께
BOX_THICKNESS_RATIO = 0.004

# YOLO 설정
YOLO_CONFIDENCE = 0.35
YOLO_IMAGE_SIZE = 960
YOLO_NMS_IOU = 0.5

# YOLO와 Haar 간 중복 판단 기준
YOLO_HAAR_IOU_THRESHOLD = 0.30


# ========================================
# 3. 파일명 숫자 정렬
# ========================================

def natural_sort_key(path):

    numbers = re.findall(r"\d+", path.stem)

    if numbers:
        return int(numbers[-1]), path.name.lower()

    return float("inf"), path.name.lower()


# ========================================
# 4. 한글 경로 이미지 읽기 / 저장
# ========================================

def read_image(path):

    try:
        data = np.fromfile(str(path), dtype=np.uint8)

        return cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

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
# 5. 얼굴 후보 형태 검사
# ========================================

def valid_face_box(
        x, y, w, h,
        image_width,
        image_height
):

    if w <= 0 or h <= 0:
        return False

    if (
            x < 0 or y < 0
            or x + w > image_width
            or y + h > image_height
    ):
        return False

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False

    aspect = w / h

    if not MIN_FACE_ASPECT <= aspect <= MAX_FACE_ASPECT:
        return False

    return True


# ========================================
# 6. 바운딩 박스 IoU 계산
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

    union = (
            aw * ah
            + bw * bh
            - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ========================================
# 7. 중복 검출 제거
# ========================================

def remove_duplicate_boxes(
        boxes,
        threshold=0.40
):

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
# 8. Haar Cascade 얼굴 검출
# ========================================

def detect_faces(
        gray,
        frontal_cascade,
        profile_cascade
):

    image_height, image_width = gray.shape

    gray = cv2.equalizeHist(gray)

    min_size = max(
        24,
        int(min(image_width, image_height) * 0.018)
    )

    frontal_faces = []
    profile_faces = []

    # 정면 얼굴 검출
    frontal_results = frontal_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=FRONTAL_NEIGHBORS,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for x, y, w, h in frontal_results:

        box = tuple(
            map(int, (x, y, w, h))
        )

        if valid_face_box(
                *box,
                image_width,
                image_height
        ):
            frontal_faces.append(box)

    # 좌우 측면 얼굴 검출
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

            x, y, w, h = map(
                int,
                (x, y, w, h)
            )

            if flipped:
                x = image_width - x - w

            box = (x, y, w, h)

            if valid_face_box(
                    *box,
                    image_width,
                    image_height
            ):
                profile_faces.append(box)

    frontal_faces = remove_duplicate_boxes(
        frontal_faces,
        threshold=0.40
    )

    profile_faces = remove_duplicate_boxes(
        profile_faces,
        threshold=0.40
    )

    combined = frontal_faces + profile_faces

    return remove_duplicate_boxes(
        combined,
        threshold=0.40
    )


# ========================================
# 9. YOLOv8-Face 얼굴 검출
# ========================================

def detect_yolo_faces(
        image,
        yolo_model
):

    results = yolo_model.predict(
        source=image,
        imgsz=YOLO_IMAGE_SIZE,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_NMS_IOU,
        verbose=False
    )

    faces = []

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box
            )

            x = x1
            y = y1

            w = x2 - x1
            h = y2 - y1

            faces.append(
                (x, y, w, h)
            )

    return faces


# ========================================
# 10. YOLO + Haar 검출 결과 통합
# ========================================

def merge_face_boxes(
        yolo_faces,
        haar_faces
):

    # YOLO 결과 우선 적용
    merged_faces = list(yolo_faces)

    # Haar에서 추가로 검출된 얼굴만 적용
    for haar_box in haar_faces:

        duplicate = False

        for yolo_box in yolo_faces:

            if box_iou(
                    haar_box,
                    yolo_box
            ) >= YOLO_HAAR_IOU_THRESHOLD:

                duplicate = True
                break

        if not duplicate:
            merged_faces.append(haar_box)

    # 최종 중복 제거
    return remove_duplicate_boxes(
        merged_faces,
        threshold=0.40
    )


# ========================================
# 11. 얼굴 영역 확장
# ========================================

def expand_box(
        x, y, w, h,
        image_width,
        image_height
):

    pad_x = int(w * FACE_PADDING_RATIO)
    pad_y = int(h * FACE_PADDING_RATIO)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)

    x2 = min(
        image_width,
        x + w + pad_x
    )

    y2 = min(
        image_height,
        y + h + pad_y
    )

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )


# ========================================
# 12. 얼굴 모자이크
# ========================================

def mosaic_face(
        image,
        x, y, w, h
):

    image_height, image_width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(image_width, x + w)
    y2 = min(image_height, y + h)

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return

    small_w = max(
        1,
        face.shape[1] // MOSAIC_SIZE
    )

    small_h = max(
        1,
        face.shape[0] // MOSAIC_SIZE
    )

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
# 13. Ground Truth 비교 및 평가
# ========================================

def evaluate_results():

    if not GROUND_TRUTH_PATH.exists():

        print(
            "\n[안내] ground_truth.csv 파일이 없어 "
            "평가를 건너뜁니다."
        )

        return

    gt_data = {}

    try:

        with open(
                GROUND_TRUTH_PATH,
                mode="r",
                encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                filename = row["파일명"].strip()

                try:
                    gt_count = int(row["실제 얼굴 수"])

                except (ValueError, TypeError):
                    gt_count = 0

                gt_data[filename] = gt_count

    except Exception as e:

        print(
            f"[평가 오류] ground_truth.csv 읽기 실패: {e}"
        )

        return

    detected_counts = {
        filename: 0
        for filename in gt_data.keys()
    }

    if CSV_PATH.exists():

        try:

            with open(
                    CSV_PATH,
                    mode="r",
                    encoding="utf-8-sig"
            ) as f:

                reader = csv.DictReader(f)

                for row in reader:

                    fname = row["image_name"]
                    status = row["status"]

                    if (
                            fname in detected_counts
                            and status == "detected"
                    ):

                        detected_counts[fname] += 1

        except Exception as e:

            print(
                f"[평가 오류] face_mosaic_results.csv 읽기 실패: {e}"
            )

            return

    total_evaluated = len(gt_data)

    exact_matches = 0
    over_detections = 0
    under_detections = 0

    eval_results = []

    for filename, gt_count in gt_data.items():

        det_count = detected_counts.get(
            filename,
            0
        )

        if det_count == gt_count:

            result_status = "Exact Match"
            exact_matches += 1

        elif det_count > gt_count:

            result_status = "Over-detection"
            over_detections += 1

        else:

            result_status = "Under-detection"
            under_detections += 1

        eval_results.append({

            "filename": filename,
            "ground_truth": gt_count,
            "detected": det_count,
            "evaluation": result_status

        })

    accuracy = (
        exact_matches / total_evaluated * 100
        if total_evaluated > 0
        else 0
    )

    eval_csv_columns = [
        "filename",
        "ground_truth",
        "detected",
        "evaluation"
    ]

    try:

        with open(
                EVAL_CSV_PATH,
                mode="w",
                newline="",
                encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=eval_csv_columns
            )

            writer.writeheader()
            writer.writerows(eval_results)

    except Exception as e:

        print(
            f"[평가 오류] evaluation_results.csv 저장 실패: {e}"
        )

    print("\n" + "=" * 60)
    print("모자이크 처리 성능 평가 결과")
    print("=" * 60)

    print(f"평가 대상 이미지 수: {total_evaluated}개")
    print(f"과검출: {over_detections}개")
    print(f"미검출: {under_detections}개")

    print(f"평가 결과 저장: {EVAL_CSV_PATH}")

    print("=" * 60)


# ========================================
# 14. 메인 실행
# ========================================

def main():

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("YOLOv8-Face + Haar Cascade 얼굴 모자이크")
    print("=" * 60)

    print(f"프로젝트 경로: {BASE_DIR}")
    print(f"입력 폴더: {INPUT_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"YOLO 모델: {YOLO_MODEL_PATH}")
    print(f"좌표 CSV: {CSV_PATH}")

    # ------------------------------------
    # Haar Cascade 모델 로드
    # ------------------------------------

    frontal_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
    )

    profile_path = (
            cv2.data.haarcascades
            + "haarcascade_profileface.xml"
    )

    frontal_cascade = cv2.CascadeClassifier(
        frontal_path
    )

    profile_cascade = cv2.CascadeClassifier(
        profile_path
    )

    if frontal_cascade.empty():

        print("[오류] 정면 얼굴 모델 로드 실패")
        return

    if profile_cascade.empty():

        print("[오류] 측면 얼굴 모델 로드 실패")
        return

    print("[완료] Haar Cascade 모델 로드 성공")

    # ------------------------------------
    # YOLO 모델 로드
    # ------------------------------------

    if not YOLO_MODEL_PATH.exists():

        print(
            f"[오류] YOLO 모델 파일이 없습니다: "
            f"{YOLO_MODEL_PATH}"
        )

        return

    try:

        yolo_model = YOLO(
            str(YOLO_MODEL_PATH)
        )

        print("[완료] YOLOv8-Face 모델 로드 성공")

    except Exception as e:

        print(
            f"[오류] YOLO 모델 로드 실패: {e}"
        )

        return

    # ------------------------------------
    # 이미지 파일 검색
    # ------------------------------------

    image_files = sorted(
        [
            p for p in INPUT_DIR.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
                and not p.stem.lower().endswith("_mosaic")
        )
        ],
        key=natural_sort_key
    )

    if not image_files:

        print(
            "[안내] data/images 폴더에 사진이 없습니다."
        )

        return

    # ------------------------------------
    # CSV 컬럼
    # ------------------------------------

    csv_columns = [

        "image_name",
        "image_width",
        "image_height",

        "face_index",

        "x", "y", "w", "h",

        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",

        "status"

    ]

    success_count = 0
    failed_count = 0
    total_faces = 0

    # ------------------------------------
    # CSV 저장 및 이미지 처리
    # ------------------------------------

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

        for index, image_path in enumerate(
                image_files,
                1
        ):

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

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # ====================================
            # 1. YOLO 얼굴 검출
            # ====================================

            try:

                yolo_faces = detect_yolo_faces(
                    image,
                    yolo_model
                )

            except Exception as e:

                print(
                    f"  [YOLO 검출 오류] {e}"
                )

                failed_count += 1
                continue

            # ====================================
            # 2. Haar 얼굴 검출
            # ====================================

            haar_faces = detect_faces(
                gray,
                frontal_cascade,
                profile_cascade
            )

            # ====================================
            # 3. 유효한 좌표 검사
            # ====================================

            yolo_faces = [

                (x, y, w, h)

                for x, y, w, h in yolo_faces

                if valid_face_box(
                    x, y, w, h,
                    original_width,
                    original_height
                )

            ]

            haar_faces = [

                (x, y, w, h)

                for x, y, w, h in haar_faces

                if valid_face_box(
                    x, y, w, h,
                    original_width,
                    original_height
                )

            ]

            # ====================================
            # 4. 두 모델 결과 통합
            # ====================================

            faces = merge_face_boxes(
                yolo_faces,
                haar_faces
            )

            faces.sort(
                key=lambda b: (b[1], b[0])
            )

            print(
                f"  YOLO 검출: {len(yolo_faces)}개"
            )

            print(
                f"  Haar 검출: {len(haar_faces)}개"
            )

            print(
                f"  최종 검출: {len(faces)}개"
            )

            # ====================================
            # 5. 얼굴이 없는 경우
            # ====================================

            if not faces:

                writer.writerow({

                    "image_name": image_path.name,
                    "image_width": original_width,
                    "image_height": original_height,

                    "face_index": "",

                    "x": "",
                    "y": "",
                    "w": "",
                    "h": "",

                    "box_x1": "",
                    "box_y1": "",
                    "box_x2": "",
                    "box_y2": "",

                    "status": "no_face_detected"

                })

            # ====================================
            # 6. 모자이크 및 바운딩 박스
            # ====================================

            for face_index, (x, y, w, h) in enumerate(
                    faces,
                    start=1
            ):

                bx, by, bw, bh = expand_box(
                    x, y, w, h,
                    original_width,
                    original_height
                )

                # 얼굴 모자이크
                mosaic_face(
                    image,
                    bx, by, bw, bh
                )

                # 바운딩 박스 두께
                thickness = max(
                    3,
                    int(
                        max(
                            original_width,
                            original_height
                        ) * BOX_THICKNESS_RATIO
                    )
                )

                # 빨간색 사각형
                cv2.rectangle(
                    image,

                    (bx, by),

                    (
                        bx + bw - 1,
                        by + bh - 1
                    ),

                    (0, 0, 255),

                    thickness
                )

                # 글자 크기
                font_scale = max(
                    0.7,
                    max(
                        original_width,
                        original_height
                    ) / 1800
                )

                # 얼굴 번호
                cv2.putText(
                    image,

                    f"Face {face_index}",

                    (
                        bx,
                        max(25, by - 8)
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    font_scale,

                    (0, 0, 255),

                    max(2, thickness // 2),

                    cv2.LINE_AA
                )

                # CSV 좌표 저장
                writer.writerow({

                    "image_name": image_path.name,

                    "image_width": original_width,
                    "image_height": original_height,

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

            # ====================================
            # 7. 결과 이미지 저장
            # ====================================

            output_path = OUTPUT_DIR / (
                f"{image_path.stem}_mosaic"
                f"{image_path.suffix}"
            )

            if save_image(
                    output_path,
                    image
            ):

                print(
                    f"  -> 저장 완료: {output_path.name}"
                )

                success_count += 1
                total_faces += len(faces)

            else:

                print("  -> 저장 실패")

                failed_count += 1

    # ========================================
    # 8. 작업 결과 출력
    # ========================================

    print("\n" + "=" * 60)
    print("작업 완료")
    print("=" * 60)

    print(f"전체 이미지: {len(image_files)}개")
    print(f"총 검출 얼굴 수: {total_faces}개")
    print(f"결과 이미지 폴더: {OUTPUT_DIR}")
    print(f"좌표 CSV: {CSV_PATH}")

    print("=" * 60)

    # ========================================
    # 9. Ground Truth 평가
    # ========================================

    evaluate_results()


# ========================================
# 15. 프로그램 실행
# ========================================

if __name__ == "__main__":

    main()
