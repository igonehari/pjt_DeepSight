
import cv2
from pathlib import Path

# 프로젝트 최상위 폴더: pjt_mosaic
BASE_DIR = Path(__file__).resolve().parent.parent

# 폴더 경로
INPUT_DIR = BASE_DIR / "data" / "images"
VIDEO_DIR = BASE_DIR / "data" / "video"
OUTPUT_DIR = BASE_DIR / "output"

# 지원 이미지 확장자
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def mosaic_face(image, x, y, w, h, mosaic_size=40):
    """얼굴 영역을 모자이크 처리"""
    face = image[y:y + h, x:x + w]

    if face.size == 0:
        return

    # 축소 크기를 최대 40x40으로 설정
    small_w = min(mosaic_size, max(1, w // 40))
    small_h = min(mosaic_size, max(1, h // 40))

    small = cv2.resize(
        face,
        (small_w, small_h),
        interpolation=cv2.INTER_LINEAR
    )

    # 원래 얼굴 크기로 확대
    mosaic = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    image[y:y + h, x:x + w] = mosaic


def main():
    # 폴더가 없으면 자동 생성
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Haar Cascade 얼굴 검출기
    cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("얼굴 검출기를 불러오지 못했습니다.")
        return

    # images 폴더의 이미지 파일 검색
    image_files = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file()
           and p.suffix.lower() in IMAGE_EXTENSIONS
           and not p.stem.lower().endswith("_mosaic")
    ]

    if not image_files:
        print(f"이미지가 없습니다: {INPUT_DIR}")
        return

    print(f"이미지 {len(image_files)}개 처리 시작\n")

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[실패] 이미지 읽기 오류: {image_path.name}")
            continue

        # 얼굴 검출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 모든 검출 얼굴에 모자이크 적용
        for (x, y, w, h) in faces:
            mosaic_face(image, x, y, w, h, mosaic_size=40)

        # 파일명 뒤에 _mosaic 추가
        output_name = (
            f"{image_path.stem}_mosaic{image_path.suffix}"
        )
        output_path = OUTPUT_DIR / output_name

        # 결과 저장
        if cv2.imwrite(str(output_path), image):
            print(
                f"[완료] {output_name} "
                f"| 얼굴 {len(faces)}개 검출"
            )
        else:
            print(f"[실패] 저장 오류: {output_name}")

    print("\n모든 작업이 완료되었습니다.")
    print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
