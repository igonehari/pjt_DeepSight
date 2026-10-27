"""03: Haar 얼굴 검출 후 픽셀화 모자이크, 원본은 그대로 유지."""
from pathlib import Path
import argparse
import sys
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from shared.common import get_cascade, detect_faces, image_files, read_image, save_image, save_csv


def mosaic(image, x, y, w, h):
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(w_img, x+w), min(h_img, y+h)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    tiny = cv2.resize(roi, (max(1, (x2-x1)//40), max(1, (y2-y1)//40)),
                      interpolation=cv2.INTER_LINEAR)
    roi[:] = cv2.resize(tiny, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description="얼굴 자동 모자이크 (검수 필수)")
    parser.add_argument("--input", default=str(ROOT / "data/images"))
    args = parser.parse_args()

    cascade = get_cascade()
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    rows = []
    for path in image_files(args.input):
        image = read_image(path)
        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 💡 [정면 검출 강화] minNeighbors를 6~7로 높이고, 최소 크기(minSize)를 키워 오인식 방지
        faces_frontal = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

        # 측면 검출 (측면은 1개 정도만 깔끔하게 잡히도록 기존 유지)
        faces_profile1 = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        # 좌우 반전 측면 검출
        flipped_gray = cv2.flip(gray, 1)
        faces_profile2 = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        corrected_profile2 = []
        for (x, y, w, h) in faces_profile2:
            new_x = w_img - x - w
            corrected_profile2.append((new_x, y, w, h))

        all_faces = list(faces_frontal) + list(faces_profile1) + list(corrected_profile2)

        print(f"[{path.name}] 정면: {len(faces_frontal)}개 | 측면1: {len(faces_profile1)}개 | 측면2(반전): {len(corrected_profile2)}개")

        # 모자이크 적용 (여유 패딩 10% 추가)
        for x, y, w, h in all_faces:
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            nx = max(0, x - pad_w)
            ny = max(0, y - pad_h)
            nw = w + (pad_w * 2)
            nh = h + (pad_h * 2)
            mosaic(image, int(nx), int(ny), int(nw), int(nh))

        save_image(ROOT / "outputs" / f"{path.stem}_mosaic.jpg", image)
        rows.append({"filename": path.name, "detected_faces": len(all_faces)})

    save_csv(ROOT / "outputs/face_mosaic_results.csv", ["filename", "detected_faces"], rows)
    print("주의: 얼굴 검출 누락 가능. 비식별화 완료로 단정하지 말고 수동 검수하세요.")

if __name__ == "__main__":
    main()