"""Lab 2: Gaussian blur, median filter, bilateral filter, Sobel and Canny edges."""
from pathlib import Path
import cv2
import numpy as np
from labs.common import DATA, OUTPUTS, PREVIEWS


def add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 1, cv2.LINE_AA)
    return out


def run(output_dir: Path = OUTPUTS) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    img_path = DATA / "noisy-scene.png"
    if not img_path.exists():
        from data.generate_samples import generate_all
        generate_all(DATA)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Missing noisy-scene.png at {img_path}")

    # 1. Gaussian Blur
    gauss = cv2.GaussianBlur(bgr, (9, 9), 1.8)

    # 2. Median Blur
    median = cv2.medianBlur(bgr, 7)

    # 3. Bilateral Filter
    bilateral = cv2.bilateralFilter(bgr, 9, 75, 75)

    # 4. Sobel Gradient Magnitude
    gray_gauss = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_gauss, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_gauss, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobel_x, sobel_y)
    mag = np.uint8(np.clip(mag / (mag.max() + 1e-5) * 255, 0, 255))
    sobel_bgr = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

    # 5. Canny Edge Detection
    canny = cv2.Canny(gray_gauss, 65, 145)
    canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    # Build 2x3 Grid (Tile size: 360x225)
    t_w, t_h = 360, 225
    p1 = add_label(cv2.resize(bgr, (t_w, t_h)), "1. Raw Noisy Image")
    p2 = add_label(cv2.resize(gauss, (t_w, t_h)), "2. GaussianBlur (k=9, s=1.8)")
    p3 = add_label(cv2.resize(median, (t_w, t_h)), "3. medianBlur (k=7)")
    p4 = add_label(cv2.resize(bilateral, (t_w, t_h)), "4. bilateralFilter (d=9)")
    p5 = add_label(cv2.resize(sobel_bgr, (t_w, t_h)), "5. Sobel Magnitude")
    p6 = add_label(cv2.resize(canny_bgr, (t_w, t_h)), f"6. Canny Edges ({cv2.countNonZero(canny)} px)")

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    canvas = np.vstack([row1, row2])

    for target_dir in [output_dir, PREVIEWS]:
        cv2.imwrite(str(target_dir / "02-filters-and-edges.png"), canvas)

    metrics = {
        "sobel_max": float(mag.max()),
        "canny_edges": int(cv2.countNonZero(canny)),
        "canvas_shape": canvas.shape,
    }
    print(f"Lab 2 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
