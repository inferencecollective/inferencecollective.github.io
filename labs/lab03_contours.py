"""Lab 3: Binary thresholding, morphological filtering, and contour bounding boxes."""
from pathlib import Path
import cv2
import numpy as np
from labs.common import DATA, OUTPUTS, PREVIEWS


def add_banner(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 34), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.putText(out, text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 250), 1, cv2.LINE_AA)
    return out


def run(output_dir: Path = OUTPUTS) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    img_path = DATA / "binary-shapes.png"
    if not img_path.exists():
        from data.generate_samples import generate_all
        generate_all(DATA)

    raw_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise FileNotFoundError(f"Missing binary-shapes.png at {img_path}")

    # 1. Binary Threshold
    _, thresh = cv2.threshold(raw_gray, 127, 255, cv2.THRESH_BINARY)

    # 2. Morphological Opening (Remove small speckle noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 3. Find Contours
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Filter contours and draw bounding boxes on color canvas
    annotated = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    detected_shapes = []

    colors = [(59, 130, 246), (16, 185, 129), (239, 68, 68), (245, 158, 11), (168, 85, 247)]

    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        color = colors[idx % len(colors)]

        # Draw contour outline
        cv2.drawContours(annotated, [c], -1, color, 2)
        # Draw bounding rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # Draw centroid
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        label = f"#{len(detected_shapes)+1} (A={int(area)})"
        cv2.putText(annotated, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        detected_shapes.append({"index": idx, "area": area, "bbox": (x, y, w, h)})

    # Build 2x2 comparison panel
    raw_bgr = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    t_w, t_h = 480, 300
    p1 = add_banner(cv2.resize(raw_bgr, (t_w, t_h)), "1. Raw Gray with Noise Specks")
    p2 = add_banner(cv2.resize(thresh_bgr, (t_w, t_h)), "2. cv2.threshold(127, 255)")
    p3 = add_banner(cv2.resize(cleaned_bgr, (t_w, t_h)), "3. Morphology MORPH_OPEN (Specks Removed)")
    p4 = add_banner(cv2.resize(annotated, (t_w, t_h)), f"4. Bounding Boxes ({len(detected_shapes)} Objects Detected)")

    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    canvas = np.vstack([row1, row2])

    for target_dir in [output_dir, PREVIEWS]:
        cv2.imwrite(str(target_dir / "03-contours-and-morphology.png"), canvas)

    metrics = {
        "raw_contours": len(contours),
        "filtered_shapes": len(detected_shapes),
        "canvas_shape": canvas.shape,
    }
    print(f"Lab 3 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
