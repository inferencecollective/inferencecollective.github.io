"""Lab 4: 4-Corner Homography, Perspective Correction, and Adaptive Thresholding."""
from pathlib import Path
import cv2
import numpy as np
from labs.common import DATA, OUTPUTS, PREVIEWS


def process(output_dir: Path = OUTPUTS) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    img_path = DATA / "perspective-card.png"
    corners_path = DATA / "document-corners.npy"

    if not img_path.exists() or not corners_path.exists():
        from data.generate_samples import generate_all
        generate_all(DATA)

    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Missing perspective-card.png at {img_path}")

    src_corners = np.load(str(corners_path)).astype(np.float32)

    target_w, target_h = 480, 330
    dst_corners = np.float32([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1],
    ])

    # 1. Compute 3x3 Homography Matrix
    H = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # 2. Rectify perspective
    aligned = cv2.warpPerspective(image, H, (target_w, target_h))

    # 3. High-Contrast Adaptive Thresholding
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 7
    )
    thresh_bgr = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

    # 4. Draw corner detection markers on tilted scene
    tilted_annotated = image.copy()
    pts = src_corners.astype(np.int32)
    cv2.polylines(tilted_annotated, [pts], True, (0, 255, 255), 2)
    corner_labels = ["TL (0)", "TR (1)", "BR (2)", "BL (3)"]
    for idx, (pt, label) in enumerate(zip(pts, corner_labels)):
        cv2.circle(tilted_annotated, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(tilted_annotated, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Build Side-by-Side Comparison Canvas (1080x380)
    # Resize components for clean layout
    t_h = 330
    tilted_resized = cv2.resize(tilted_annotated, (int(image.shape[1] * t_h / image.shape[0]), t_h))
    aligned_padded = aligned.copy()
    thresh_padded = thresh_bgr.copy()

    # Add header bars
    def add_header(img, text):
        out = img.copy()
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (out.shape[1], 30), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
        cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)
        return out

    p1 = add_header(tilted_resized, "1. Tilted Scene + 4 Detected Corners")
    p2 = add_header(aligned_padded, "2. warpPerspective Rectified (480x330)")
    p3 = add_header(thresh_padded, "3. Adaptive Threshold High-Contrast")

    # Combine horizontally
    canvas = np.hstack([p1, p2, p3])

    for target_dir in [output_dir, PREVIEWS]:
        cv2.imwrite(str(target_dir / "04-document-perspective.png"), canvas)

    result = {
        "aligned": aligned,
        "thresholded": thresholded,
        "homography": H.tolist(),
        "canvas_shape": canvas.shape,
    }
    print(f"Lab 4 complete: aligned shape={aligned.shape}, H={H.shape}")
    return result


run = process

if __name__ == "__main__":
    process()
