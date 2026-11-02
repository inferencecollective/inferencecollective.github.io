"""Lab 1: OpenCV image loading, BGR/RGB conversion, HSV color masking, and resizing."""
from pathlib import Path
import cv2
import numpy as np
from labs.common import DATA, OUTPUTS, PREVIEWS


def add_banner(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    # Draw dark translucent banner at bottom
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 34), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.putText(out, text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 250), 1, cv2.LINE_AA)
    return out


def run(output_dir: Path = OUTPUTS) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    img_path = DATA / "color-scene.png"
    if not img_path.exists():
        from data.generate_samples import generate_all
        generate_all(DATA)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")

    h, w, c = bgr.shape

    # 1. Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. HSV color segmentation of the synthetic orange ball
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 90, 120])
    upper_orange = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    masked_ball = cv2.bitwise_and(bgr, bgr, mask=mask)

    # 3. Resized (shrink to 320x200 with INTER_AREA)
    small = cv2.resize(bgr, (320, 200), interpolation=cv2.INTER_AREA)

    # 4. Rotation with warpAffine
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 0.9)
    rotated = cv2.warpAffine(bgr, M, (w, h), borderValue=(15, 23, 42))

    # Assemble 2x2 comparison preview (640x400 tiles)
    p1 = add_banner(bgr, f"1. Original BGR ({w}x{h})")
    p2 = add_banner(gray_bgr, "2. cv2.COLOR_BGR2GRAY")
    p3 = add_banner(masked_ball, f"3. HSV Orange Ball Mask ({int(np.count_nonzero(mask))} px)")
    p4 = add_banner(rotated, "4. warpAffine 15° Rotation")

    tile_w, tile_h = 480, 300
    r1 = cv2.resize(p1, (tile_w, tile_h))
    r2 = cv2.resize(p2, (tile_w, tile_h))
    r3 = cv2.resize(p3, (tile_w, tile_h))
    r4 = cv2.resize(p4, (tile_w, tile_h))

    row1 = np.hstack([r1, r2])
    row2 = np.hstack([r3, r4])
    canvas = np.vstack([row1, row2])

    for target_dir in [output_dir, PREVIEWS]:
        cv2.imwrite(str(target_dir / "01-color-and-resize.png"), canvas)

    metrics = {
        "original_shape": bgr.shape,
        "orange_pixels": int(np.count_nonzero(mask)),
        "small_shape": small.shape,
        "preview_shape": canvas.shape,
    }
    print(f"Lab 1 complete: {metrics}")
    return metrics


if __name__ == "__main__":
    run()
