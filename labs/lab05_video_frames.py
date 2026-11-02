"""Lab 5: Sequential video frame tracking, HSV centroid moments, and VideoWriter export."""
from pathlib import Path
import cv2
import numpy as np
from labs.common import DATA, OUTPUTS, PREVIEWS


def process(output_dir: Path = OUTPUTS) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)

    frames_dir = DATA / "frames"
    frame_paths = sorted(frames_dir.glob("frame-*.png"))

    if len(frame_paths) < 12:
        from data.generate_samples import generate_all
        generate_all(DATA)
        frame_paths = sorted(frames_dir.glob("frame-*.png"))

    trajectory = []
    annotated_frames = []

    # 1. Track object centroid across all frames
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([10, 90, 120]), np.array([35, 255, 255]))

        M = cv2.moments(mask)
        center = None
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
            trajectory.append(center)

        # Draw trajectory trail
        annotated = frame.copy()
        if len(trajectory) > 1:
            pts = np.array(trajectory, np.int32)
            cv2.polylines(annotated, [pts], False, (56, 189, 248), 2, cv2.LINE_AA)

        if center is not None:
            cv2.circle(annotated, center, 28, (0, 255, 255), 2)
            cv2.circle(annotated, center, 4, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"TARGET ({center[0]}, {center[1]})",
                (center[0] - 55, center[1] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        annotated_frames.append(annotated)

    # 2. Write short AVI video with VideoWriter
    video_path = output_dir / "05-tracking.avi"
    video_exported = False

    if annotated_frames:
        h, w = annotated_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (w, h))
        if writer.isOpened():
            try:
                for f in annotated_frames:
                    writer.write(f)
                video_exported = True
            finally:
                writer.release()

    # 3. Build Multi-Frame Composite Preview (2x2 key frames: 0, 3, 7, 11)
    key_indices = [0, 3, 7, min(11, len(annotated_frames) - 1)]
    key_tiles = []
    for idx in key_indices:
        tile = annotated_frames[idx].copy()
        cv2.putText(
            tile,
            f"KEYFRAME #{idx:02d}",
            (tile.shape[1] - 160, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (56, 189, 248),
            1,
            cv2.LINE_AA,
        )
        key_tiles.append(cv2.resize(tile, (480, 270)))

    row1 = np.hstack([key_tiles[0], key_tiles[1]])
    row2 = np.hstack([key_tiles[2], key_tiles[3]])
    canvas = np.vstack([row1, row2])

    for target_dir in [output_dir, PREVIEWS]:
        cv2.imwrite(str(target_dir / "05-video-frame-tracking.png"), canvas)

    result = {
        "frames_tracked": len(annotated_frames),
        "trajectory_points": len(trajectory),
        "video_exported": video_exported,
        "video_path": str(video_path) if video_exported else None,
        "canvas_shape": canvas.shape,
    }
    print(f"Lab 5 complete: tracked {len(annotated_frames)} frames, video_exported={video_exported}")
    return result


run = process

if __name__ == "__main__":
    process()
