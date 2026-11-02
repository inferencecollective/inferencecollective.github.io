"""Generate deterministic synthetic visual fixtures for Pillow and OpenCV tutorials and labs."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def generate_all(data_dir: Path = None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. PILLOW FIXTURES
    # ==========================================
    # scene-01.png: Sunset & Mountains (800x600)
    img1 = Image.new("RGB", (800, 600), "#0f172a")
    d1 = ImageDraw.Draw(img1)
    for y in range(400):
        r = int(15 + (245 - 15) * (y / 400))
        g = int(23 + (110 - 23) * (y / 400))
        b = int(42 + (30 - 42) * (y / 400))
        d1.line([(0, y), (800, y)], fill=(r, g, b))
    d1.ellipse([320, 180, 480, 340], fill="#fde047")
    d1.polygon([(0, 450), (250, 220), (500, 450)], fill="#334155")
    d1.polygon([(250, 450), (550, 180), (800, 450)], fill="#1e293b")
    d1.rectangle([(0, 450), (800, 600)], fill="#0f172a")
    img1.save(data_dir / "scene-01.png")

    # scene-02.png: Architectural Grid & Textures (800x600)
    img2 = Image.new("RGB", (800, 600), "#1e1e24")
    d2 = ImageDraw.Draw(img2)
    for x in range(0, 800, 40):
        d2.line([(x, 0), (x, 600)], fill="#2d3748", width=1)
    for y in range(0, 600, 40):
        d2.line([(0, y), (800, y)], fill="#2d3748", width=1)
    for i, h in enumerate([420, 500, 380, 460, 520, 340]):
        bx = 60 + i * 115
        d2.rectangle([(bx, 600 - h), (bx + 90, 600)], fill="#3b82f6", outline="#60a5fa", width=2)
        for wy in range(600 - h + 20, 580, 30):
            for wx in range(bx + 15, bx + 75, 25):
                d2.rectangle([(wx, wy), (wx + 12, wy + 18)], fill="#fef08a")
    img2.save(data_dir / "scene-02.png")

    # scene-03.png: Forest Lake at Night (800x600)
    img3 = Image.new("RGB", (800, 600), "#090d16")
    d3 = ImageDraw.Draw(img3)
    for y in range(350):
        b = int(22 + (70 - 22) * (y / 350))
        d3.line([(0, y), (800, y)], fill=(9, 13, b))
    d3.ellipse([620, 60, 720, 160], fill="#e2e8f0")
    for tx in range(0, 820, 60):
        d3.polygon([(tx, 420), (tx + 30, 260), (tx + 60, 420)], fill="#064e3b")
        d3.polygon([(tx + 5, 420), (tx + 30, 300), (tx + 55, 420)], fill="#047857")
    d3.rectangle([(0, 420), (800, 600)], fill="#0f2b38")
    img3.save(data_dir / "scene-03.png")

    # scene-04.png: Abstract Geometric Pattern (800x600)
    img4 = Image.new("RGB", (800, 600), "#fafafa")
    d4 = ImageDraw.Draw(img4)
    colors = ["#f43f5e", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b"]
    for idx, c in enumerate(colors):
        cx = 160 * (idx + 0.5)
        cy = 300 + 80 * math.sin(idx * 1.5)
        r = 65
        d4.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c, outline="#1e293b", width=3)
    img4.save(data_dir / "scene-04.png")

    # logo.png: Transparent RGBA Watermark Badge (240x240)
    logo = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
    dl = ImageDraw.Draw(logo)
    dl.ellipse([10, 10, 230, 230], fill=(16, 185, 129, 210), outline=(255, 255, 255, 240), width=6)
    dl.polygon([(120, 40), (200, 120), (120, 200), (40, 120)], fill=(255, 255, 255, 230))
    dl.ellipse([85, 85, 155, 155], fill=(15, 23, 42, 240))
    dl.ellipse([105, 105, 135, 135], fill=(52, 211, 153, 255))
    logo.save(data_dir / "logo.png")

    # ==========================================
    # 2. OPENCV FIXTURES
    # ==========================================
    import cv2

    frames_dir = data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # color-scene.png: (height=400, width=640, channels=3, BGR)
    color_scene = np.zeros((400, 640, 3), dtype=np.uint8)
    for y in range(260):
        color_scene[y, :] = [int(180 - y * 0.4), int(120 - y * 0.2), int(40 + y * 0.1)]
    cv2.circle(color_scene, (480, 120), 45, (60, 220, 255), -1)
    pts1 = np.array([[0, 280], [180, 140], [360, 280]], np.int32)
    pts2 = np.array([[220, 280], [420, 110], [640, 280]], np.int32)
    cv2.fillPoly(color_scene, [pts1], (80, 60, 50))
    cv2.fillPoly(color_scene, [pts2], (110, 85, 70))
    color_scene[280:, :] = [45, 120, 40]
    cv2.rectangle(color_scene, (60, 300), (160, 370), (220, 80, 40), -1)
    cv2.circle(color_scene, (300, 335), 35, (40, 40, 220), -1)
    # Synthetic Orange Ball (BGR: [20, 140, 240], matches HSV [10..35, 90..255, 120..255])
    cv2.circle(color_scene, (500, 335), 35, (20, 140, 240), -1)
    cv2.imwrite(str(data_dir / "color-scene.png"), color_scene)

    # noisy-scene.png: Gaussian + Salt-and-Pepper Noise
    noisy = color_scene.copy()
    np.random.seed(42)
    gauss = np.random.normal(0, 18, noisy.shape).astype(np.float32)
    noisy = np.clip(noisy.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    sp = np.random.rand(400, 640)
    noisy[sp < 0.015] = [255, 255, 255]
    noisy[sp > 0.985] = [0, 0, 0]
    cv2.imwrite(str(data_dir / "noisy-scene.png"), noisy)

    # binary-shapes.png: 5 disconnected geometric shapes
    binary = np.zeros((400, 640), dtype=np.uint8)
    cv2.circle(binary, (100, 120), 50, 255, -1)
    cv2.rectangle(binary, (240, 70), (360, 170), 255, -1)
    tri = np.array([[500, 70], [440, 170], [560, 170]], np.int32)
    cv2.fillPoly(binary, [tri], 255)
    diamond = np.array([[180, 260], [240, 330], [180, 400], [120, 330]], np.int32)
    cv2.fillPoly(binary, [diamond], 255)
    cv2.ellipse(binary, (460, 310), (70, 45), 30, 0, 360, 255, -1)
    # small speckle noise
    for _ in range(20):
        rx, ry = np.random.randint(0, 640), np.random.randint(0, 400)
        if binary[ry, rx] == 0:
            binary[ry, rx] = 255
    cv2.imwrite(str(data_dir / "binary-shapes.png"), binary)

    # perspective-card.png & document-corners.npy
    card_w, card_h = 480, 330
    card = np.ones((card_h, card_w, 3), dtype=np.uint8) * 245
    cv2.rectangle(card, (10, 10), (card_w - 10, card_h - 10), (40, 100, 220), 4)
    cv2.putText(card, "AI DATA FOUNDATION", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 40, 80), 2)
    cv2.putText(card, "DOCUMENT PASSPORT", (45, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    cv2.line(card, (45, 135), (card_w - 45, 135), (100, 100, 100), 1)
    for idx, text in enumerate(["SPEC: IEEE-P2800 / AI TRUST", "ISSUED: 2026-11-02", "HASH: 9F8A-77C1-44B2-019E", "VALIDITY: LEVEL 4 PRODUCTION"]):
        cv2.putText(card, text, (45, 175 + idx * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.circle(card, (card_w - 75, 230), 40, (50, 160, 80), -1)

    desk = np.zeros((400, 640, 3), dtype=np.uint8)
    for y in range(400):
        desk[y, :] = [30 + (y % 15), 45 + (y % 10), 60 + (y % 20)]

    src_corners = np.float32([[110, 65], [520, 40], [565, 345], [75, 330]])
    dst_card_corners = np.float32([[0, 0], [card_w - 1, 0], [card_w - 1, card_h - 1], [0, card_h - 1]])
    H_inv = cv2.getPerspectiveTransform(dst_card_corners, src_corners)
    warped_card = cv2.warpPerspective(card, H_inv, (640, 400))
    mask_card = cv2.warpPerspective(np.ones((card_h, card_w), dtype=np.uint8) * 255, H_inv, (640, 400))
    desk[mask_card > 0] = warped_card[mask_card > 0]

    cv2.imwrite(str(data_dir / "perspective-card.png"), desk)
    np.save(str(data_dir / "document-corners.npy"), src_corners)

    # 12 sequential video tracking frames (frames/frame-00.png .. frame-11.png)
    for f_idx in range(12):
        f_img = np.zeros((360, 640, 3), dtype=np.uint8)
        for y in range(360):
            f_img[y, :] = [25 + y // 12, 30 + y // 15, 35 + y // 18]
        t = f_idx / 11.0
        bx = int(80 + t * 480)
        by = int(270 - 180 * 4 * t * (1 - t))
        cv2.circle(f_img, (bx, by), 24, (20, 140, 240), -1)
        cv2.putText(f_img, f"FRAME {f_idx:02d} | T={f_idx*0.08:.2f}s", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.imwrite(str(frames_dir / f"frame-{f_idx:02d}.png"), f_img)

    # ==========================================
    # 3. SCIKIT-IMAGE FIXTURES
    # ==========================================
    yy, xx = np.mgrid[:400, :640]

    # scene-rgb.png: Rich contrast RGB scientific scene (400x640x3, uint8)
    scene_rgb = np.zeros((400, 640, 3), dtype=np.uint8)
    for y in range(400):
        for c in range(3):
            scene_rgb[y, :, c] = int(20 + 200 * (y / 400.0) * ((c + 1) / 3.0))
    # Cellular structures in RGB channels
    for idx, (cy, cx, cr, col) in enumerate([
        (120, 160, 60, [220, 60, 80]),
        (150, 480, 70, [60, 210, 120]),
        (280, 260, 75, [70, 130, 240]),
        (290, 520, 55, [240, 190, 50]),
    ]):
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        m = dist <= cr
        for c in range(3):
            scene_rgb[m, c] = np.clip(scene_rgb[m, c] * 0.4 + col[c] * (1.0 - dist[m] / cr) * 0.8, 0, 255).astype(np.uint8)
    Image.fromarray(scene_rgb).save(data_dir / "scene-rgb.png")

    # cells-reference.png: Ground truth clean microscopy (400x640, uint8)
    cells_ref = np.zeros((400, 640), dtype=np.uint8)
    for cy, cx, r, peak in [
        (100, 120, 45, 210), (110, 320, 40, 230), (120, 520, 50, 200),
        (250, 180, 48, 220), (260, 420, 42, 240), (320, 300, 38, 190)
    ]:
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        m = dist <= r
        val = (1.0 - (dist[m] / r)**1.5) * peak
        cells_ref[m] = np.maximum(cells_ref[m], val.astype(np.uint8))
    Image.fromarray(cells_ref).save(data_dir / "cells-reference.png")

    # cells-noisy.png: Reference corrupted with Gaussian and Poisson noise
    np.random.seed(42)
    noise_gauss = np.random.normal(0, 22, cells_ref.shape)
    noise_sp = np.random.rand(*cells_ref.shape)
    cells_noisy = np.clip(cells_ref.astype(np.float32) + noise_gauss, 0, 255).astype(np.uint8)
    cells_noisy[noise_sp < 0.01] = 255
    cells_noisy[noise_sp > 0.99] = 0
    Image.fromarray(cells_noisy).save(data_dir / "cells-noisy.png")

    # cells-synthetic.png: 8 isolated cells with small speckle artifacts (< 50 px)
    cells_synth = np.zeros((400, 640), dtype=np.uint8)
    synth_centers = [
        (80, 100, 35, 25, 20),
        (90, 280, 30, 40, -15),
        (100, 480, 40, 30, 45),
        (220, 150, 35, 35, 0),
        (230, 360, 45, 25, -30),
        (240, 530, 30, 35, 10),
        (330, 240, 38, 28, 60),
        (340, 440, 32, 38, -45),
    ]
    for cy, cx, ry, rx, angle in synth_centers:
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        x_rot = cos_a * (xx - cx) + sin_a * (yy - cy)
        y_rot = -sin_a * (xx - cx) + cos_a * (yy - cy)
        dist = (x_rot / rx)**2 + (y_rot / ry)**2
        mask = dist <= 1.0
        intensity = np.clip((1.0 - dist) * 180 + 70, 0, 240)
        cells_synth[mask] = intensity[mask].astype(np.uint8)
    # Add small speckles (< 20 pixels)
    for _ in range(25):
        sy, sx = np.random.randint(20, 380), np.random.randint(20, 620)
        if cells_synth[sy, sx] == 0:
            cells_synth[sy-1:sy+2, sx-1:sx+2] = 160
    Image.fromarray(cells_synth).save(data_dir / "cells-synthetic.png")

    # cells-touching.png: 8 touching cells (4 touching pairs)
    cells_touch = np.zeros((400, 640), dtype=np.uint8)
    touching_pairs = [
        (100, 140, 30), (100, 190, 32),
        (100, 440, 35), (100, 495, 28),
        (280, 160, 32), (280, 215, 30),
        (280, 460, 30), (280, 515, 34)
    ]
    for cy, cx, r in touching_pairs:
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        mask = dist <= r
        val = np.clip((1.0 - dist / r) * 170 + 75, 0, 245)
        cells_touch[mask] = np.maximum(cells_touch[mask], val[mask].astype(np.uint8))
    Image.fromarray(cells_touch).save(data_dir / "cells-touching.png")

    # volume-cells.npy: 3D Volumetric stack (24, 80, 80) with 3 spheres
    vol = np.zeros((24, 80, 80), dtype=np.float32)
    zz, yy3, xx3 = np.mgrid[:24, :80, :80]
    spheres = [
        (6, 25, 30, 4, 12, 12),
        (12, 55, 50, 5, 14, 14),
        (18, 30, 60, 4, 10, 15)
    ]
    for cz, cy, cx, rz, ry, rx in spheres:
        dist = ((zz - cz)/rz)**2 + ((yy3 - cy)/ry)**2 + ((xx3 - cx)/rx)**2
        mask = dist <= 1.0
        vol[mask] = np.clip((1.0 - dist[mask]) * 0.8 + 0.2, 0, 1.0)
    np.save(str(data_dir / "volume-cells.npy"), vol)

    # ==========================================
    # 4. PYTORCH FIXTURES
    # ==========================================
    import torch

    # 4.1. pytorch-regression.pt: 1D linear regression with noise (100 samples)
    torch.manual_seed(42)
    x_reg = torch.linspace(-2.5, 3.5, 100).unsqueeze(1)
    y_reg = 3.5 * x_reg + 1.2 + torch.randn_like(x_reg) * 0.75
    torch.save({"x": x_reg, "y": y_reg, "true_w": 3.5, "true_b": 1.2}, data_dir / "pytorch-regression.pt")

    # 4.2. pytorch-moons.pt: 2D non-linear binary classification (300 samples)
    np.random.seed(42)
    n_moons = 300
    n_half = n_moons // 2
    theta = np.linspace(0, np.pi, n_half)
    x1 = np.stack([np.cos(theta), np.sin(theta)], axis=1) + np.random.normal(0, 0.08, (n_half, 2))
    x2 = np.stack([1.0 - np.cos(theta), 1.0 - np.sin(theta) - 0.5], axis=1) + np.random.normal(0, 0.08, (n_half, 2))
    X_moons = torch.tensor(np.vstack([x1, x2]), dtype=torch.float32)
    y_moons = torch.tensor(np.hstack([np.zeros(n_half), np.ones(n_half)]), dtype=torch.long)
    torch.save({"X": X_moons, "y": y_moons}, data_dir / "pytorch-moons.pt")

    # 4.3. pytorch-lines-16x16.pt: Horizontal (0) vs Vertical (1) line images (400 samples)
    torch.manual_seed(42)
    n_images = 400
    images = torch.zeros((n_images, 1, 16, 16), dtype=torch.float32)
    line_labels = torch.zeros(n_images, dtype=torch.long)
    for i in range(n_images):
        cls = i % 2
        line_labels[i] = cls
        img = torch.randn(1, 16, 16) * 0.08
        if cls == 0:
            r1 = int(np.random.randint(2, 14))
            img[0, r1, :] += 1.0
            if np.random.rand() > 0.4:
                r2 = int((r1 + np.random.choice([3, 4, -3, -4])) % 16)
                img[0, r2, :] += 0.8
        else:
            c1 = int(np.random.randint(2, 14))
            img[0, :, c1] += 1.0
            if np.random.rand() > 0.4:
                c2 = int((c1 + np.random.choice([3, 4, -3, -4])) % 16)
                img[0, :, c2] += 0.8
        images[i] = torch.clamp(img, 0.0, 1.0)
    torch.save({"images": images, "labels": line_labels}, data_dir / "pytorch-lines-16x16.pt")

    # 4.4. pytorch-corpus.pt: Toy text classification dataset (200 sequences, vocab 16)
    vocab = {
        "<PAD>": 0, "<UNK>": 1,
        "great": 2, "excellent": 3, "fast": 4, "clean": 5, "loved": 6, "superb": 7,
        "poor": 8, "slow": 9, "broken": 10, "dirty": 11, "hated": 12, "terrible": 13,
        "system": 14, "query": 15
    }
    torch.manual_seed(42)
    n_seqs = 200
    seq_len = 6
    tokens = torch.zeros((n_seqs, seq_len), dtype=torch.long)
    text_labels = torch.zeros(n_seqs, dtype=torch.long)
    pos_ids = [2, 3, 4, 5, 6, 7]
    neg_ids = [8, 9, 10, 11, 12, 13]
    neutral_ids = [14, 15]
    for i in range(n_seqs):
        label = i % 2
        text_labels[i] = label
        pool = pos_ids if label == 1 else neg_ids
        actual_len = int(np.random.randint(2, seq_len + 1))
        for t in range(actual_len):
            if np.random.rand() > 0.3:
                tokens[i, t] = int(np.random.choice(pool))
            else:
                tokens[i, t] = int(np.random.choice(neutral_ids))
    torch.save({"tokens": tokens, "labels": text_labels, "vocab": vocab}, data_dir / "pytorch-corpus.pt")

    # 4.5. pytorch-sequences.pt: Cyclic arithmetic sequence patterns for causal transformer
    torch.manual_seed(42)
    n_transformer_seqs = 300
    t_seq_len = 8
    seq_data = torch.zeros((n_transformer_seqs, t_seq_len), dtype=torch.long)
    for i in range(n_transformer_seqs):
        start = (i * 3 + 1) % 10
        stride = (i % 3) + 1
        for t in range(t_seq_len):
            seq_data[i, t] = (start + t * stride) % 10 + 2
    torch.save({"sequences": seq_data, "vocab_size": 12, "seq_len": t_seq_len}, data_dir / "pytorch-sequences.pt")

    # ==========================================
    # 5. TORCHVISION FIXTURES: synthetic_shapes
    # ==========================================
    shapes_root = data_dir / "synthetic_shapes"
    splits = {
        "train": 24,  # 24 images per class = 72 images
        "val": 8,     # 8 images per class = 24 images
        "test": 8     # 8 images per class = 24 images
    }
    classes = ["diagonal", "horizontal", "vertical"]
    colors = [
        (56, 189, 248),   # Sky blue
        (52, 211, 153),   # Emerald
        (253, 224, 71),   # Yellow
        (244, 63, 94),    # Rose
        (167, 139, 250),  # Purple
    ]

    np.random.seed(42)
    for split_name, count in splits.items():
        for cls_name in classes:
            target_dir = shapes_root / split_name / cls_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                img_arr = np.zeros((64, 64, 3), dtype=np.uint8)
                base_b = int(np.random.randint(15, 30))
                img_arr[:, :] = [base_b // 2, base_b // 2 + 5, base_b]
                pil_img = Image.fromarray(img_arr)
                draw = ImageDraw.Draw(pil_img)
                color = colors[idx % len(colors)]
                w = int(np.random.randint(2, 5))

                if cls_name == "horizontal":
                    y1 = 12 + (idx * 7) % 40
                    draw.line([(6, y1), (58, y1)], fill=color, width=w)
                    if idx % 2 == 1:
                        y2 = (y1 + 18) % 52 + 6
                        draw.line([(8, y2), (56, y2)], fill=color, width=max(1, w - 1))
                elif cls_name == "vertical":
                    x1 = 12 + (idx * 7) % 40
                    draw.line([(x1, 6), (x1, 58)], fill=color, width=w)
                    if idx % 2 == 1:
                        x2 = (x1 + 18) % 52 + 6
                        draw.line([(x2, 8), (x2, 56)], fill=color, width=max(1, w - 1))
                elif cls_name == "diagonal":
                    if idx % 2 == 0:
                        draw.line([(6, 6), (58, 58)], fill=color, width=w)
                    else:
                        draw.line([(6, 58), (58, 6)], fill=color, width=w)

                pil_img.save(target_dir / f"shape_{idx:03d}.png")

    # ==========================================
    # 6. TENSORFLOW FIXTURES
    # ==========================================
    # A. 1D Linear regression dataset for tf.GradientTape
    np.random.seed(42)
    x_reg = np.linspace(-5.0, 5.0, 200, dtype=np.float32).reshape(-1, 1)
    true_w = 3.0
    true_b = -12.0
    noise_reg = np.random.normal(0.0, 0.4, size=x_reg.shape).astype(np.float32)
    y_reg = true_w * x_reg + true_b + noise_reg
    np.savez(data_dir / "tf-regression.npz", x=x_reg, y=y_reg, true_w=true_w, true_b=true_b)
    np.save(data_dir / "tf-regression.npy", np.hstack([x_reg, y_reg]))

    # B. 2D Non-linear classification dataset (moons) for Keras
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    X_moons, y_moons = make_moons(n_samples=500, noise=0.20, random_state=42)
    X_moons = X_moons.astype(np.float32)
    y_moons = y_moons.astype(np.int32)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_moons, y_moons, test_size=100, random_state=42, stratify=y_moons
    )
    np.savez(
        data_dir / "tf-moons.npz",
        X_train=X_train_m,
        X_test=X_test_m,
        y_train=y_train_m,
        y_test=y_test_m
    )

    # C. 28x28 Grayscale synthetic shapes dataset (circles vs squares) for tf.data + CNN
    # Shape format: (N, 28, 28, 1) float32 in [0, 1]
    def generate_shape_images(count: int, seed: int):
        np.random.seed(seed)
        imgs = np.zeros((count, 28, 28, 1), dtype=np.float32)
        labels = np.zeros((count,), dtype=np.int32)
        for i in range(count):
            label = i % 2  # 0: circle, 1: square
            labels[i] = label
            canvas = np.zeros((28, 28), dtype=np.uint8)
            cx = int(np.random.randint(11, 17))
            cy = int(np.random.randint(11, 17))
            radius = int(np.random.randint(6, 10))
            if label == 0:
                cv2.circle(canvas, (cx, cy), radius, 255, -1)
            else:
                cv2.rectangle(canvas, (cx - radius, cy - radius), (cx + radius, cy + radius), 255, -1)
            # Add subtle Gaussian noise
            noise = np.random.normal(0, 10, (28, 28)).astype(np.float32)
            canvas_f = np.clip(canvas.astype(np.float32) + noise, 0, 255) / 255.0
            imgs[i, :, :, 0] = canvas_f
        return imgs, labels

    X_train_s, y_train_s = generate_shape_images(240, seed=101)
    X_test_s, y_test_s = generate_shape_images(60, seed=202)
    np.savez(
        data_dir / "tf-shapes-28x28.npz",
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train_s,
        y_test=y_test_s
    )

    # ==========================================
    # 7. KERAS 3 FIXTURES
    # ==========================================
    # A. 1D Linear regression: y ≈ 2.5x - 0.8 + noise
    np.random.seed(42)
    x_all = np.linspace(-3.0, 3.0, 200, dtype=np.float32).reshape(-1, 1)
    true_slope = 2.5
    true_intercept = -0.8
    noise_k = np.random.normal(0.0, 0.35, size=x_all.shape).astype(np.float32)
    y_all = true_slope * x_all + true_intercept + noise_k

    x_train_k, x_test_k = x_all[:150], x_all[150:]
    y_train_k, y_test_k = y_all[:150], y_all[150:]
    np.savez(
        data_dir / "keras-regression.npz",
        x_train=x_train_k,
        x_test=x_test_k,
        y_train=y_train_k,
        y_test=y_test_k,
        true_slope=true_slope,
        true_intercept=true_intercept
    )

    # B. 2D Concentric Circles: Train/Val/Test for EarlyStopping
    np.random.seed(42)
    n_points = 400
    # Inside circle: r in [0, 0.6] -> class 1 (inner)
    r_in = np.random.uniform(0.0, 0.55, size=n_points // 2)
    theta_in = np.random.uniform(0.0, 2 * np.pi, size=n_points // 2)
    x1_in = r_in * np.cos(theta_in)
    x2_in = r_in * np.sin(theta_in)
    pts_in = np.column_stack([x1_in, x2_in])
    labels_in = np.ones(n_points // 2, dtype=np.int32)

    # Outside ring: r in [0.75, 1.3] -> class 0 (outer)
    r_out = np.random.uniform(0.75, 1.25, size=n_points // 2)
    theta_out = np.random.uniform(0.0, 2 * np.pi, size=n_points // 2)
    x1_out = r_out * np.cos(theta_out)
    x2_out = r_out * np.sin(theta_out)
    pts_out = np.column_stack([x1_out, x2_out])
    labels_out = np.zeros(n_points // 2, dtype=np.int32)

    X_circ = np.vstack([pts_in, pts_out]).astype(np.float32)
    y_circ = np.concatenate([labels_in, labels_out]).astype(np.int32)

    perm = np.random.permutation(n_points)
    X_circ = X_circ[perm]
    y_circ = y_circ[perm]

    # Split 280 train, 60 val, 60 test
    X_tr_c, y_tr_c = X_circ[:280], y_circ[:280]
    X_va_c, y_va_c = X_circ[280:340], y_circ[280:340]
    X_te_c, y_te_c = X_circ[340:], y_circ[340:]
    np.savez(
        data_dir / "keras-circles.npz",
        X_train=X_tr_c,
        X_val=X_va_c,
        X_test=X_te_c,
        y_train=y_tr_c,
        y_val=y_va_c,
        y_test=y_te_c
    )

    # C. 24x24 Grayscale Bar Patterns (Horizontal vs Vertical) in Channels-Last
    def generate_bar_images(count: int, seed: int):
        np.random.seed(seed)
        imgs = np.zeros((count, 24, 24, 1), dtype=np.float32)
        labels = np.zeros((count,), dtype=np.int32)
        for i in range(count):
            label = i % 2  # 0: horizontal bar, 1: vertical bar
            labels[i] = label
            canvas = np.zeros((24, 24), dtype=np.uint8)
            pos = int(np.random.randint(6, 18))
            thickness = int(np.random.randint(3, 6))
            if label == 0:
                canvas[max(0, pos - thickness // 2):min(24, pos + thickness // 2 + 1), :] = 255
            else:
                canvas[:, max(0, pos - thickness // 2):min(24, pos + thickness // 2 + 1)] = 255
            # Add subtle Gaussian noise
            noise = np.random.normal(0, 10, (24, 24)).astype(np.float32)
            canvas_f = np.clip(canvas.astype(np.float32) + noise, 0, 255) / 255.0
            imgs[i, :, :, 0] = canvas_f
        return imgs, labels

    X_tr_b, y_tr_b = generate_bar_images(200, seed=301)
    X_te_b, y_te_b = generate_bar_images(60, seed=402)
    np.savez(
        data_dir / "keras-bars-24x24.npz",
        X_train=X_tr_b,
        X_test=X_te_b,
        y_train=y_tr_b,
        y_test=y_te_b
    )

    # D. Integer Token Sequences of shape (batch, 8)
    np.random.seed(42)
    vocab_size = 32
    target_token = 7
    seq_len = 8
    total_seqs = 400
    seqs = np.random.randint(0, vocab_size, size=(total_seqs, seq_len), dtype=np.int32)
    # Ensure balanced targets: half have target_token, half do not
    seq_labels = np.zeros(total_seqs, dtype=np.int32)
    for i in range(total_seqs // 2):
        # Insert target_token at random index
        rand_pos = np.random.randint(0, seq_len)
        seqs[i, rand_pos] = target_token
        seq_labels[i] = 1
    for i in range(total_seqs // 2, total_seqs):
        # Ensure no target_token
        seqs[i][seqs[i] == target_token] = (target_token + 1) % vocab_size
        seq_labels[i] = 0

    perm_seq = np.random.permutation(total_seqs)
    seqs = seqs[perm_seq]
    seq_labels = seq_labels[perm_seq]

    np.savez(
        data_dir / "keras-sequences.npz",
        X_train=seqs[:320],
        X_test=seqs[320:],
        y_train=seq_labels[:320],
        y_test=seq_labels[320:],
        target_token=target_token,
        vocab_size=vocab_size
    )

    print(f"Generated all Pillow, OpenCV, scikit-image, PyTorch, TorchVision, TensorFlow and Keras 3 synthetic fixtures in {data_dir} successfully!")


if __name__ == "__main__":
    generate_all()
