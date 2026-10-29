"""Генерация синтетических тестовых видео с известным положением объекта (ground truth).

Объект - плоская текстурированная «обложка» 1280x720. Он рассматривается как
прямоугольник в 3D, для каждого кадра задаётся поза камеры (повороты и сдвиг),
углы объекта проецируются моделью камеры-обскуры и объект переносится в кадр
гомографией. Первый кадр каждого видео - объект во весь кадр, как в задании.

Видео:
  test1_easy        - сдвиг, небольшой поворот и масштаб, чистое изображение
  test2_perspective - полный поворот 360°, сильный наклон (перспектива), масштаб x3
  test3_hard        - перекрытие, смаз, шум, перепады яркости, сжатие JPEG,
                      уход объекта за край и полностью из кадра

Для каждого видео рядом сохраняется <имя>_gt.npz: corners (N,4,2) - углы объекта,
visible (N,) - видимая доля объекта (с учётом границ кадра и перекрытия).
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import data

ROOT = Path(__file__).parent
W, H = 1280, 720
FPS = 30
F = 1000.0                                  # фокусное расстояние, px
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
OBJ_W, OBJ_H = 1.6, 0.9                     # размер объекта в «метрах»


def font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def make_object() -> np.ndarray:
    """Текстурированная «коробка настольной игры» 1280x720 (BGR)."""
    img = Image.new("RGB", (W, H), (245, 235, 210))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(1)
    # фон из разноцветных кругов - много углов и текстуры
    for _ in range(140):
        x, y, r = rng.integers(0, W), rng.integers(0, H), rng.integers(8, 40)
        d.ellipse([x - r, y - r, x + r, y + r], outline=tuple(int(v) for v in rng.integers(60, 200, 3)), width=3)
    img.paste(Image.fromarray(data.astronaut()).resize((560, 560)), (40, 80))
    img.paste(Image.fromarray(data.chelsea()).resize((300, 200)), (930, 470))
    img.paste(Image.fromarray(data.coffee()).resize((280, 186)), (630, 484))
    d.rectangle([0, 0, W - 1, H - 1], outline=(150, 30, 30), width=18)
    d.text((640, 70), "COMPUTER", font=font(96, True), fill=(150, 30, 30))
    d.text((640, 175), "VISION", font=font(96, True), fill=(30, 60, 150))
    d.text((645, 300), "keypoints  ·  descriptors\nmatching  ·  homography", font=font(36), fill=(30, 30, 30))
    d.text((645, 410), "ITMO 2026 · Lab 2", font=font(40, True), fill=(20, 110, 60))
    # «штрихкод»
    x = 1060
    for wbar in rng.integers(2, 8, 30):
        d.rectangle([x, 380, x + wbar, 450], fill=(0, 0, 0))
        x += wbar + int(rng.integers(2, 5))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def make_background(seed: int) -> np.ndarray:
    """Фон с собственной текстурой (город) - источник ложных соответствий."""
    town = cv2.imread(str(ROOT.parent / "LW1" / "images" / "input" / "town.jpg"))
    bg = cv2.resize(town, (int(W * 1.4), int(W * 1.4 * town.shape[0] / town.shape[1])))
    if seed % 2:
        bg = cv2.flip(bg, 1)
    return bg


def rotation(yaw, pitch, roll):
    """Матрица поворота по углам (градусы): roll - вокруг оси взгляда."""
    y, p, r = np.radians([yaw, pitch, roll])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Rz @ Ry @ Rx


def project_corners(yaw, pitch, roll, tx, ty, tz):
    """Углы объекта (TL, TR, BR, BL) в пикселях кадра."""
    obj = np.array([[-OBJ_W / 2, -OBJ_H / 2, 0], [OBJ_W / 2, -OBJ_H / 2, 0],
                    [OBJ_W / 2, OBJ_H / 2, 0], [-OBJ_W / 2, OBJ_H / 2, 0]])
    cam = obj @ rotation(yaw, pitch, roll).T + [tx, ty, tz]
    p = cam @ K.T
    return p[:, :2] / p[:, 2:]


def render_object(frame, obj_pyr, corners):
    """Переносит объект в кадр по гомографии; возвращает маску объекта.
    При сильном уменьшении берётся уровень пирамиды, чтобы не было алиасинга."""
    scale = np.sqrt(cv2.contourArea(corners.astype(np.float32)) / (W * H))
    lvl = int(np.clip(np.floor(np.log2(1 / max(scale, 1e-3) / 1.3)), 0, len(obj_pyr) - 1))
    src = obj_pyr[lvl]
    sh, sw = src.shape[:2]
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]), corners.astype(np.float32))
    warped = cv2.warpPerspective(src, Hm, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), Hm, (W, H), flags=cv2.INTER_LINEAR)
    a = mask[..., None].astype(np.float32) / 255
    frame[:] = (warped * a + frame * (1 - a)).astype(np.uint8)
    return mask > 127


def motion_blur(img, dx, dy):
    length = int(np.hypot(dx, dy))
    if length < 2:
        return img
    k = np.zeros((length * 2 + 1, length * 2 + 1), np.float32)
    c = length
    cv2.line(k, (int(c - dx / 2), int(c - dy / 2)), (int(c + dx / 2), int(c + dy / 2)), 1, 1, cv2.LINE_AA)
    return cv2.filter2D(img, -1, k / k.sum())


# --------------------------------------------------------------- сценарии
def pose_easy(t):
    """t в [0, 1]. Возвращает (yaw, pitch, roll, tx, ty, tz)."""
    s = 2 * np.pi * t
    return (8 * np.sin(s), 5 * np.sin(2 * s), 10 * np.sin(s),
            0.5 * np.sin(s), 0.25 * np.sin(2 * s), 2.6 + 0.6 * np.cos(s))


def pose_perspective(t):
    s = 2 * np.pi * t
    return (50 * np.sin(s), 40 * np.sin(1.5 * s), 360 * t,
            0.4 * np.sin(2 * s), 0.2 * np.cos(s), 3.2 - 1.6 * np.cos(s))  # z: 1.6..4.8


def pose_hard(t):
    s = 2 * np.pi * t
    tx = 0.6 * np.sin(1.5 * s)
    # 0.55..0.75: объект уходит вправо за край кадра и полностью исчезает, потом возвращается
    if 0.55 < t < 0.75:
        tx += 3.5 * np.sin(np.pi * (t - 0.55) / 0.2)
    return (25 * np.sin(s), 20 * np.cos(2 * s), 30 * np.sin(s),
            tx, 0.25 * np.sin(3 * s), 2.8 + 0.8 * np.sin(s))


SCENES = {
    "test1_easy":        dict(pose=pose_easy, n=240, hard=False),
    "test2_perspective": dict(pose=pose_perspective, n=300, hard=False),
    "test3_hard":        dict(pose=pose_hard, n=360, hard=True),
}


def generate(name, pose, n, hard, obj, out_dir):
    rng = np.random.default_rng(sum(map(ord, name)))
    bg = make_background(len(name))
    obj_pyr = [obj]
    for _ in range(4):
        obj_pyr.append(cv2.pyrDown(obj_pyr[-1]))

    writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    corners_all = np.zeros((n, 4, 2), np.float32)
    visible = np.zeros(n, np.float32)
    prev_center = None

    for i in range(n):
        if i == 0:
            # первый кадр - крупное изображение объекта во весь кадр
            frame = obj.copy()
            corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            mask = np.ones((H, W), bool)
        else:
            t = (i - 1) / (n - 1)
            # медленное панорамирование фона - камера тоже движется
            ox = int((bg.shape[1] - W) * (0.5 + 0.45 * np.sin(2 * np.pi * t)))
            oy = int((bg.shape[0] - H) * (0.5 + 0.45 * np.cos(2 * np.pi * t)))
            frame = bg[oy:oy + H, ox:ox + W].copy()
            corners = project_corners(*pose(t))
            mask = render_object(frame, obj_pyr, corners)

        occl = np.zeros((H, W), bool)
        if hard and i > 0:
            # перекрытие: серый «палец/рука», проходящая через объект дважды
            for t0, t1 in ((0.12, 0.32), (0.80, 0.98)):
                t = i / n
                if t0 < t < t1:
                    u = (t - t0) / (t1 - t0)
                    cx = int(-200 + u * (W + 400))
                    cy = int(H * 0.55)
                    occ = np.zeros((H, W), np.uint8)
                    cv2.ellipse(occ, (cx, cy), (170, 420), 15, 0, 360, 255, -1)
                    occl = occ > 0
                    frame[occl] = (0.15 * frame[occl] + 0.85 * np.array([110, 140, 190])).astype(np.uint8)
            # смаз в направлении движения объекта
            c = corners.mean(0)
            if prev_center is not None:
                d = (c - prev_center) * 0.8
                frame = motion_blur(frame, d[0], d[1])
            prev_center = c
            # освещение: усиление/ослабление яркости и контраста
            gain = 1.0 + 0.45 * np.sin(2 * np.pi * i / 90)
            frame = np.clip(frame.astype(np.float32) * gain + 15 * np.cos(2 * np.pi * i / 70), 0, 255)
            # шум сенсора и сжатие JPEG
            frame = np.clip(frame + rng.normal(0, 7, frame.shape), 0, 255).astype(np.uint8)
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
            frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        area = cv2.contourArea(corners.astype(np.float32))
        visible[i] = (mask & ~occl).sum() / max(area, 1.0)
        corners_all[i] = corners
        writer.write(frame)

    writer.release()
    np.savez(out_dir / f"{name}_gt.npz", corners=corners_all, visible=visible)
    print(f"{name}: {n} кадров, объект виден (>=20 %) на {(visible >= 0.2).sum()} кадрах")


def main():
    out_dir = ROOT / "videos"
    out_dir.mkdir(exist_ok=True)
    obj = make_object()
    cv2.imwrite(str(out_dir / "object.png"), obj)
    for name, cfg in SCENES.items():
        generate(name, cfg["pose"], cfg["n"], cfg["hard"], obj, out_dir)


if __name__ == "__main__":
    main()
