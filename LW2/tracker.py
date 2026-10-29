"""Трекинг плоского объекта на видео по ключевым точкам.

Схема (tracking-by-detection):
  1. На первом кадре вырезается эталон объекта (весь кадр или заданный ROI),
     на нём один раз находятся ключевые точки и дескрипторы.
  2. На каждом следующем кадре находятся ключевые точки, дескрипторы
     сопоставляются с эталоном (kNN, k=2) и фильтруются ratio-тестом Лоу.
  3. По соответствиям методом RANSAC (MAGSAC++) оценивается гомография
     эталон -> кадр, ею переносятся углы эталона - получается рамка.
  4. Гомография проверяется на правдоподобие (число и доля инлаеров,
     выпуклость четырёхугольника, площадь, отсутствие отражения);
     если проверка не пройдена, считается, что объекта в кадре нет.

Каждый кадр сравнивается с исходным эталоном, а не с предыдущим кадром,
поэтому ошибка не накапливается (нет дрейфа) и после потери объекта
(перекрытие, выход из кадра) он находится заново без дополнительной логики.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

DETECTORS = ("sift", "orb", "akaze", "brisk")


def create_detector(name: str):
    """Возвращает (детектор+дескриптор, флаг бинарности дескриптора)."""
    name = name.lower()
    if name == "sift":
        return cv2.SIFT_create(nfeatures=4000), False
    if name == "orb":
        return cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8, fastThreshold=10), True
    if name == "akaze":
        return cv2.AKAZE_create(threshold=5e-4), True
    if name == "brisk":
        return cv2.BRISK_create(thresh=20), True
    raise ValueError(f"Неизвестный детектор '{name}', доступны: {', '.join(DETECTORS)}")


def create_matcher(binary: bool):
    if binary:
        # LSH-индекс FLANN для бинарных дескрипторов (расстояние Хэмминга)
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    else:
        # KD-деревья для вещественных дескрипторов (SIFT, L2)
        index_params = dict(algorithm=1, trees=5)
    return cv2.FlannBasedMatcher(index_params, dict(checks=64))


@dataclass
class TrackResult:
    found: bool
    corners: np.ndarray | None = None      # 4x2, координаты углов объекта в исходном кадре
    homography: np.ndarray | None = None   # 3x3, эталон -> исходный кадр
    n_keypoints: int = 0                   # ключевых точек в кадре
    n_matches: int = 0                     # соответствий после ratio-теста
    n_inliers: int = 0                     # инлаеров RANSAC
    inlier_points: np.ndarray | None = field(default=None, repr=False)  # Nx2, в координатах кадра
    source: str = ""                       # detect - по эталону, flow - оптическим потоком,
                                           # predict - объект потерян, положение предсказано
    reason: str = ""                       # почему объект не принят (для отладки)


class KeypointTracker:
    def __init__(self, detector: str = "sift", ratio: float = 0.75, min_inliers: int = 15,
                 min_inlier_ratio: float = 0.25, ransac_thresh: float = 4.0, max_side: int = 960,
                 use_flow: bool = True, min_flow_points: int = 12, max_predict: int = 15):
        """
        detector         - sift | orb | akaze | brisk
        ratio            - порог ratio-теста Лоу (d1 / d2 < ratio)
        min_inliers      - минимум инлаеров, чтобы считать объект найденным
        min_inlier_ratio - минимальная доля инлаеров среди соответствий
        ransac_thresh    - порог ошибки перепроецирования для RANSAC, px (в масштабе обработки)
        max_side         - кадр уменьшается до этого размера по большей стороне перед обработкой
        use_flow         - если детектирование не удалось, сопровождать объект оптическим потоком
        min_flow_points  - минимум точек-инлаеров при сопровождении потоком
        max_predict      - сколько кадров подряд можно предсказывать положение потерянного
                           объекта (0 - не предсказывать)
        """
        self.detector_name = detector.lower()
        self.detector, binary = create_detector(detector)
        self.matcher = create_matcher(binary)
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_thresh = ransac_thresh
        self.max_side = max_side
        self.use_flow = use_flow
        self.min_flow_points = min_flow_points
        self.max_predict = max_predict

        self.tpl_kp = None
        self.tpl_pts = None       # Nx2, координаты точек эталона (в масштабе обработки)
        self.tpl_desc = None
        self.tpl_corners = None   # 4x2, углы эталона (в масштабе обработки)
        self.tpl_area = 0.0

        self.prev_gray = None
        self.flow_pts = None      # Nx2, текущие положения сопровождаемых точек
        self.flow_tpl = None      # Nx2, координаты тех же точек на эталоне
        self.flow_n0 = 0
        self.last_corners = None  # 4x2, последнее найденное положение углов
        self.velocity = None      # 4x2, скорость углов, px/кадр
        self.n_lost = 0           # сколько кадров подряд объект не найден

    # ------------------------------------------------------------------ utils
    def _prepare(self, frame: np.ndarray):
        """Перевод в оттенки серого и уменьшение. Возвращает (gray, scale)."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = min(1.0, self.max_side / max(h, w)) if self.max_side else 1.0
        if scale < 1.0:
            gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        return gray, scale

    # ------------------------------------------------------------------- init
    def init(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> int:
        """Запоминает эталон. roi = (x, y, w, h) в координатах кадра; None - весь кадр.
        Возвращает количество ключевых точек эталона."""
        gray, scale = self._prepare(frame)
        if roi is None:
            x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]
        else:
            x, y, w, h = (int(round(v * scale)) for v in roi)

        mask = np.zeros_like(gray)
        mask[y:y + h, x:x + w] = 255
        self.tpl_kp, self.tpl_desc = self.detector.detectAndCompute(gray, mask)
        if self.tpl_desc is None or len(self.tpl_kp) < self.min_inliers:
            raise RuntimeError("На эталоне слишком мало ключевых точек: выберите более текстурный объект")

        self.tpl_pts = np.float32([kp.pt for kp in self.tpl_kp])
        self.tpl_corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        self.tpl_area = float(w * h)
        self.matcher.clear()
        self.matcher.add([self.tpl_desc])
        self.matcher.train()
        self.prev_gray, self.flow_pts, self.flow_tpl = gray, None, None
        self.last_corners, self.velocity, self.n_lost = None, None, 0
        return len(self.tpl_kp)

    # ----------------------------------------------------------------- update
    def update(self, frame: np.ndarray) -> TrackResult:
        """Ищет объект на кадре. Сначала - детектирование по эталону; если оно не удалось,
        а на предыдущем кадре объект был найден - сопровождение точек оптическим потоком."""
        if self.tpl_desc is None:
            raise RuntimeError("Сначала вызовите init()")

        gray, scale = self._prepare(frame)
        res, H = self._detect(gray)
        if res.found:
            res.source = "detect"
            if self.use_flow:
                self._seed_flow(gray, H)
        elif self.use_flow and self.flow_pts is not None:
            fres, H = self._track_flow(gray)
            if fres.found:
                fres.n_keypoints, fres.n_matches = res.n_keypoints, res.n_matches
                res = fres
            else:
                self.flow_pts = self.flow_tpl = None
        if not res.found:
            res, H = self._predict(gray.shape, res)
        else:
            self._update_motion(res.corners)
        self.prev_gray = gray

        if res.found:
            # перевод из масштаба обработки в координаты исходного кадра
            res.corners = res.corners / scale
            res.homography = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ H
            res.inlier_points = res.inlier_points / scale
        return res

    # ------------------------------------------------- детектирование по эталону
    def _detect(self, gray: np.ndarray):
        kp, desc = self.detector.detectAndCompute(gray, None)
        n_kp = 0 if kp is None else len(kp)
        if desc is None or n_kp < self.min_inliers:
            return TrackResult(False, n_keypoints=n_kp, reason="мало ключевых точек"), None

        # kNN-сопоставление кадр -> эталон и ratio-тест Лоу
        knn = self.matcher.knnMatch(desc, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < self.ratio * m[1].distance]
        if len(good) < self.min_inliers:
            return TrackResult(False, n_keypoints=n_kp, n_matches=len(good), reason="мало соответствий"), None

        src = self.tpl_pts[[m.trainIdx for m in good]]
        dst = np.float32([kp[m.queryIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, self.ransac_thresh,
                                    maxIters=5000, confidence=0.999)
        if H is None:
            return TrackResult(False, n_keypoints=n_kp, n_matches=len(good), reason="гомография не найдена"), None

        inl = inl.ravel().astype(bool)
        n_inl = int(inl.sum())
        res = TrackResult(False, n_keypoints=n_kp, n_matches=len(good), n_inliers=n_inl)

        corners = cv2.perspectiveTransform(self.tpl_corners[None], H)[0]
        res.reason = self._check(H, corners, n_inl, len(good), self.min_inliers)
        if res.reason:
            return res, None
        res.found, res.corners, res.inlier_points = True, corners, dst[inl]
        return res, H

    # ------------------------------------------ сопровождение оптическим потоком
    def _seed_flow(self, gray: np.ndarray, H: np.ndarray):
        """Выбирает хорошо отслеживаемые углы (Shi-Tomasi) внутри рамки объекта и
        запоминает их координаты на эталоне (через H^-1)."""
        corners = cv2.perspectiveTransform(self.tpl_corners[None], H)[0]
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, corners.round().astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))  # не брать точки на самой границе
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=4, mask=mask)
        if pts is None or len(pts) < self.min_flow_points:
            self.flow_pts = self.flow_tpl = None
            return
        self.flow_pts = pts.reshape(-1, 2).astype(np.float32)
        self.flow_tpl = cv2.perspectiveTransform(self.flow_pts[None], np.linalg.inv(H))[0]
        self.flow_n0 = len(self.flow_pts)

    def _track_flow(self, gray: np.ndarray):
        """Переносит точки с предыдущего кадра пирамидальным методом Лукаса-Канаде
        и по соответствиям «эталон -> текущие положения точек» строит гомографию."""
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p0 = self.flow_pts.reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **lk)
        # проверка «вперёд-назад»: точка, перенесённая обратно, должна вернуться на место
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **lk)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        res = TrackResult(False, source="flow")
        if ok.sum() < self.min_flow_points:
            res.reason = "поток: мало точек"
            return res, None

        tpl, cur = self.flow_tpl[ok], p1.reshape(-1, 2)[ok]
        H, inl = cv2.findHomography(tpl, cur, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.999)
        if H is None:
            res.reason = "поток: гомография не найдена"
            return res, None
        inl = inl.ravel().astype(bool)
        res.n_inliers = int(inl.sum())
        corners = cv2.perspectiveTransform(self.tpl_corners[None], H)[0]
        # доля инлаеров для потока должна быть высокой: точки уже принадлежат объекту
        res.reason = self._check(H, corners, res.n_inliers, len(tpl), self.min_flow_points, min_ratio=0.5)
        if res.reason:
            return res, None

        res.found, res.corners, res.inlier_points = True, corners, cur[inl]
        self.flow_tpl, self.flow_pts = tpl[inl], cur[inl]
        # точки постепенно теряются (уходят за край, закрываются) - добираем новые
        if len(self.flow_pts) < max(0.5 * self.flow_n0, 2 * self.min_flow_points):
            self._seed_flow(gray, H)
        return res, H

    # ------------------------------------------------------ предсказание положения
    def _update_motion(self, corners: np.ndarray):
        """Запоминает положение и скорость углов (по двум последовательным измерениям)."""
        if self.last_corners is not None and self.n_lost == 0:
            self.velocity = corners - self.last_corners
        else:
            self.velocity = np.zeros_like(corners)
        self.last_corners, self.n_lost = corners, 0

    def _predict(self, shape, res: TrackResult, damping: float = 0.8):
        """Объект не найден (сильный смаз, перекрытие): рамка продолжает движение
        с последней скоростью, скорость затухает. Не дольше max_predict кадров и
        только пока предсказанная рамка в основном остаётся в кадре."""
        if self.last_corners is None or self.n_lost >= self.max_predict:
            return res, None
        self.velocity = self.velocity * damping
        corners = self.last_corners + self.velocity
        quad = corners.astype(np.float32)
        h, w = shape
        frame_rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        inside, _ = cv2.intersectConvexConvex(quad, frame_rect)
        if inside < 0.5 * cv2.contourArea(quad):  # объект, скорее всего, ушёл из кадра
            self.last_corners = None
            return res, None
        self.last_corners = corners
        self.n_lost += 1
        H = cv2.getPerspectiveTransform(self.tpl_corners, quad)
        pred = TrackResult(True, corners=corners, inlier_points=np.empty((0, 2), np.float32),
                           n_keypoints=res.n_keypoints, n_matches=res.n_matches,
                           source="predict", reason=res.reason)
        return pred, H

    # ------------------------------------------------------------- проверки
    def _check(self, H, corners, n_inl, n_matches, min_inliers, min_ratio=None) -> str:
        """Проверка правдоподобия найденной гомографии. Пустая строка - всё в порядке."""
        if n_inl < min_inliers:
            return "мало инлаеров"
        if n_inl < (min_ratio or self.min_inlier_ratio) * n_matches:
            return "низкая доля инлаеров"
        # однородная координата w углов должна быть > 0, иначе точка «за камерой»
        w = H[2] @ np.vstack([self.tpl_corners.T, np.ones(4)])
        if np.any(w <= 0):
            return "углы за камерой"
        if not cv2.isContourConvex(corners.reshape(-1, 1, 2).astype(np.float32)):
            return "невыпуклая рамка"
        # обход углов должен сохранить направление, иначе это зеркальное отражение,
        # а плоский объект при съёмке спереди так выглядеть не может
        area = cv2.contourArea(corners.astype(np.float32), oriented=True)
        if np.sign(area) != np.sign(cv2.contourArea(self.tpl_corners, oriented=True)):
            return "отражение"
        area = abs(area)
        if not (1e-3 * self.tpl_area < area < 50 * self.tpl_area):
            return "неправдоподобная площадь"
        # слишком сильная перспектива (противоположные стороны отличаются > 6 раз)
        sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        if max(sides[0], sides[2]) > 6 * min(sides[0], sides[2]) or \
           max(sides[1], sides[3]) > 6 * min(sides[1], sides[3]):
            return "вырожденная перспектива"
        return ""


# ---------------------------------------------------------------- рисование
def draw_result(frame: np.ndarray, res: TrackResult, label: str = "Object",
                color=(0, 255, 0), draw_points: bool = False) -> np.ndarray:
    """Рисует рамку объекта с подписью (на копии кадра)."""
    out = frame.copy()
    th = max(2, round(max(out.shape[:2]) / 400))
    if not res.found:
        return out
    if res.source == "predict":
        color = (0, 215, 255)  # предсказанная рамка - жёлтая

    if draw_points and res.inlier_points is not None:
        for p in res.inlier_points.astype(int):
            cv2.circle(out, tuple(p), th + 1, (0, 200, 255), -1, cv2.LINE_AA)

    quad = res.corners.reshape(-1, 1, 2).round().astype(np.int32)
    cv2.polylines(out, [quad], True, color, th, cv2.LINE_AA)
    # отметка "верхней" стороны эталона, чтобы была видна ориентация объекта
    cv2.line(out, tuple(quad[0, 0]), tuple(quad[1, 0]), (0, 0, 255), th, cv2.LINE_AA)

    # подпись у самого верхнего угла рамки
    text = f"{label} (predicted)" if res.source == "predict" else f"{label} ({res.n_inliers} pts)"
    fs = th * 0.35
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    x, y = res.corners[np.argmin(res.corners[:, 1])]
    x = int(np.clip(x - tw / 2, 0, out.shape[1] - tw - 4))
    y = int(np.clip(y - 2 * th, tht + base + 4, out.shape[0] - 2))
    cv2.rectangle(out, (x, y - tht - base - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(out, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)
    return out


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Служебная информация в левом верхнем углу (кадр, детектор, fps)."""
    th = max(1, round(max(frame.shape[:2]) / 700))
    fs = th * 0.5
    y = 0
    for s in lines:
        (tw, tht), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y += tht + base + 6
        cv2.rectangle(frame, (4, y - tht - base - 2), (tw + 12, y + 2), (0, 0, 0), -1)
        cv2.putText(frame, s, (8, y - base), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return frame
