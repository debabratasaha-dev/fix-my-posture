from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "opencv-python is not installed. Run: python -m pip install opencv-python"
    ) from exc

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is not installed. Run: python -m pip install mediapipe"
    ) from exc

try:
    import winsound
except ImportError:
    winsound = None


NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

REQUIRED_LANDMARKS = (
    NOSE,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
)

DISPLAY_CONNECTIONS = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_EYE, RIGHT_EYE),
    (LEFT_EAR, RIGHT_EAR),
    (NOSE, LEFT_EYE),
    (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR),
    (RIGHT_EYE, RIGHT_EAR),
    (LEFT_EAR, LEFT_SHOULDER),
    (RIGHT_EAR, RIGHT_SHOULDER),
)

LABELS = {
    NOSE: "Nose",
    LEFT_EAR: "LEar",
    RIGHT_EAR: "REar",
    LEFT_SHOULDER: "LSh",
    RIGHT_SHOULDER: "RSh",
}


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass(slots=True)
class FeatureSnapshot:
    shoulder_tilt_deg: float
    face_tilt_deg: float
    neck_angle_deg: float
    head_center_offset_ratio: float
    head_gap_ratio: float
    forward_head_ratio: Optional[float]


@dataclass(slots=True)
class PostureMetrics:
    score: float
    status: str
    color: tuple[int, int, int]
    reasons: list[str]
    snapshot: FeatureSnapshot


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def score_from_error(error: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 100.0
    return clamp(100.0 * (1.0 - (error / tolerance)), 0.0, 100.0)


def point_2d(landmarks, index: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return landmark.x, landmark.y


def midpoint_2d(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def distance_2d(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def landmark_visible(landmark) -> bool:
    visibility = getattr(landmark, "visibility", 1.0)
    presence = getattr(landmark, "presence", 1.0)
    return visibility >= 0.45 and presence >= 0.45


def landmarks_usable(landmarks) -> bool:
    if not landmarks or len(landmarks) <= RIGHT_SHOULDER:
        return False
    return all(landmark_visible(landmarks[index]) for index in REQUIRED_LANDMARKS)


def calculate_forward_head_ratio(world_landmarks) -> Optional[float]:
    if not world_landmarks or len(world_landmarks) <= RIGHT_SHOULDER:
        return None

    left_shoulder = world_landmarks[LEFT_SHOULDER]
    right_shoulder = world_landmarks[RIGHT_SHOULDER]
    left_ear = world_landmarks[LEFT_EAR]
    right_ear = world_landmarks[RIGHT_EAR]

    shoulder_width = math.dist(
        (left_shoulder.x, left_shoulder.y, left_shoulder.z),
        (right_shoulder.x, right_shoulder.y, right_shoulder.z),
    )
    if shoulder_width <= 1e-6:
        return None

    shoulder_mid_z = (left_shoulder.z + right_shoulder.z) / 2.0
    ear_mid_z = (left_ear.z + right_ear.z) / 2.0
    return abs(ear_mid_z - shoulder_mid_z) / shoulder_width


def extract_snapshot(image_landmarks, world_landmarks) -> Optional[FeatureSnapshot]:
    if not landmarks_usable(image_landmarks):
        return None

    nose = point_2d(image_landmarks, NOSE)
    left_eye = point_2d(image_landmarks, LEFT_EYE)
    right_eye = point_2d(image_landmarks, RIGHT_EYE)
    left_shoulder = point_2d(image_landmarks, LEFT_SHOULDER)
    right_shoulder = point_2d(image_landmarks, RIGHT_SHOULDER)

    shoulder_mid = midpoint_2d(left_shoulder, right_shoulder)
    shoulder_width = distance_2d(left_shoulder, right_shoulder)
    if shoulder_width <= 1e-6:
        return None

    shoulder_tilt_deg = abs(
        math.degrees(
            math.atan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0],
            )
        )
    )
    face_tilt_deg = abs(
        math.degrees(
            math.atan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0],
            )
        )
    )

    dx = nose[0] - shoulder_mid[0]
    dy = shoulder_mid[1] - nose[1]
    neck_angle_deg = 90.0 if dy <= 0 else abs(math.degrees(math.atan2(dx, dy)))
    head_center_offset_ratio = abs(dx) / max(shoulder_width / 2.0, 1e-6)
    head_gap_ratio = dy / shoulder_width

    return FeatureSnapshot(
        shoulder_tilt_deg=shoulder_tilt_deg,
        face_tilt_deg=face_tilt_deg,
        neck_angle_deg=neck_angle_deg,
        head_center_offset_ratio=head_center_offset_ratio,
        head_gap_ratio=head_gap_ratio,
        forward_head_ratio=calculate_forward_head_ratio(world_landmarks),
    )


def add_reason(reasons: list[str], message: str) -> None:
    if message not in reasons:
        reasons.append(message)


def compute_generic_metrics(snapshot: FeatureSnapshot) -> PostureMetrics:
    shoulder_score = score_from_error(snapshot.shoulder_tilt_deg, 10.0)
    face_score = score_from_error(snapshot.face_tilt_deg, 8.0)
    neck_score = score_from_error(snapshot.neck_angle_deg, 14.0)
    center_score = score_from_error(snapshot.head_center_offset_ratio, 0.35)
    head_height_score = clamp(100.0 * (snapshot.head_gap_ratio - 0.38) / 0.34, 0.0, 100.0)

    weighted_score = (
        (0.22 * shoulder_score)
        + (0.13 * face_score)
        + (0.27 * neck_score)
        + (0.20 * center_score)
        + (0.18 * head_height_score)
    )

    reasons: list[str] = []
    if snapshot.shoulder_tilt_deg > 8.0:
        add_reason(reasons, "Level your shoulders")
    if snapshot.neck_angle_deg > 10.0:
        add_reason(reasons, "Straighten your neck")
    if snapshot.head_center_offset_ratio > 0.18:
        add_reason(reasons, "Keep your head centered")
    if snapshot.face_tilt_deg > 7.0:
        add_reason(reasons, "Keep your face upright")
    if snapshot.head_gap_ratio < 0.50:
        add_reason(reasons, "Sit taller")

    if weighted_score >= 80.0:
        status = "STRAIGHT"
        color = (70, 200, 70)
    elif weighted_score >= 65.0:
        status = "ADJUST"
        color = (0, 215, 255)
    else:
        status = "BENT"
        color = (0, 0, 255)

    return PostureMetrics(
        score=weighted_score,
        status=status,
        color=color,
        reasons=reasons,
        snapshot=snapshot,
    )


def compute_calibrated_metrics(
    snapshot: FeatureSnapshot,
    baseline: FeatureSnapshot,
) -> PostureMetrics:
    shoulder_score = score_from_error(
        abs(snapshot.shoulder_tilt_deg - baseline.shoulder_tilt_deg), 7.0
    )
    face_score = score_from_error(abs(snapshot.face_tilt_deg - baseline.face_tilt_deg), 6.0)
    neck_score = score_from_error(abs(snapshot.neck_angle_deg - baseline.neck_angle_deg), 10.0)
    center_score = score_from_error(
        abs(snapshot.head_center_offset_ratio - baseline.head_center_offset_ratio), 0.18
    )
    head_height_score = score_from_error(
        abs(snapshot.head_gap_ratio - baseline.head_gap_ratio), 0.16
    )

    scores = [shoulder_score, face_score, neck_score, center_score, head_height_score]
    weights = [0.20, 0.12, 0.28, 0.18, 0.22]

    forward_delta = None
    if (
        snapshot.forward_head_ratio is not None
        and baseline.forward_head_ratio is not None
    ):
        forward_delta = abs(snapshot.forward_head_ratio - baseline.forward_head_ratio)
        scores.append(score_from_error(forward_delta, 0.12))
        weights.append(0.12)

    weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)

    reasons: list[str] = []
    if abs(snapshot.shoulder_tilt_deg - baseline.shoulder_tilt_deg) > 5.0:
        add_reason(reasons, "Level your shoulders")
    if abs(snapshot.neck_angle_deg - baseline.neck_angle_deg) > 7.0:
        add_reason(reasons, "Straighten your neck")
    if abs(snapshot.head_center_offset_ratio - baseline.head_center_offset_ratio) > 0.10:
        add_reason(reasons, "Keep your head centered")
    if snapshot.head_gap_ratio < (baseline.head_gap_ratio - 0.08):
        add_reason(reasons, "Sit taller")
    if forward_delta is not None and snapshot.forward_head_ratio is not None:
        if snapshot.forward_head_ratio > (baseline.forward_head_ratio + 0.08):
            add_reason(reasons, "Pull your head back slightly")

    if weighted_score >= 82.0:
        status = "STRAIGHT"
        color = (70, 200, 70)
    elif weighted_score >= 68.0:
        status = "ADJUST"
        color = (0, 215, 255)
    else:
        status = "BENT"
        color = (0, 0, 255)

    return PostureMetrics(
        score=weighted_score,
        status=status,
        color=color,
        reasons=reasons,
        snapshot=snapshot,
    )


def build_metrics(
    image_landmarks,
    world_landmarks,
    baseline: Optional[FeatureSnapshot],
) -> Optional[PostureMetrics]:
    snapshot = extract_snapshot(image_landmarks, world_landmarks)
    if snapshot is None:
        return None
    if baseline is None:
        return compute_generic_metrics(snapshot)
    return compute_calibrated_metrics(snapshot, baseline)


def pixel_point(landmarks, index: int, frame_width: int, frame_height: int) -> tuple[int, int]:
    point = landmarks[index]
    return int(point.x * frame_width), int(point.y * frame_height)


def draw_visual_guides(frame, image_landmarks, metrics: PostureMetrics, warning_active: bool) -> None:
    height, width = frame.shape[:2]
    points = {
        index: pixel_point(image_landmarks, index, width, height)
        for index in REQUIRED_LANDMARKS
    }
    shoulder_mid = (
        int((points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) / 2),
        int((points[LEFT_SHOULDER][1] + points[RIGHT_SHOULDER][1]) / 2),
    )

    line_color = metrics.color
    accent = (245, 245, 245)

    for start, end in DISPLAY_CONNECTIONS:
        cv2.line(frame, points[start], points[end], accent, 2, cv2.LINE_AA)

    neck_length = max(
        80,
        int(distance_2d(points[LEFT_SHOULDER], points[RIGHT_SHOULDER]) * 1.5),
    )
    reference_top = (shoulder_mid[0], max(10, shoulder_mid[1] - neck_length))
    cv2.line(frame, shoulder_mid, reference_top, (120, 120, 120), 2, cv2.LINE_AA)
    cv2.line(frame, shoulder_mid, points[NOSE], line_color, 3, cv2.LINE_AA)

    for index, label in LABELS.items():
        cv2.circle(frame, points[index], 6, line_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (points[index][0] + 8, points[index][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            accent,
            1,
            cv2.LINE_AA,
        )

    if warning_active:
        cv2.rectangle(frame, (8, 8), (width - 8, height - 8), (0, 0, 255), 5)


def draw_panel(
    frame,
    metrics: Optional[PostureMetrics],
    smoothed_score: Optional[float],
    baseline: Optional[FeatureSnapshot],
    fps: Optional[float],
    notice_text: str,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (430, 245), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    title_color = (235, 235, 235)
    cv2.putText(frame, "FixMyPosture", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, title_color, 2, cv2.LINE_AA)

    mode_text = "Baseline: calibrated" if baseline else "Baseline: generic"
    cv2.putText(frame, mode_text, (35, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if metrics is None or smoothed_score is None:
        cv2.putText(
            frame,
            "No body detected",
            (35, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Keep face and shoulders inside the frame",
            (35, 148),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    else:
        color = metrics.color
        cv2.putText(
            frame,
            f"Straightness score: {smoothed_score:05.1f}%",
            (35, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Posture: {metrics.status}",
            (35, 148),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            color,
            2,
            cv2.LINE_AA,
        )

        bar_left = 35
        bar_top = 170
        bar_width = 340
        filled_width = int(bar_width * clamp(smoothed_score / 100.0, 0.0, 1.0))
        cv2.rectangle(frame, (bar_left, bar_top), (bar_left + bar_width, bar_top + 18), (80, 80, 80), 2)
        cv2.rectangle(frame, (bar_left, bar_top), (bar_left + filled_width, bar_top + 18), color, -1)

        hint_text = metrics.reasons[0] if metrics.reasons else "Great posture. Keep it steady."
        cv2.putText(
            frame,
            f"Hint: {hint_text}",
            (35, 218),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            title_color,
            1,
            cv2.LINE_AA,
        )

    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:04.1f}",
            (320, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    if notice_text:
        cv2.putText(
            frame,
            notice_text,
            (35, 238),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "C: calibrate  R: reset baseline  Q: quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def play_alert_sound(mute: bool) -> None:
    if mute or winsound is None:
        return
    try:
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except RuntimeError:
        pass


def create_landmarker(model_path: Path):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return PoseLandmarker.create_from_options(options)


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(camera_index)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live posture correction app using MediaPipe Pose Landmarker."
    )
    parser.add_argument(
        "--model",
        default="pose_landmarker.task",
        help="Path to the MediaPipe Pose Landmarker task model.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to open.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=68.0,
        help="Alert when the smoothed score stays below this value.",
    )
    parser.add_argument(
        "--alert-delay",
        type=float,
        default=2.0,
        help="Seconds of bad posture before the notification sound plays.",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=4.0,
        help="Minimum seconds between two posture alerts.",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Disable the Windows alert sound.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable selfie-style mirroring in the preview window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        raise SystemExit("Could not open the webcam. Check the camera index and Windows camera permissions.")

    baseline: Optional[FeatureSnapshot] = None
    latest_snapshot: Optional[FeatureSnapshot] = None
    smoothed_score: Optional[float] = None
    bad_posture_since: Optional[float] = None
    last_alert_time = 0.0
    notice_text = "Press C while sitting straight to save a baseline."
    notice_until = time.perf_counter() + 6.0
    last_frame_time: Optional[float] = None
    smoothed_fps: Optional[float] = None

    window_title = "FixMyPosture"

    with create_landmarker(model_path) as landmarker:
        try:
            while True:
                frame_ok, frame = capture.read()
                if not frame_ok:
                    raise RuntimeError("The webcam stopped returning frames.")

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.perf_counter() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                metrics: Optional[PostureMetrics] = None
                warning_active = False

                if result.pose_landmarks:
                    image_landmarks = result.pose_landmarks[0]
                    world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                    metrics = build_metrics(image_landmarks, world_landmarks, baseline)
                    if metrics is not None:
                        latest_snapshot = metrics.snapshot
                        if smoothed_score is None:
                            smoothed_score = metrics.score
                        else:
                            smoothed_score = (0.82 * smoothed_score) + (0.18 * metrics.score)

                        now = time.perf_counter()
                        if smoothed_score < args.score_threshold:
                            if bad_posture_since is None:
                                bad_posture_since = now
                            warning_active = (now - bad_posture_since) >= args.alert_delay
                            if warning_active and (now - last_alert_time) >= args.alert_cooldown:
                                play_alert_sound(args.mute)
                                last_alert_time = now
                                notice_text = "Straighten your posture"
                                notice_until = now + 1.8
                        else:
                            bad_posture_since = None

                        draw_visual_guides(frame, image_landmarks, metrics, warning_active)
                    else:
                        latest_snapshot = None
                        bad_posture_since = None
                else:
                    latest_snapshot = None
                    bad_posture_since = None

                now = time.perf_counter()
                if last_frame_time is not None:
                    instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
                    if smoothed_fps is None:
                        smoothed_fps = instant_fps
                    else:
                        smoothed_fps = (0.85 * smoothed_fps) + (0.15 * instant_fps)
                last_frame_time = now

                active_notice = notice_text if now <= notice_until else ""
                draw_panel(frame, metrics, smoothed_score, baseline, smoothed_fps, active_notice)

                if metrics and warning_active:
                    cv2.putText(
                        frame,
                        "STRAIGHTEN YOUR POSTURE",
                        (40, frame.shape[0] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    if latest_snapshot is not None:
                        baseline = latest_snapshot
                        notice_text = "Baseline saved. Monitoring against your straight posture."
                        notice_until = time.perf_counter() + 2.5
                    else:
                        notice_text = "No valid pose in frame to calibrate."
                        notice_until = time.perf_counter() + 1.8
                if key == ord("r"):
                    baseline = None
                    notice_text = "Baseline cleared. Using generic posture rules."
                    notice_until = time.perf_counter() + 2.0
        finally:
            capture.release()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
