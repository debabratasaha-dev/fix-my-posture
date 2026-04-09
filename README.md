# FixMyPosture

Developers always face back pain due to bad posture, so I vibe-coded a Posture correction app in Python using MediaPipe Pose Landmarker and local ML model `pose_landmarker.task` to fix it.

## What it does

- Opens your webcam and runs live posture detection.
- Uses face and shoulder landmarks to estimate straightness.
- Shows the camera view with posture guides and landmark labels.
- Displays a live straightness score out of 100.
- Warns you with an on-screen message and Windows sound if you stay bent for too long.
- Lets you press `C` while sitting straight to save your own baseline posture.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python posture_monitor.py --model pose_landmarker.task
```

## Controls

- `C` saves your current straight posture as the personal baseline.
- `R` clears the saved baseline and goes back to generic posture rules.
- `Q` quits the app.

## Notes

- Keep your face and both shoulders visible to the webcam for the best result.
- Calibration improves forward-head and slouch detection because the app can compare you against your own straight posture.
- If you have multiple cameras, change the webcam with `--camera 1`, `--camera 2`, and so on.
- Use `--no-mirror` if you do not want the selfie-style preview.
