
# Circular landing-pad detector (Linux prototype)

A color-independent-ish OpenCV prototype intended for laptop testing first, then Raspberry Pi 5 deployment. It uses grayscale Hough-circle detection and reports the circle center. If camera-to-pad height is supplied, it estimates the pad center's relative horizontal offset under a nadir/level-camera assumption.

## Install (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip v4l-utils ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate test images

```bash
python generate_test_images.py
```

## Test a still image

```bash
python detector.py --image test_images/filled_circle.png --save-output result.png
python detector.py --image test_images/ring.png --param2 25 --save-output ring-result.png
```

Use `--no-display` for headless execution.

## Webcam or video

```bash
v4l2-ctl --list-devices
python detector.py --camera 0
python detector.py --video path/to/video.mp4
```

## Estimate position in metres

```bash
python detector.py --video flight_test.mp4 --altitude-m 2.0 --hfov-deg 120
```

`--altitude-m` must be the vertical distance from the camera optical center to the landing-pad plane, not necessarily the flight controller's reported altitude above sea level. `--hfov-deg 120` is only a placeholder. Calibrate the actual lens/camera; action-camera wide-angle distortion can make the simple model inaccurate.

Coordinate convention: `relative_forward_m > 0` means pad center is forward of the drone; `relative_right_m > 0` means it is right of the drone. This assumes a downward-facing, level camera, image top aligned with drone-forward, image right aligned with drone-right, and camera offset corrected with the optional offset arguments.

## USB camera caveat

A RunCam 5 USB port may be intended for file transfer/charging rather than live UVC webcam streaming. Verify that Linux sees a live capture device before relying on USB video. Run `v4l2-ctl --list-devices` and inspect `/dev/video*`. If no compatible live stream appears, use a supported video-output/capture path; OpenCV cannot read video that the OS does not expose.

## Important limits

- The detector is not a trained model and is not a certified landing system.
- Other circles can be mistaken for the pad. Add temporal tracking, expected-size constraints and a calibrated confidence gate before flight use.
- Position estimation is approximate and assumes a nadir/level camera. If the drone tilts, use attitude data and ray/ground-plane intersection, or estimate pad pose with known geometry and calibrated intrinsics.
- For reliable metric position, calibrate camera intrinsics/distortion and camera mounting extrinsics. Never feed these estimates directly into flight control until validated in simulation and controlled tests with an independent abort/failsafe.
