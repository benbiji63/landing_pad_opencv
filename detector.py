#!/usr/bin/env python3
"""Color-independent-ish circular landing-pad detector and nadir position estimator.

Position output assumes a downward-facing, approximately level camera, with the
image top aligned with drone-forward and image right aligned with drone-right.
Altitude must be height from camera optical center to the pad plane, in metres.
"""

import argparse
import json
import math
import time

import cv2
import numpy as np


def detect_pad(frame, args):
    """Find circle candidates using Hough circles plus contour geometry.

    Grayscale/edge geometry avoids dependence on one specific pad color.
    The candidate score is heuristic, not a calibrated probability.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = max(3, int(args.blur))
    if blur % 2 == 0:
        blur += 1
    gray = cv2.GaussianBlur(gray, (blur, blur), 1.5)
    h, w = gray.shape
    max_radius = args.max_radius if args.max_radius > 0 else min(h, w) // 2
    edges = cv2.Canny(gray, max(10, args.param1 * 0.35), args.param1)
    candidates = []

    def add_candidate(x, y, radius, shape_score=0.5):
        if radius < args.min_radius or radius > max_radius:
            return
        angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        xs = np.clip(np.round(x + (radius - 1) * np.cos(angles)).astype(int), 0, w - 1)
        ys = np.clip(np.round(y + (radius - 1) * np.sin(angles)).astype(int), 0, h - 1)
        edge_support = float(np.mean(edges[ys, xs] > 0))
        center_dist = np.linalg.norm(np.array([x, y]) - np.array([w / 2.0, h / 2.0]))
        centrality = 1.0 - min(center_dist / np.linalg.norm([w / 2.0, h / 2.0]), 1.0)
        score = 0.65 * edge_support + 0.25 * shape_score + 0.10 * centrality
        candidates.append(
            (score, (int(round(x)), int(round(y)), int(round(radius))), edge_support)
        )

    # Candidate source 1: Hough circles. Useful for partial circular boundaries.
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=args.dp,
        minDist=args.min_distance,
        param1=args.param1,
        param2=args.param2,
        minRadius=args.min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            add_candidate(x, y, r)

    # Candidate source 2: contours. Helps with outlined/ring-shaped pads and
    # different contrast patterns; reject highly elongated contours.
    contour_edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(
        contour_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < np.pi * args.min_radius * args.min_radius * 0.20:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < args.min_circularity:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(bh, 1)
        if not 0.65 <= aspect <= 1.55:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        add_candidate(cx, cy, radius, circularity)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, circle, edge_support = candidates[0]
    if edge_support < args.min_edge_support:
        return None
    return {"circle": circle, "score": float(score), "edge_support": edge_support}


def estimate_relative_position(
    frame_shape,
    circle,
    altitude_m,
    hfov_deg,
    camera_forward_offset_m=0.0,
    camera_right_offset_m=0.0,
):
    """Estimate pad center relative to drone in FRD horizontal axes.

    Positive forward = in front of drone; positive right = right of drone.
    Assumes nadir/level camera, small lens distortion after calibration,
    and altitude measured from camera optical center to the pad plane.
    """
    h, w = frame_shape[:2]
    x, y, radius = circle
    hfov = math.radians(hfov_deg)
    fx = (w / 2.0) / math.tan(hfov / 2.0)
    # Approximate square pixels. Replace with calibrated fx/fy for accuracy.
    fy = fx
    cx, cy = w / 2.0, h / 2.0

    # With a downward-facing camera, image top corresponds to drone-forward.
    right_m = (x - cx) * altitude_m / fx + camera_right_offset_m
    forward_m = (cy - y) * altitude_m / fy + camera_forward_offset_m
    return {
        "relative_forward_m": round(float(forward_m), 3),
        "relative_right_m": round(float(right_m), 3),
        "height_to_pad_m": round(float(altitude_m), 3),
        "position_method": "nadir_camera_altitude_projection",
        "position_valid_for_navigation": False,
    }


def draw_overlay(frame, detection, position):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), (255, 180, 0), cv2.MARKER_CROSS, 20, 1)
    if detection:
        x, y, r = detection["circle"]
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)
        cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.line(out, (w // 2, h // 2), (x, y), (255, 0, 255), 1)
        cv2.putText(
            out,
            f"score={detection['score']:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        if position:
            txt = (
                f"F {position['relative_forward_m']:+.2f} m  "
                f"R {position['relative_right_m']:+.2f} m  "
                f"H {position['height_to_pad_m']:.2f} m"
            )
            cv2.putText(
                out, txt, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )
    else:
        cv2.putText(
            out,
            "PAD NOT DETECTED",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return out


def process_frame(frame, args):
    detection = detect_pad(frame, args)
    result = {
        "timestamp_monotonic_s": round(time.monotonic(), 3),
        "detected": detection is not None,
    }
    position = None
    if detection:
        x, y, r = detection["circle"]
        h, w = frame.shape[:2]
        result.update(
            {
                "center_px": {"x": x, "y": y},
                "radius_px": r,
                "detector_score": round(detection["score"], 3),
                "edge_support": round(detection["edge_support"], 3),
                "normalized_error": {
                    "x": round((x - w / 2) / (w / 2), 4),
                    "y": round((y - h / 2) / (h / 2), 4),
                },
            }
        )
        if args.altitude_m is not None:
            position = estimate_relative_position(
                frame.shape,
                detection["circle"],
                args.altitude_m,
                args.hfov_deg,
                args.camera_forward_offset_m,
                args.camera_right_offset_m,
            )
            result["relative_position"] = position
    return result, draw_overlay(frame, detection, position)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a still image")
    src.add_argument("--video", help="Path to a video file")
    src.add_argument("--camera", type=int, help="V4L2 camera index, usually 0")
    p.add_argument("--altitude-m", type=float, help="Camera-to-pad height in metres")
    p.add_argument(
        "--hfov-deg",
        type=float,
        default=120.0,
        help="Approximate horizontal field of view; calibrate for accuracy",
    )
    p.add_argument("--camera-forward-offset-m", type=float, default=0.0)
    p.add_argument("--camera-right-offset-m", type=float, default=0.0)
    p.add_argument("--dp", type=float, default=1.2)
    p.add_argument("--min-distance", type=float, default=100)
    p.add_argument("--param1", type=float, default=120)
    p.add_argument(
        "--param2",
        type=float,
        default=30,
        help="Lower detects more candidates but increases false positives",
    )
    p.add_argument("--min-radius", type=int, default=15)
    p.add_argument(
        "--max-radius",
        type=int,
        default=0,
        help="0 uses half the shorter frame dimension",
    )
    p.add_argument("--blur", type=int, default=7)
    p.add_argument("--min-edge-support", type=float, default=0.10)
    p.add_argument("--min-circularity", type=float, default=0.62)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-output", help="Save annotated image (image input only)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.altitude_m is not None and args.altitude_m <= 0:
        raise SystemExit("--altitude-m must be greater than zero")
    if not 1 < args.hfov_deg < 179:
        raise SystemExit("--hfov-deg must be between 1 and 179")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        result, annotated = process_frame(frame, args)
        print(json.dumps(result), flush=True)
        if args.save_output and not cv2.imwrite(args.save_output, annotated):
            raise SystemExit(f"Could not save output: {args.save_output}")
        if not args.no_display:
            cv2.imshow("Landing pad detector (press q to quit)", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(
        source, cv2.CAP_V4L2 if args.camera is not None else cv2.CAP_ANY
    )
    if not cap.isOpened():
        raise SystemExit(
            "Could not open source. Check camera index/path and permissions."
        )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    break
                print(
                    json.dumps({"detected": False, "error": "frame_read_failed"}),
                    flush=True,
                )
                time.sleep(0.1)
                continue
            result, annotated = process_frame(frame, args)
            print(json.dumps(result), flush=True)
            if not args.no_display:
                cv2.imshow("Landing pad detector (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
