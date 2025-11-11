import cv2
import numpy as np
import time
import os
from datetime import datetime

# --- User-tweakable defaults ---
PREFERRED_INDEX = 0
REQUESTED_RES = (1280, 720)       # initial target; the code will auto-fallback
SHOW_DEBUG = False

# --- Hough/processing params ---
HOUGH_DP = 1.2
HOUGH_MIN_DIST_FRAC = 0.8
HOUGH_CANNY = 120
HOUGH_ACC = 18
HOUGH_MIN_RAD_FRAC = 0.05
HOUGH_MAX_RAD_FRAC = 0.5

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Could not load Haar cascades from OpenCV. Reinstall opencv-python if needed.")

# Calibration state
pixels_per_mm = None
awaiting_calibration_clicks = False
calibration_points = []

# ---------------- Camera open/reprobe helpers ----------------

BACKENDS = [
    (cv2.CAP_MSMF, "MSMF"),
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_VFW, "VFW"),
    (cv2.CAP_ANY, "ANY"),
]
INDICES = [PREFERRED_INDEX] + [i for i in range(6) if i != PREFERRED_INDEX]
RESOLUTIONS = [
    REQUESTED_RES,
    (960, 540),
    (640, 480),
    (1920, 1080),  # try FHD last (some cams only stream MJPG at FHD)
]
FOURCCS = [
    ("MJPG", "MJPG"),
    ("YUY2", "YUY2"),   # uncompressed; large bandwidth but often reliable
    ("H264", "H264"),   # some cams expose hardware H.264
    ("", "DEFAULT"),
]

def _set_fourcc(cap, fourcc_str):
    if not fourcc_str:
        return
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

def _try_read(cap, tries=8, delay=0.05):
    """Try to fetch a frame several times (allowing autoexposure to settle)."""
    ok = False
    frame = None
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(delay)
    return False, None

def open_verified_camera():
    print("[Camera] Probing backends/indices/resolutions/codecs...")
    for backend, bname in BACKENDS:
        for idx in INDICES:
            for (w, h) in RESOLUTIONS:
                cap = cv2.VideoCapture(idx, backend)
                opened = cap.isOpened()
                print(f"[Camera] Try backend={bname} index={idx} -> opened={opened}")
                if not opened:
                    cap.release()
                    continue

                # Request resolution & a reasonable FPS
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, 30)

                # Try several FOURCCs
                for fourcc, label in FOURCCS:
                    _set_fourcc(cap, fourcc)
                    time.sleep(0.12)  # let pipeline settle
                    ok, frame = _try_read(cap, tries=10, delay=0.06)
                    if ok:
                        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        fourcc_now = int(cap.get(cv2.CAP_PROP_FOURCC))
                        fourcc_now = "".join([chr((fourcc_now >> 8*i) & 0xFF) for i in range(4)])
                        print(f"[Camera] ✅ backend={bname} index={idx} res={aw}x{ah} fps≈{fps:.1f} fourcc={fourcc_now or label}")
                        return cap, (aw, ah), bname, idx, fourcc_now or label
                    else:
                        print(f"[Camera]   fourcc={label} failed to deliver frames; trying next codec...")
                # None of the codecs worked at this resolution
                cap.release()
    return None, None, None, None, None

def re_probe_camera():
    print("[Camera] Re-probing due to repeated frame failures...")
    return open_verified_camera()

# ---------------- UI helpers & pupil detection ----------------

def on_mouse(event, x, y, flags, param):
    global awaiting_calibration_clicks, calibration_points, pixels_per_mm
    if awaiting_calibration_clicks and event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        if len(calibration_points) == 2:
            (x1, y1), (x2, y2) = calibration_points
            px_dist = np.hypot(x2 - x1, y2 - y1)
            print(f"[Calibration] Pixel distance: {px_dist:.2f}px")
            try:
                mm = float(input("[Calibration] Enter real distance in millimeters (mm): ").strip())
                if mm <= 0:
                    raise ValueError
                pixels_per_mm = px_dist / mm
                print(f"[Calibration] Scale set: {pixels_per_mm:.3f} px/mm  (1 mm = {1.0/pixels_per_mm:.3f} px)")
            except Exception:
                print("[Calibration] Invalid value. Calibration aborted.")
                pixels_per_mm = None
            calibration_points.clear()
            awaiting_calibration_clicks = False

def annotate_text(img, text, x=8, y=28, scale=0.7, color=(255,255,255)):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def find_pupil_in_eye(eye_gray):
    h, w = eye_gray.shape[:2]
    norm = clahe.apply(eye_gray)
    blur = cv2.medianBlur(norm, 5)

    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 5)
    kernel = np.ones((3,3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    pupil = None
    min_dist = max(6, int(HOUGH_MIN_DIST_FRAC * h))
    min_r = max(3, int(HOUGH_MIN_RAD_FRAC * h))
    max_r = max(min_r + 1, int(HOUGH_MAX_RAD_FRAC * h))

    edges = cv2.Canny(blur, 40, 120)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
                               minDist=min_dist,
                               param1=HOUGH_CANNY, param2=HOUGH_ACC,
                               minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        best_score = 1e9; best = None
        for (cx, cy, r) in circles:
            r = int(r)
            y1 = max(0, cy - r); y2 = min(h, cy + r)
            x1 = max(0, cx - r); x2 = min(w, cx + r)
            roi = blur[y1:y2, x1:x2]
            if roi.size == 0: continue
            mask = np.zeros_like(roi, dtype=np.uint8)
            cv2.circle(mask, (r, r), r, 255, -1)
            dark_score = cv2.mean(roi, mask=mask)[0]
            if dark_score < best_score:
                best_score = dark_score; best = (int(cx), int(cy), int(r))
        if best is not None:
            pupil = best

    if pupil is None:
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:3]:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                if r >= min_r and r <= max_r:
                    pupil = (int(cx), int(cy), int(r)); break

    debug = {"norm": norm, "thr": thr, "edges": edges}
    return pupil, debug

# ---------------- Main ----------------

def main():
    global awaiting_calibration_clicks, calibration_points, pixels_per_mm

    cap, (W, H), backend_name, cam_idx, fourcc_used = open_verified_camera()
    if cap is None:
        raise RuntimeError("Could not obtain frames from any camera/backend/resolution/codec.\n"
                           "Check Windows Camera privacy settings and close other apps using the camera.")

    print(f"[Camera] Using backend={backend_name} index={cam_idx} res={W}x{H} fourcc={fourcc_used}")

    cv2.namedWindow("Pupil Diameter")
    cv2.setMouseCallback("Pupil Diameter", on_mouse)

    face_fail_count = 0
    frame_fail_count = 0
    FAIL_REPROBE_THRESHOLD = 25  # consecutive failed reads before reprobe

    last_save = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            frame_fail_count += 1
            print("[Camera] Frame not received. Trying to recover...")
            time.sleep(0.05)
            if frame_fail_count >= FAIL_REPROBE_THRESHOLD:
                cap.release()
                cap, (W, H), backend_name, cam_idx, fourcc_used = re_probe_camera()
                if cap is None:
                    print("[Camera] Re-probe failed. Exiting.")
                    break
                frame_fail_count = 0
            continue
        else:
            frame_fail_count = 0

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_disp = frame.copy()

        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(120,120))
        pupils_found = []

        for (fx, fy, fw, fh) in faces[:1]:
            cv2.rectangle(frame_disp, (fx, fy), (fx+fw, fy+fh), (120,120,120), 1)
            face_roi_gray = frame_gray[fy:fy+fh, fx:fx+fw]
            face_roi_color = frame_disp[fy:fy+fh, fx:fx+fw]

            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(40,40))
            if len(eyes) == 0:
                fh2 = fh // 2
                upper = face_roi_gray[0:fh2, :]
                h_u, w_u = upper.shape[:2]
                eyes = [(0,0,w_u//2,h_u), (w_u//2,0,w_u//2,h_u)]

            for (ex, ey, ew, eh) in eyes[:2]:
                eye_gray = face_roi_gray[ey:ey+eh, ex:ex+ew]
                eye_color = face_roi_color[ey:ey+eh, ex:ex+ew]
                pupil, debug = find_pupil_in_eye(eye_gray)
                if pupil is not None:
                    cx, cy, r = pupil
                    cv2.circle(eye_color, (cx, cy), r, (0,255,0), 2)
                    cv2.circle(eye_color, (cx, cy), 2, (0,255,0), -1)
                    diameter_px = 2.0 * r
                    if pixels_per_mm:
                        diameter_mm = diameter_px / pixels_per_mm
                        label = f"{diameter_mm:.2f} mm ({diameter_px:.1f} px)"
                    else:
                        label = f"{diameter_px:.1f} px"
                    annotate_text(eye_color, f"Pupil: {label}", 6, 20, scale=0.55, color=(0,255,0))
                    pupils_found.append(diameter_px)

                if SHOW_DEBUG:
                    dbg = cv2.resize(debug["norm"], (ew//2, eh//2))
                    thr = cv2.resize(debug["thr"], (ew//2, eh//2))
                    edges = cv2.resize(debug["edges"], (ew//2, eh//2))
                    eye_color[0:eh//2, 0:ew//2] = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
                    eye_color[0:eh//2, ew//2:ew] = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
                    eye_color[eh//2:eh, 0:ew//2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        annotate_text(frame_disp, "Q: quit  K: calibrate  R: reset  S: save", 8, 24)
        if pixels_per_mm:
            annotate_text(frame_disp, f"Scale: {pixels_per_mm:.3f} px/mm  (press R to reset)", 8, 50, scale=0.65, color=(0,255,255))
        else:
            annotate_text(frame_disp, "Scale: not set (press K to calibrate for mm)", 8, 50, scale=0.65, color=(0,215,255))

        if awaiting_calibration_clicks:
            annotate_text(frame_disp, "Calibration: click TWO points, then enter mm in terminal", 8, 76, scale=0.65, color=(0,255,255))
            for p in calibration_points:
                cv2.circle(frame_disp, p, 5, (0,255,255), -1)
            if len(calibration_points) == 2:
                cv2.line(frame_disp, calibration_points[0], calibration_points[1], (0,255,255), 2)

        if len(pupils_found) >= 1:
            mean_px = float(np.mean(pupils_found))
            if pixels_per_mm:
                mean_mm = mean_px / pixels_per_mm
                annotate_text(frame_disp, f"Mean pupil diameter: {mean_mm:.2f} mm", 8, H - 16, scale=0.9, color=(0,255,0))
            else:
                annotate_text(frame_disp, f"Mean pupil diameter: {mean_px:.1f} px", 8, H - 16, scale=0.9, color=(0,255,0))

        cv2.imshow("Pupil Diameter", frame_disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('k'), ord('K')):
            awaiting_calibration_clicks = True
            calibration_points.clear()
            print("\n[Calibration] Click TWO points in the video window that are a known distance apart,")
            print("then enter that real distance in millimeters in the terminal.\n")
        elif key in (ord('r'), ord('R')):
            pixels_per_mm = None
            awaiting_calibration_clicks = False
            calibration_points.clear()
            print("[Calibration] Cleared. Reporting in pixels.")
        elif key in (ord('s'), ord('S')):
            now = time.time()
            if now - last_save > 0.5:
                last_save = now
                fname = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fname, frame_disp)
                print(f"[Saved] {os.path.abspath(fname)}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
