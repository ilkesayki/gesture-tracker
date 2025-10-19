# >>> BUILD: gesture_v2_panel_only_4_gestures <<<
import sys, time, math, os
import cv2
import numpy as np

# ==== CONFIG ====
CONF = {
    "win_w": 960, "win_h": 540,
    "panel_w": 480, "panel_h": 480,
    "model_complexity": 1,
    "min_det_conf": 0.6,
    "min_track_conf": 0.5,
    "debounce_frames": 6,          # jest sabitleme
    "face_min_conf": 0.5,
    "near_face_ratio": 0.4,       # INDEX_NEAR_FACE için eşik
}
GESTURE_IMAGES = {
    "INDEX_NEAR_FACE": "imgs/index_near_face.jpg",
    "MIDDLE_UP":       "imgs/middle_up.jpg",
    "INDEX_UP":        "imgs/index_up.jpg",
    "IDLE":            "imgs/idle.jpg",
}
# ================

# Logları sustur
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

# MediaPipe
try:
    import mediapipe as mp
except Exception as e:
    print("MediaPipe import edilemedi. Venv'te şunu çalıştır:\n"
          "python -m pip install 'mediapipe>=0.10.10,<0.11'")
    raise

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

TIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
PIP_IDS = [3, 6, 10, 14, 18]
WRIST_ID = 0
INDEX_MCP = 5

# ---------- yardımcılar ----------
def landmark_to_np(landmarks, image_shape):
    h, w = image_shape[:2]
    pts = []
    for lm in landmarks.landmark:
        pts.append(np.array([lm.x * w, lm.y * h, lm.z], dtype=np.float32))
    return np.stack(pts)  # (21,3)

def dist2d(a, b):
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))

def face_center_and_size(detection, frame_shape):
    h, w = frame_shape[:2]
    rb = detection.location_data.relative_bounding_box
    cx = (rb.xmin + rb.width/2.0) * w
    cy = (rb.ymin + rb.height/2.0) * h
    size = max(rb.width * w, rb.height * h)
    return np.array([cx, cy], dtype=np.float32), float(size)

def finger_states(lms_np, handedness_label):
    states = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
    thumb_tip = lms_np[TIP_IDS[0]]
    thumb_ip  = lms_np[PIP_IDS[0]]
    # Başparmak yönü (sağ/sol ele göre x ekseni)
    if handedness_label == "Right":
        states["thumb"] = thumb_tip[0] > thumb_ip[0]
    else:
        states["thumb"] = thumb_tip[0] < thumb_ip[0]
    # Diğer parmaklar (tip daha yukarıysa açık)
    names = ["index", "middle", "ring", "pinky"]
    for i, name in enumerate(names, start=1):
        tip = lms_np[TIP_IDS[i]]
        pip = lms_np[PIP_IDS[i]]
        states[name] = tip[1] < pip[1]
    return states

# ---- SADE JEST MANTIĞI: sadece 4 durum ----
# 1) INDEX_NEAR_FACE   2) MIDDLE_UP   3) INDEX_UP   4) IDLE
def classify_four(lms_np, handed_label, face_info):
    states = finger_states(lms_np, handed_label)

    # INDEX_UP: işaret açık, orta+yüzük+serçe kapalı (başparmak serbest)
    index_up = states["index"] and (not states["middle"]) and (not states["ring"]) and (not states["pinky"])

    # MIDDLE_UP: orta açık, işaret+yüzük+serçe kapalı (başparmak serbest)
    middle_up = states["middle"] and (not states["index"]) and (not states["ring"]) and (not states["pinky"])

    # INDEX_NEAR_FACE: yüz var + işaret ucu yüz merkezine yakın
    near_face = False
    if face_info is not None:
        face_ctr, face_size = face_info
        index_tip = lms_np[8][:2]
        thr = CONF["near_face_ratio"] * (face_size + 1e-6)
        if dist2d(index_tip, face_ctr) < thr:
            near_face = True

    if near_face:
        return "INDEX_NEAR_FACE"
    if middle_up:
        return "MIDDLE_UP"
    if index_up:
        return "INDEX_UP"
    return "IDLE"

def draw_label(img, text, org, scale=0.8):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    x = max(5, min(x, img.shape[1]-w-15))
    y = max(h+baseline+8, min(y, img.shape[0]-8))
    cv2.rectangle(img, (x, y - h - baseline - 6), (x + w + 10, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

# ---- Kamera (fallback + warm-up) ----
def open_camera(index=0):
    for idx in (index, 1 - index):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        time.sleep(0.2)
        ok = False
        for _ in range(8):
            ret, _ = cap.read()
            if ret:
                ok = True
                break
            time.sleep(0.05)
        if ok:
            print(f"[camera] using index {idx} (AVFOUNDATION)")
            return cap
        cap.release()
    cap = cv2.VideoCapture(index)  # CAP_ANY
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return cap

def switch_to(idx, current_cap):
    new_cap = open_camera(idx)
    if new_cap.isOpened():
        ok = False
        for _ in range(8):
            ret, _ = new_cap.read()
            if ret:
                ok = True
                break
            time.sleep(0.03)
        if ok:
            if current_cap is not None:
                current_cap.release()
            print(f"[camera] switched to index {idx}")
            return new_cap, idx
        new_cap.release()
    print(f"[camera] could not switch to {idx}")
    return None, None

# ---- Panel yardımcıları ----
def letterbox(img, out_size):
    H, W = img.shape[:2]
    outW, outH = out_size
    scale = min(outW / W, outH / H)
    nw, nh = int(W * scale), int(H * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((outH, outW, 3), dtype=np.uint8)
    x0 = (outW - nw) // 2
    y0 = (outH - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def make_placeholder(text, size, bg=(40,40,40)):
    W, H = size
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    y = 40
    for line in text.split("\n"):
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,200,255), 2, cv2.LINE_AA)
        y += 32
    return canvas

def load_image_safe(path, panel_size):
    W, H = panel_size
    if path and os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return letterbox(img, (W, H))
        return make_placeholder(f"Cannot read:\n{path}", (W, H))
    return make_placeholder("No image:\n"+str(path), (W, H))

def build_panel_cache(panel_size):
    cache = {}
    for gest, path in GESTURE_IMAGES.items():
        cache[gest] = load_image_safe(path, panel_size)
    for g in ["INDEX_NEAR_FACE", "MIDDLE_UP", "INDEX_UP", "IDLE"]:
        if g not in cache:
            cache[g] = make_placeholder(f"{g}\n(no mapping)", panel_size)
    return cache

# ---------------------- main ----------------------
def main():
    print(">>> BUILD: gesture_v2_panel_only_4_gestures is running")
    cam_index = 0
    if len(sys.argv) >= 2:
        try:
            cam_index = int(sys.argv[1])
        except:
            pass

    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"Kamera açılamadı (index {cam_index}). 0/1 ile tekrar deneyin.")
        return

    # Pencereler
    win_cam   = "Gesture Tracker — [q/ESC] quit  [C] switch  [0/1] jump  [H] panel  [R] reload"
    win_panel = "Gesture Panel"
    cv2.namedWindow(win_cam,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_panel, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_cam,   CONF["win_w"],   CONF["win_h"])
    cv2.resizeWindow(win_panel, CONF["panel_w"], CONF["panel_h"])

    prev_t = time.time()
    stable_gesture = "IDLE"
    last_gesture = "IDLE"
    count_same = 0
    N_STABLE = CONF["debounce_frames"]
    miss = 0

    panel_size = (CONF["panel_w"], CONF["panel_h"])
    panel_cache = build_panel_cache(panel_size)
    show_panel = True

    # paneli ilk karede göster
    cv2.imshow(win_panel, panel_cache.get(stable_gesture))

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=CONF["model_complexity"],
        min_detection_confidence=CONF["min_det_conf"],
        min_tracking_confidence=CONF["min_track_conf"]
    ) as hands, mp_face.FaceDetection(model_selection=0, min_detection_confidence=CONF["face_min_conf"]) as face_det:

        while True:
            ret, frame = cap.read()
            if not ret:
                miss += 1
                if miss > 30:
                    cap.release()
                    cap = open_camera(cam_index)
                    miss = 0
                time.sleep(0.02)
                continue
            miss = 0

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Yüz
            face_info = None
            f_res = face_det.process(rgb)
            if f_res and f_res.detections:
                det = max(f_res.detections, key=lambda d: d.score[0] if d.score else 0.0)
                if det.score and det.score[0] >= CONF["face_min_conf"]:
                    face_info = face_center_and_size(det, frame.shape)
                    ctr, sz = face_info
                    cv2.circle(frame, (int(ctr[0]), int(ctr[1])), max(6, int(sz*0.02)), (0, 120, 255), 2)

            # Elller
            result = hands.process(rgb)

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_lms = result.multi_hand_landmarks[0]
                handed    = result.multi_handedness[0]
                handed_label = handed.classification[0].label  # "Left"/"Right"
                lms_np = landmark_to_np(hand_lms, frame.shape)

                gest = classify_four(lms_np, handed_label, face_info)

                # debounce
                if gest == last_gesture:
                    count_same += 1
                else:
                    count_same = max(1, count_same - 2)
                    last_gesture = gest
                if count_same >= N_STABLE:
                    if gest != stable_gesture:
                        stable_gesture = gest
                        if show_panel:
                            cv2.imshow(win_panel, panel_cache.get(stable_gesture))

                # çizim
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(thickness=2)
                )
                wrist_xy = lms_np[WRIST_ID][:2].astype(int)
                draw_label(frame, f"{handed_label}: {stable_gesture}",
                           (int(wrist_xy[0]), int(wrist_xy[1])))

            # FPS & ipucu
            now = time.time()
            fps = 1.0 / (now - prev_t + 1e-6)
            prev_t = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"[C] switch 0<->1  [0]/[1] jump  [H] panel  [R] reload  [q]/[ESC] quit (cam={cam_index})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, "BUILD: gesture_v2_panel_only_4_gestures", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Gesture: {stable_gesture}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)

            cv2.imshow(win_cam, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:  # q / ESC
                break
            elif k == ord('c'):
                target = 1 - cam_index
                new_cap, new_idx = switch_to(target, cap)
                if new_cap is not None:
                    cap, cam_index = new_cap, new_idx
            elif k in (ord('0'), ord('1')):
                target = 0 if k == ord('0') else 1
                if target != cam_index:
                    new_cap, new_idx = switch_to(target, cap)
                    if new_cap is not None:
                        cap, cam_index = new_cap, new_idx
            elif k == ord('h'):
                show_panel = not show_panel
                if show_panel:
                    cv2.imshow(win_panel, panel_cache.get(stable_gesture))
            elif k == ord('r'):
                panel_cache = build_panel_cache(panel_size)
                cv2.imshow(win_panel, panel_cache.get(stable_gesture))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
