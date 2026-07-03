import threading
import time
import traceback
import sys
import os
from collections import deque
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None
from vision_module import CameraSystem
from arduino_link import ArduinoLink
from automation_logic import AutomationLogic

app = Flask(__name__)

# Initialize components — auto-select simulation mode on non-Pi (no Picamera2)
if Picamera2 is not None:
    messenger = ArduinoLink(port='/dev/ttyUSB0', use_simulation=False)
else:
    messenger = ArduinoLink(use_simulation=True)
eyes = CameraSystem(
    focal_length_px=AutomationLogic.FOCAL_LENGTH_PX,
    marker_physical_width_mm=AutomationLogic.MARKER_PHYSICAL_WIDTH_MM,
)
brain = AutomationLogic(eyes, messenger)

# Camera
camera_available = False
camera = None
webcam = None
if Picamera2 is not None:
    try:
        camera = Picamera2()
        camera.configure(camera.create_preview_configuration(
            main={"format": "BGR888", "size": (1296, 972)}))
        camera.start()
        camera_available = True
    except Exception as e:
        print(f"[CAMERA] Not available: {e}")
else:
    try:
        # On Windows, OpenCV defaults to the MSMF backend, which frequently grabs only the
        # first frame from a laptop webcam and then fails every subsequent grab (freezing the
        # feed on one still). DirectShow is far more reliable, so prefer it on Windows.
        backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
        webcam = cv2.VideoCapture(0, backend)
        if webcam.isOpened():
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera_available = True
            print(f"[CAMERA] Laptop webcam opened (simulation mode, backend={backend})")
        else:
            webcam.release()
            webcam = None
            print("[CAMERA] No webcam found — vision features disabled")
    except Exception as e:
        print(f"[CAMERA] Webcam open failed: {e}")

# Shared frame buffers. latest_frame is annotated (overlays drawn on it) for the MJPEG stream;
# latest_clean_frame is the raw pre-overlay frame used for calibration capture (so the on-feed
# graphics never obscure the marker being measured).
latest_frame = None
latest_clean_frame = None
frame_lock = threading.Lock()

# Homing state lives on the messenger (messenger.has_homed / .is_homing) so the brain can
# enforce it as an operating precondition too; /status just reflects those flags.

MM_PER_STEP = AutomationLogic.MM_PER_STEP_XY   # single source of truth for mm/step

# Last status message from brain.run_cycle() — polled by /status
last_status_message = "System Ready."

# Last abort/safety-stop reason — sticky so the operator can see *why* an operation ended.
# brain.run_cycle() returns the reason for one cycle then drops to IDLE ("System Ready."),
# so we latch it here (with a timestamp) and surface it via /status until dismissed.
last_abort_reason = None
last_abort_time = None

# ── Debug log ────────────────────────────────────────────────────────────────
# A rolling history of the brain's per-cycle status so the operator can see what the
# robot is doing (and why) instead of one flickering line. The automation loop runs
# ~30 fps but the status string only changes on a real transition, so we dedup: a
# repeat bumps `count` on the last entry (showing a stall as one line ticking ×N)
# rather than spamming identical rows. Each entry is also appended to a per-run file
# so a crashed or manually-killed run still leaves a post-mortem trail.
debug_log = deque(maxlen=300)
debug_log_lock = threading.Lock()
_log_seq = 0
_prev_log_msg = None

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
_log_file = None
try:
    os.makedirs(_LOG_DIR, exist_ok=True)
    _log_path = os.path.join(_LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    _log_file = open(_log_path, 'a', encoding='utf-8')
    print(f"[DEBUG LOG] writing to {_log_path}")
except Exception as e:
    print(f"[DEBUG LOG] could not open log file: {e}")


def _record_debug(status_message):
    """Append a structured debug entry for this cycle, deduping repeats into a ×N count.
    Safe to call every frame — only a changed message creates a new row."""
    global _log_seq, _prev_log_msg
    with debug_log_lock:
        if status_message == _prev_log_msg and debug_log:
            debug_log[-1]['count'] += 1
            return
        _prev_log_msg = status_message
        _log_seq += 1
        entry = {
            'id':           _log_seq,
            't':            time.time(),
            'state':        brain.current_state,
            'msg':          status_message,
            'x_err':        brain.last_x_err,
            'z_err':        brain.last_z_err,
            'seq_step':     getattr(brain, '_seq_step', None),
            'search_count': getattr(brain, 'search_count', None),
            'lost_frames':  getattr(brain, 'lost_frames', None),
            'grip':         messenger.grip_engaged,
            'busy':         messenger.is_busy,
            'count':        1,
        }
        debug_log.append(entry)
        if _log_file is not None:
            try:
                ts = datetime.fromtimestamp(entry['t']).strftime('%H:%M:%S')
                _log_file.write(
                    f"{ts}  [{entry['state']}]  {status_message}  "
                    f"(x_err={entry['x_err']}, z_err={entry['z_err']}, "
                    f"step={entry['seq_step']}, grip={int(entry['grip'])})\n"
                )
                _log_file.flush()
            except Exception:
                pass

_align_flash_until = 0.0
_prev_aligned = False

# Feed health (no-frame / all-black detection)
_frame_bad_since = None

# Arduino link heartbeat — periodically poll position so a healthy-but-idle link keeps producing RX
# (the Arduino is silent between events), letting messenger.link_alive distinguish idle from dead.
_last_ping = 0.0
PING_INTERVAL_S = 2.0


def _draw_label(frame, text, x, y, fg, scale=0.5, thick=1):
    """Draw text with a dark background chip so it stays readable over any scene.
    (x, y) is the bottom-left of the text baseline."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (x - 3, y - th - 5), (x + tw + 3, y + base - 1), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y - 2), font, scale, fg, thick, cv2.LINE_AA)


def _dashed_line(frame, p1, p2, color, thick=1, dash=9, gap=6):
    p1 = np.array(p1, float); p2 = np.array(p2, float)
    d = float(np.hypot(*(p2 - p1)))
    if d < 1:
        return
    u = (p2 - p1) / d
    step = dash + gap
    n = int(d // step) + 1
    for i in range(n):
        a = p1 + u * (i * step)
        b = p1 + u * min(i * step + dash, d)
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)


def _dashed_poly(frame, pts, color, thick=1):
    for i in range(len(pts)):
        _dashed_line(frame, pts[i], pts[(i + 1) % len(pts)], color, thick)


def _draw_checks(frame, x, y, checks):
    """Row of small pass/fail chips (e.g. L/R  U/D  Depth) — green = in tolerance, red = off.
    Lets the operator see WHICH term is keeping the marker red without leaving the feed."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, ok in checks:
        col = (0, 210, 0) if ok else (0, 0, 255)
        _draw_label(frame, label, x, y, col, scale=0.4)
        (tw, _), _ = cv2.getTextSize(label, font, 0.4, 1)
        x += tw + 10


def _tilt_lines(o, t, tol):
    """Self-explanatory tilt readout. Returns a list of (text, color) lines: 'Tilt OK' (green)
    when square-on within tolerance, else a red header plus, per offending axis, the live angle
    AND its acceptable range — so the number explains itself (e.g. 'Yaw 16deg (ok: -10 to 10)')."""
    GREEN, RED = (0, 210, 0), (0, 0, 255)
    bad = []
    for k, lab in (('roll', 'Roll'), ('pitch', 'Pitch'), ('yaw', 'Yaw')):
        lo, hi = t[k] - tol[k + '_deg'], t[k] + tol[k + '_deg']
        if not (lo <= o[k] <= hi):
            bad.append(f"{lab} {o[k]:.0f}deg (ok: {lo:.0f} to {hi:.0f})")
    if not bad:
        return [("Tilt OK", GREEN)]
    return [("Tilt out of range:", RED)] + [("  " + b, RED) for b in bad]


def _draw_depth_bar(frame, cur_w, aim_w, eng_w, tol_w, col):
    """Vertical 'drive-in' depth gauge on the right edge. Marker WIDTH is the depth proxy (the
    closer the scoop, the wider the marker), so this surfaces the Y-approach — the one axis with
    no other on-feed presence. The scale is FIXED from the calibrated widths (so nothing drifts
    frame-to-frame): a green band = the engaged depth (± tolerance) you drive in to, a faint tick
    = the line-up depth, and a pointer rises to the current depth, turning green inside the band."""
    h, w = frame.shape[:2]
    x, top, bot = w - 22, 74, h - 74
    lo = min(aim_w, eng_w) * 0.80          # fixed ends — from the calibration widths only
    hi = max(aim_w, eng_w) * 1.15
    if hi <= lo:
        return

    def y_of(val):
        f = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        return int(bot - f * (bot - top))   # wider (closer / more engaged) sits higher

    cv2.rectangle(frame, (x, top), (x + 12, bot), (55, 55, 55), -1)          # track
    yb1, yb2 = y_of(eng_w + tol_w), y_of(eng_w - tol_w)                      # "engaged" band
    cv2.rectangle(frame, (x, yb1), (x + 12, yb2), (0, 120, 0), -1)
    _draw_label(frame, "engaged", x - 58, (yb1 + yb2) // 2 + 4, (0, 210, 0), scale=0.35)
    yl = y_of(aim_w)                                                         # line-up tick
    cv2.line(frame, (x - 4, yl), (x + 16, yl), (170, 170, 170), 1, cv2.LINE_AA)
    _draw_label(frame, "line up", x - 54, yl + 4, (170, 170, 170), scale=0.35)
    yc = y_of(cur_w)                                                         # current-depth pointer
    pc = (0, 210, 0) if abs(cur_w - eng_w) <= tol_w else (255, 255, 255)
    cv2.fillPoly(frame, [np.array([(x - 2, yc), (x - 11, yc - 5), (x - 11, yc + 5)])], pc)
    cv2.line(frame, (x, yc), (x + 12, yc), pc, 1, cv2.LINE_AA)
    _draw_label(frame, "DEPTH", x - 14, top - 6, col, scale=0.4)


def _draw_crosshair(frame, tx, ty, xr, zr, col, faint=False):
    """Aim crosshair (+) and its dashed X/Z position-tolerance window. `faint` = just a small
    marker (idle preview of a saved profile crosshair, no tolerance box)."""
    if faint:
        c = tuple(int(v * 0.55) for v in col)
        cv2.line(frame, (tx - 6, ty), (tx + 6, ty), c, 1, cv2.LINE_AA)
        cv2.line(frame, (tx, ty - 6), (tx, ty + 6), c, 1, cv2.LINE_AA)
        return
    _dashed_poly(frame, [(tx - xr, ty - zr), (tx + xr, ty - zr),
                         (tx + xr, ty + zr), (tx - xr, ty + zr)], col, 1)
    cv2.line(frame, (tx - 9, ty), (tx + 9, ty), col, 1, cv2.LINE_AA)
    cv2.line(frame, (tx, ty - 9), (tx, ty + 9), col, 1, cv2.LINE_AA)
    _draw_label(frame, "tol", tx + xr + 3, ty - zr + 12, col, scale=0.4)


def _draw_phase_banner(frame, phases, idx, col):
    """Top-left strip of the op's phases, left→right: done = green, active = bright/boxed,
    upcoming = grey. Keeps the operator's eyes on the video during a live op (#5)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 8, 18
    for i, ph in enumerate(phases):
        if i == idx:
            c, sc = col, 0.5
        elif i < idx:
            c, sc = (90, 200, 120), 0.42
        else:
            c, sc = (130, 130, 130), 0.42
        _draw_label(frame, ph, x, y, c, scale=sc)
        (tw, _), _ = cv2.getTextSize(ph, font, sc, 1)
        x += tw + 10


def _draw_feed_banner(frame):
    """Overlay a clear 'no camera frames' message so a black/dead feed isn't silent."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 200), 8)
    cv2.rectangle(frame, (0, h // 2 - 42), (w, h // 2 + 42), (0, 0, 110), -1)
    cv2.putText(frame, "NO CAMERA FRAMES", (24, h // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "camera likely in use - Ctrl+C, kill stray python, reopen",
                (24, h // 2 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 255), 1, cv2.LINE_AA)


def _automation_loop():
    global latest_frame, latest_clean_frame, last_status_message, _align_flash_until, _prev_aligned
    global _frame_bad_since, last_abort_reason, last_abort_time, _last_ping
    while True:
        # Heartbeat first, independent of the camera: poll position every PING_INTERVAL_S so an idle
        # link stays "alive". Skip in sim, during homing (firmware is busy), and while a move is in
        # flight (don't interleave an extra command on the serial line).
        if (not messenger.is_simulated and not messenger.is_homing and not messenger.is_busy
                and time.time() - _last_ping > PING_INTERVAL_S):
            messenger.ping()
            _last_ping = time.time()

        if not camera_available:
            time.sleep(0.1)
            continue
        try:
            got = False
            if camera is not None:
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # picamera2 outputs RGB natively
                got = True
            elif webcam is not None:
                ret, frame = webcam.read()                       # cv2.VideoCapture gives BGR
                got = bool(ret)
                if not got:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # canvas for the banner
            else:
                time.sleep(0.1)
                continue

            # Camera may be mounted upside down — rotate 180° at the SOURCE so vision
            # (brain.run_cycle), the clean calibration frame, AND the streamed feed all see the
            # same corrected image. Flipping only the viewer would leave alignment math working on
            # the un-flipped frame. Toggled via /set_flip, persisted in calibration.json.
            if got and brain.cal.get('flip_feed'):
                frame = cv2.flip(frame, -1)   # -1 = flip both axes = 180° rotation

            # ── Feed health: flag no-frame / all-black for ~2s ──
            now = time.time()
            bad = not got
            if got:
                bad = float(frame.mean()) < 8.0   # near-black
            if bad:
                if _frame_bad_since is None:
                    _frame_bad_since = now
            else:
                _frame_bad_since = None
            unhealthy = _frame_bad_since is not None and (now - _frame_bad_since) > 2.0

            if not got:
                # No real image to process — publish a banner frame and move on.
                if unhealthy:
                    _draw_feed_banner(frame)
                with frame_lock:
                    latest_frame = frame
                time.sleep(0.05)
                continue

            # Stash a clean (pre-overlay) copy for calibration capture before any drawing.
            with frame_lock:
                latest_clean_frame = frame.copy()

            status_message = brain.run_cycle(frame)
            last_status_message = status_message
            _record_debug(status_message)

            # Latch abort / safety-stop reasons so they don't vanish when the brain drops to IDLE.
            if status_message and (status_message.startswith("ABORT")
                                   or status_message.startswith("SAFETY STOP")):
                last_abort_reason = status_message
                last_abort_time = time.time()

            # Always outline/label visible markers on the feed, even in states that don't run
            # vision (e.g. IDLE) — alignment handlers only detect while actively aligning.
            if not eyes.last_corners:
                eyes._detect_all(frame)

            if brain.is_aligned and not _prev_aligned:
                _align_flash_until = time.time() + 0.5
            _prev_aligned = brain.is_aligned

            # Active-op context, computed once and reused by the marker overlays below.
            #   ap        : which calibration profile is live ('pickup'/'setdown'), else None
            #   tx_c/ty_c : the effective aim crosshair. During a VISION op we always have one —
            #               the saved/active crosshair, or the frame centre the alignment math
            #               falls back to — so the tolerance box, chips and arrow are consistent
            #               (rule: they appear during an op, never when idle). None when idle.
            fh, fw = frame.shape[:2]
            ap    = brain.active_profile
            tol   = brain.active_tol
            if ap is not None:
                tx_c = eyes.target_x if eyes.target_x is not None else fw // 2
                ty_c = eyes.target_y if eyes.target_y is not None else fh // 2
            else:
                tx_c, ty_c = eyes.target_x, eyes.target_y
            eng_w = brain.profile(ap)['target_width_px'] if ap else None

            # Outline + label every detected marker. The active/target marker is green only when
            # ALL alignment terms are in tolerance, else red; non-active markers are dim grey and
            # carry no telemetry (keeps a multi-marker scene readable — #7). Each label sits on a
            # dark chip so it stays readable over any background.
            for mid, corners in eyes.last_corners:
                pts = corners.astype(int)
                is_target = (brain.target_id is not None and mid == brain.target_id)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                if is_target:
                    cur_w     = eyes._marker_size(corners)
                    x_ok      = (tx_c is None or abs(cx - tx_c) <= tol['x_px'])
                    z_ok      = (ty_c is None or abs(cy - ty_c) <= tol['z_px'])
                    pos_ok    = x_ok and z_ok
                    width_ok  = (eng_w is None or abs(cur_w - eng_w) <= tol['width_px'])
                    orient_ok = brain.is_orientation_ok
                    color = (0, 255, 0) if (pos_ok and width_ok and orient_ok) else (0, 0, 255)
                    thick = 3
                else:
                    color = (160, 160, 160)
                    thick = 2
                cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True, color, thick, cv2.LINE_AA)
                lx    = int(pts[:, 0].min())
                top_y = int(pts[:, 1].min())
                bot_y = int(pts[:, 1].max())
                label_y = top_y - 6 if top_y > 24 else bot_y + 18   # below the marker if no room above
                _draw_label(frame, brain.label_for(mid), lx, label_y, color)

                if is_target:
                    # Plain-English alignment chips (no robot-axis jargon): left/right, up/down,
                    # depth — each green = in tolerance, red = off. Shows WHY the marker is red.
                    _draw_checks(frame, lx, label_y + 16,
                                 [("L/R", x_ok), ("U/D", z_ok), ("Depth", width_ok)])
                    # Tilt readout: 'Tilt OK', else each off-axis with its acceptable range so the
                    # number is self-explanatory (e.g. 'Yaw 16deg (ok: -10 to 10)').
                    if eyes.last_orientation:
                        ty_line = label_y + 32
                        for text, tcol in _tilt_lines(eyes.last_orientation, brain.orient_target, tol):
                            _draw_label(frame, text, lx, ty_line, tcol, scale=0.42)
                            ty_line += 15
                    # Error vector toward the crosshair — drawn only while off-position, so it
                    # vanishes once L/R + U/D are in tolerance. Amber when close, red when far.
                    if tx_c is not None and not pos_ok:
                        mag    = ((cx - tx_c) ** 2 + (cy - ty_c) ** 2) ** 0.5
                        avec_c = (0, 165, 255) if mag <= 1.5 * max(tol['x_px'], tol['z_px']) else (0, 0, 255)
                        cv2.arrowedLine(frame, (cx, cy), (int(tx_c), int(ty_c)),
                                        avec_c, 2, cv2.LINE_AA, tipLength=0.25)

            # Saved calibration references, drawn as the captured marker OUTLINES (perspective):
            # cyan = pick-up, magenta = set-down. Each profile has two: the aim pose (thick) and a
            # second deeper pose (thin), labelled in plain words so the operator can verify the
            # calibration directly. The second pose differs by op: pick-up's is the 'engaged' depth
            # (engage_corners), set-down's is the 'seated' pose (seated_corners). Hidden only in Track
            # (demo): that's a generic centering test, so the outlines there are unrelated clutter.
            if brain.current_state != AutomationLogic.STATE_DEMO_FOLLOW:
                for prof_name, col, name, second_key, second_lbl in (
                        ('pickup',  (255, 200, 0), 'pick-up',  'engage_corners', 'engaged'),
                        ('setdown', (255, 0, 200), 'set-down', 'seated_corners', 'seated')):
                    p = brain.profile(prof_name)
                    aim_lbl = "over-slot" if prof_name == 'setdown' else "line-up"
                    if p['aim_corners']:
                        poly = np.array(p['aim_corners'], dtype=int).reshape(-1, 1, 2)
                        cv2.polylines(frame, [poly], True, col, 2, cv2.LINE_AA)
                        ax, ay = int(p['aim_corners'][0][0]), int(p['aim_corners'][0][1])
                        _draw_label(frame, f"{name}: {aim_lbl}", ax, ay - 4, col, scale=0.4)
                    if p.get(second_key):
                        poly = np.array(p[second_key], dtype=int).reshape(-1, 1, 2)
                        cv2.polylines(frame, [poly], True, col, 1, cv2.LINE_AA)
                        ex, ey = int(p[second_key][0][0]), int(p[second_key][0][1])
                        _draw_label(frame, f"{name}: {second_lbl}", ex, ey - 4, col, scale=0.4)

            # Aim crosshair + position-tolerance window. During a vision op there's always an
            # effective crosshair (above), so the dashed X/Z tolerance box is shown consistently
            # whenever an op runs. When idle, faintly preview each calibrated profile's saved
            # crosshair so SET XZ REF can be eyeballed at a glance (#6).
            if ap is not None and tx_c is not None and ty_c is not None:
                col = (255, 200, 0) if ap == 'pickup' else (255, 0, 200)
                _draw_crosshair(frame, int(tx_c), int(ty_c),
                                int(tol['x_px']), int(tol['z_px']), col, faint=False)
            else:
                for prof_name, col in (('pickup', (255, 200, 0)), ('setdown', (255, 0, 200))):
                    pr = brain.profile(prof_name)
                    if pr['crosshair_x'] is not None:
                        _draw_crosshair(frame, int(pr['crosshair_x']), int(pr['crosshair_y']),
                                        0, 0, col, faint=True)

            # Depth gauge (#2/#4): marker width is the Y-approach depth proxy. Only meaningful
            # while actually driving in, so it's shown only during the approach/engage of a real
            # pick-up or target set-down — not during search, coarse align or Track.
            if (brain.current_state in (AutomationLogic.STATE_PICKING_UP,
                                        AutomationLogic.STATE_SETTING_DOWN_TARGET)
                    and ap is not None and eng_w is not None):
                aim_w = brain.profile(ap)['aim_width_px']
                tgt_c = next((c for m, c in eyes.last_corners if m == brain.target_id), None)
                if aim_w is not None and tgt_c is not None:
                    col = (255, 200, 0) if ap == 'pickup' else (255, 0, 200)
                    _draw_depth_bar(frame, eyes._marker_size(tgt_c), aim_w, eng_w,
                                    tol['width_px'], col)

            # Phase progress banner (#5): where in the op we are, without reading the status text.
            phase = brain.op_phase
            if phase is not None:
                col = (255, 200, 0) if ap == 'pickup' else ((255, 0, 200) if ap == 'setdown' else (0, 255, 255))
                _draw_phase_banner(frame, phase[0], phase[1], col)

            # (Status text is shown in the status bar — State pill + Msg — not duplicated on the feed.)

            if time.time() < _align_flash_until:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 20)

            if brain.current_state == AutomationLogic.STATE_DEMO_FOLLOW:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 0), 6)

            if unhealthy:   # all-black camera — make it obvious, not a silent black feed
                _draw_feed_banner(frame)

            with frame_lock:
                latest_frame = frame
        except Exception:
            # Never let the capture thread die silently — that freezes the feed on one frame.
            print("[AUTOMATION LOOP ERROR]")
            traceback.print_exc()
            time.sleep(0.1)


def generate_frames():
    if not camera_available:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (639, 479), (0, 0, 120), 3)
        cv2.putText(img, "No Camera Detected", (90, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 220), 2)
        cv2.putText(img, "Connect camera and restart", (70, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 180), 1)
        cv2.putText(img, "to enable vision features", (90, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 180), 1)
        _, buf = cv2.imencode('.jpg', img)
        payload = buf.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
            time.sleep(1.0)

    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)   # cap ~30 fps so this generator doesn't peg a core / starve capture


def _do_home():
    # Stop any autonomous op before homing so the brain thread can't issue moves that
    # interleave with the homing sequence on the serial link.
    global last_abort_reason, last_abort_time
    brain.abort_step()
    brain.current_state = brain.STATE_IDLE
    brain.is_aligned = False
    ok = messenger.home()   # manages messenger.has_homed / .is_homing internally
    if not ok and messenger.last_home_error:
        # Surface the failure (timeout / E-stop) in the sticky "Next step" banner so a homing that
        # never completes isn't a silent hang — the operator sees why and what to check.
        last_abort_reason = f"ABORT: {messenger.last_home_error}"
        last_abort_time = time.time()


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.after_request
def add_no_cache_headers(resp):
    # This is a live control dashboard — never let the browser serve a stale page, status,
    # or (critically) a cached MJPEG frame from a previous session.
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    orient = eyes.last_orientation
    orient_rounded = (
        {k: round(v, 1) for k, v in orient.items()} if orient else None
    )
    return jsonify({
        'camera_available': camera_available,
        'simulated':      messenger.is_simulated,
        'link_alive':     messenger.link_alive,
        'link_error':     messenger.link_error,
        'homed':          messenger.has_homed,
        'is_homing':      messenger.is_homing,
        'state':          brain.current_state,
        'x_pos':          messenger.x_pos,
        'y_pos':          messenger.y_pos,
        'z_pos_mm':       round(messenger.z_pos_mm, 1),
        'aligned':        brain.is_aligned,
        'load_detected':  messenger.load_detected,
        'grip_engaged':   messenger.grip_engaged,
        'limits':         messenger.limits,
        'x_err':          brain.last_x_err,
        'z_err':          brain.last_z_err,
        'crosshair_x':    eyes.target_x,
        'crosshair_y':    eyes.target_y,
        'status_msg':     last_status_message,
        'last_abort':     (None if last_abort_reason is None else
                           {'reason': last_abort_reason, 't': last_abort_time}),
        'step_mode':         brain.step_mode,
        'awaiting_approval': brain.awaiting_approval,
        'pending_step':      brain.pending_step,
        'orientation':    orient_rounded,
        'orientation_ok': brain.is_orientation_ok,
        'orient_tol': {                              # active operation's orientation tolerances
            'roll':  brain.active_tol['roll_deg'],
            'pitch': brain.active_tol['pitch_deg'],
            'yaw':   brain.active_tol['yaw_deg'],
        },
        'orient_target': {k: round(v, 1) for k, v in brain.orient_target.items()},
        'align_tol':     dict(brain.active_tol),     # active op's full tolerance set (live readout)
        'tolerances':    brain.cal['tolerances'],    # stored per-profile (for the Setup card)
        'calibration': {
            'pickup':  {**brain.profile('pickup'),  'ready': brain.cal_ready('pickup')},
            'setdown': {**brain.profile('setdown'), 'ready': brain.cal_ready('setdown'),
                        'seated_ready': brain.seated_ready('setdown')},
        },
        'z_heights':       brain.cal['z_heights'],
        'z_ready':         brain.z_ready,
        'accel':           messenger.acceleration,
        'accel_confirmed': messenger.accel_confirmed,
        'speed':           messenger.speed,
        'speed_confirmed': messenger.speed_confirmed,
        'busy':            messenger.is_busy,
        'loaded_motion':   brain.loaded_motion,
        'mm_per_step':     MM_PER_STEP,
        'marker_labels':   brain.cal['marker_labels'],
        'flip_feed':       bool(brain.cal.get('flip_feed')),
        'settle_time_s':   brain.settle_time_s,
        'z_mm_per_s':      brain.z_mm_per_s,
    })

@app.route('/clear_abort', methods=['POST'])
def clear_abort():
    """Dismiss the latched abort/safety-stop reason (operator acknowledged it)."""
    global last_abort_reason, last_abort_time
    last_abort_reason = None
    last_abort_time = None
    return jsonify({'status': 'cleared'})

@app.route('/debug_log')
def debug_log_route():
    """Return debug-log entries. ?since=<id> returns only rows newer than that id so the
    UI can poll incrementally; omit it to fetch the whole buffer."""
    since = request.args.get('since', type=int)
    with debug_log_lock:
        items = [e for e in debug_log if since is None or e['id'] > since]
        return jsonify({'logs': items, 'last_id': _log_seq})

@app.route('/debug_log/clear', methods=['POST'])
def debug_log_clear():
    """Clear the in-memory debug view. The per-run file is untouched (durable record)."""
    global _prev_log_msg
    with debug_log_lock:
        debug_log.clear()
        _prev_log_msg = None
    return jsonify({'status': 'cleared', 'last_id': _log_seq})

@app.route('/home')
def home():
    threading.Thread(target=_do_home, daemon=True).start()
    return jsonify(result="Homing started")

@app.route('/emergency_stop')
def emergency_stop():
    messenger.emergency_stop()   # also clears messenger.has_homed / .is_homing
    brain.current_state = brain.STATE_IDLE
    brain.is_aligned = False
    return jsonify(result="EMERGENCY STOP")

@app.route('/target/pick_up/<int:marker_id>')
def target_pick_up(marker_id):
    if not camera_available:
        return jsonify(error="Camera not available"), 503
    brain.pick_up_target(marker_id)
    return jsonify(result=f"Pick Up Battery (marker ID {marker_id})")

@app.route('/demo_follow/<int:target_id>')
def demo_follow(target_id):
    if not camera_available:
        return jsonify(error="Camera not available"), 503
    brain.demo_follow(target_id)
    return jsonify(result=f"Track Marker for ID {target_id}")

@app.route('/our_slot/pick_up/<int:slot>')
def our_slot_pick_up(slot):
    if not 0 <= slot <= 2:
        return jsonify(result="Invalid slot"), 400
    brain.our_slot_pick_up(slot)
    return jsonify(result=f"Our Slot {slot + 1} Pick Up started")

@app.route('/our_slot/set_down/<int:slot>')
def our_slot_set_down(slot):
    if not 0 <= slot <= 2:
        return jsonify(result="Invalid slot"), 400
    brain.our_slot_set_down(slot)
    return jsonify(result=f"Our Slot {slot + 1} Set Down started")

@app.route('/step_mode/<int:on>')
def step_mode(on):
    """Toggle step mode (approve-each-move). off mid-op = resume autonomous."""
    brain.set_step_mode(bool(on))
    return jsonify(step_mode=brain.step_mode)

@app.route('/approve')
def approve():
    """Run the currently staged move (operator approved it)."""
    ok = brain.approve_step()
    return jsonify(approved=ok)

@app.route('/step_abort')
def step_abort():
    """Cancel the running operation from step mode."""
    brain.abort_step()
    return jsonify(result="aborted")

@app.route('/jog', methods=['POST'])
def jog():
    data = request.get_json()
    axis      = data.get('axis', 'x')
    direction = int(data.get('direction', 1))
    steps     = int(data.get('steps', 50))
    ms        = int(data.get('ms', 1000))

    # Z is pulse-based and self-throttling (pulse_lift_z drops a pulse while one
    # is in flight), and the Z-measure wizard drives it with its own long pulses
    # + explicit E-stops — so leave Z immediate and untouched here.
    if axis == 'z':
        messenger.pulse_lift_z(direction * ms)
        return jsonify(result=f"Jog Z {'+' if direction > 0 else ''}{direction * ms}ms")

    # XY: drop the jog if a move is still in flight. This is the hard backstop
    # against command flooding — at 9600 baud a held key can queue stepper moves
    # faster than they execute, backlogging the serial buffer (the gantry judders,
    # then lurches through the backlog after the key is released). Never enqueue on
    # top of an unfinished move.
    if messenger.is_busy:
        return jsonify(result="busy", busy=True)

    if axis == 'x':
        messenger.move_gantry_x(direction * steps)
        result = f"Jog X {'+' if direction > 0 else ''}{direction * steps} steps"
    elif axis == 'y':
        messenger.move_gantry_y(direction * steps)
        result = f"Jog Y {'+' if direction > 0 else ''}{direction * steps} steps"
    else:
        return jsonify(result="Unknown axis"), 400

    # Block until this move finishes (or a safety timeout) so the client can
    # self-pace a held key: the next jog only goes out once this one is done,
    # giving clean step-step-step instead of judder. Returns immediately in
    # simulation (the move commits synchronously, so is_busy is already False).
    deadline = time.time() + 8.0
    while messenger.is_busy and time.time() < deadline:
        time.sleep(0.01)
    return jsonify(result=result, busy=messenger.is_busy)

@app.route('/goto', methods=['POST'])
def goto():
    data = request.get_json()
    x_steps = int(round(float(data.get('x_mm', 0)) / MM_PER_STEP))
    y_steps = int(round(float(data.get('y_mm', 0)) / MM_PER_STEP))
    messenger.move_gantry(x_steps, y_steps)
    return jsonify(result=f"Moving to X={x_steps} Y={y_steps} steps")

@app.route('/goto_z', methods=['POST'])
def goto_z():
    target_mm = float(request.get_json().get('z_mm', 0))
    messenger.move_z_to(target_mm)
    return jsonify(result=f"Moving Z to {target_mm:.1f} mm")

@app.route('/set_accel', methods=['POST'])
def set_accel():
    accel = int(request.get_json().get('accel', 80))
    messenger.set_acceleration(accel)
    return jsonify(result=f"Accel set to {accel} steps/s²")

@app.route('/set_speed', methods=['POST'])
def set_speed():
    # Firmware clamps to [10, 200] steps/s and echoes the applied value back (synced via SPEED SET TO:).
    speed = int(request.get_json().get('speed', 60))
    messenger.set_speed(speed)
    return jsonify(result=f"Speed set to {speed} steps/s")

@app.route('/calibrate/<profile>/xz', methods=['POST'])
def calibrate_xz(profile):
    if profile not in ('pickup', 'setdown'):
        return jsonify(error="Invalid profile"), 400
    if not camera_available:
        return jsonify(error="Camera not available"), 503
    with frame_lock:
        frame = latest_clean_frame
    if frame is None:
        return jsonify(error="No frame available"), 503
    marker_id = int(request.get_json().get('marker_id', 0))
    found, mx, my, marker_w, orient, corners = eyes.locate_marker(frame, marker_id)
    if not found:
        return jsonify(error=f"Marker ID {marker_id} not visible in current frame"), 404
    p = brain.profile(profile)
    p['crosshair_x']  = int(mx)
    p['crosshair_y']  = int(my)
    p['aim_corners']  = corners                             # full outline for the feed
    p['aim_orient']   = orient                              # per-profile orientation target
    p['aim_width_px'] = int(round(marker_w))                # aim-pose depth (pickup: pre-engage)
    if profile == 'setdown':
        # Set-down's first capture is the "align-over-slot" pose (battery at full retract, parked
        # over the slot). Unlike pick-up there's a single relevant depth, so this same capture sets
        # the Y-stop width too — making the profile alignment-ready in one shot. (Pick-up's Y-stop is
        # the deeper engagement width, captured separately by the WIDTH endpoint.)
        p['target_width_px'] = int(round(marker_w))
    brain.save_calibration()
    label = "align-over-slot pose" if profile == 'setdown' else "XZ ref"
    return jsonify(result=f"{profile} {label} set to ({int(mx)}, {int(my)}), w={int(round(marker_w))}px")

@app.route('/calibrate/<profile>/width', methods=['POST'])
def calibrate_width(profile):
    if profile not in ('pickup', 'setdown'):
        return jsonify(error="Invalid profile"), 400
    if not camera_available:
        return jsonify(error="Camera not available"), 503
    with frame_lock:
        frame = latest_clean_frame
    if frame is None:
        return jsonify(error="No frame available"), 503
    marker_id = int(request.get_json().get('marker_id', 0))
    found, mx, my, marker_w, _, corners = eyes.locate_marker(frame, marker_id)
    if not found:
        return jsonify(error=f"Marker ID {marker_id} not visible in current frame"), 404
    p = brain.profile(profile)
    if profile == 'setdown':
        # Set-down's second capture is the "seated pose": where the slot flag marker sits in-frame
        # once the battery is fully seated. Used as the descent guard (stop lowering if the marker
        # reaches this without the grip releasing) + a placement sanity check. Does NOT change the
        # Y-stop width (that's the align-over-slot capture).
        p['seated_x']        = int(mx)
        p['seated_y']        = int(my)
        p['seated_width_px'] = int(round(marker_w))
        p['seated_corners']  = corners
        brain.save_calibration()
        return jsonify(result=f"setdown seated pose set to ({int(mx)}, {int(my)}), w={int(round(marker_w))}px")
    p['target_width_px'] = int(round(marker_w))
    p['engage_corners']  = corners                          # full outline at engagement depth
    brain.save_calibration()
    return jsonify(result=f"{profile} width ref set to {int(round(marker_w))}px")

@app.route('/calibrate/<profile>/reset', methods=['POST'])
def calibrate_reset(profile):
    if not brain.reset_profile(profile):
        return jsonify(error="Invalid profile"), 400
    return jsonify(result=f"{profile} calibration cleared")

@app.route('/calibrate/<profile>/tolerances', methods=['POST'])
def set_tolerances(profile):
    if not brain.set_tolerances(profile, request.get_json() or {}):
        return jsonify(error="Invalid profile"), 400
    return jsonify(result=f"{profile} tolerances updated", tolerances=brain.cal['tolerances'][profile])

@app.route('/calibrate/<profile>/tolerances/reset', methods=['POST'])
def reset_tolerances(profile):
    if not brain.reset_tolerances(profile):
        return jsonify(error="Invalid profile"), 400
    return jsonify(result=f"{profile} tolerances reset", tolerances=brain.cal['tolerances'][profile])

@app.route('/calibrate/z_heights', methods=['POST'])
def calibrate_z_heights():
    data = request.get_json() or {}
    set_keys = []
    for key in ('handle_depth_ms', 'clear_lift_ms'):
        if key in data:
            val = data[key]
            val = int(val) if val not in (None, '') else None
            brain.set_z_height(key, val)
            set_keys.append(f"{key}={val}")
    return jsonify(result="Z heights set: " + (", ".join(set_keys) if set_keys else "(nothing)"))

@app.route('/calibrate/marker_label', methods=['POST'])
def calibrate_marker_label():
    data = request.get_json()
    mid = str(int(data.get('id')))
    label = (data.get('label') or '').strip()
    if label:
        brain.cal['marker_labels'][mid] = label
    else:
        brain.cal['marker_labels'].pop(mid, None)   # empty label clears the entry
    brain.save_calibration()
    return jsonify(result=f"Marker {mid} label set to '{label}'" if label else f"Marker {mid} label cleared")

@app.route('/set_flip', methods=['POST'])
def set_flip():
    # Flip applies at the frame source (camera loop) so vision + feed stay in agreement.
    # If no 'flip' given, toggle the current value.
    data = request.get_json() or {}
    if 'flip' in data:
        val = bool(data['flip'])
    else:
        val = not bool(brain.cal.get('flip_feed'))
    brain.cal['flip_feed'] = val
    brain.save_calibration()
    return jsonify(result=f"Feed flip {'ON' if val else 'OFF'}", flip_feed=val)

@app.route('/set_settle_time', methods=['POST'])
def set_settle_time():
    """Set the post-move settle margin (seconds) — extra wait after a move finishes before the
    camera analyzes a new frame. Persisted to calibration.json."""
    data = request.get_json() or {}
    val = brain.set_settle_time(data.get('settle_time_s', brain.settle_time_s))
    return jsonify(result=f"Settle time set to {val}s", settle_time_s=val)

@app.route('/set_z_speed', methods=['POST'])
def set_z_speed():
    """Recalibrate the Z actuator speed (mm/s). Single source of truth — fixes the dashboard Z
    readout and the Z alignment gain together. Persisted to calibration.json."""
    data = request.get_json() or {}
    val = brain.set_z_speed(data.get('z_mm_per_s', brain.z_mm_per_s))
    return jsonify(result=f"Z speed set to {val} mm/s", z_mm_per_s=val)

@app.route('/target_slot/set_down/<int:marker_id>')
def target_slot_set_down(marker_id):
    if not camera_available:
        return jsonify(error="Camera not available"), 503
    brain.target_slot_set_down(marker_id)
    return jsonify(result=f"Target Slot Set Down started for marker ID {marker_id}")


if __name__ == '__main__':
    t = threading.Thread(target=_automation_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
