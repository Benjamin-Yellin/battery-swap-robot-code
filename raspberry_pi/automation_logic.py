import time
import json
import os

_CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json')

# Per-operation calibration. Pick-up references the BATTERY-case marker (scoop to handle);
# set-down references the SLOT-frame marker (carried battery to slot opening). These are
# different physical references, so each gets its own crosshair + Y-stop width. The marker
# PHYSICAL size is the same for both, so MARKER_PHYSICAL_WIDTH_MM stays a shared constant.
_EMPTY_PROFILE = {
    "crosshair_x": None, "crosshair_y": None,  # marker centre to align X to (the aim point)
    "target_width_px": None,                    # marker width at the Y-stop depth (the approach target)
    "aim_corners": None,                        # [[x,y]*4] full outline captured at the aim pose
    "aim_orient": None,                         # {roll,pitch,yaw} at the aim pose — per-profile orient target
    "aim_width_px": None,                       # marker width at the aim pose (pickup: pre-engage depth)
    "engage_corners": None,                     # [[x,y]*4] full outline at the engagement pose (pickup)
    # ── Set-down profile only: the "seated pose" — where the slot flag marker sits in-frame once the
    # battery is fully seated. Captured as the second set-down reference and used as the descent guard
    # (stop lowering if the marker reaches this without the grip releasing) + a placement sanity check.
    # Stays None for the pick-up profile (it never seats against a slot).
    "seated_x": None, "seated_y": None,         # marker centre at the seated pose
    "seated_width_px": None,                    # marker width at the seated pose
    "seated_corners": None,                     # [[x,y]*4] full outline at the seated pose (overlay)
}
_EMPTY_Z_HEIGHTS = {"handle_depth_ms": None, "clear_lift_ms": None}
# Extra wait (s) after a move actually finishes, for mechanical vibration to damp before the camera
# analyzes a new frame. Tunable in the UI; this is the default for a fresh calibration.
_DEFAULT_SETTLE_TIME_S = 0.6
# ms of actuator run-time per mm of Z travel (lab-calibrated: 56 mm over 20 000 ms). The DC actuator's
# speed drifts with load/wear, so this is UI-tunable and persisted — and it's the SINGLE source of
# truth: the dashboard Z readout, move_z_to, and the Z alignment gain all derive from it.
_DEFAULT_Z_MS_PER_MM = 20000.0 / 56.0   # ≈ 357.14 ms/mm
# Per-operation alignment tolerances (how close counts as "aligned"). Defaults below.
_DEFAULT_TOL = {"x_px": 50, "z_px": 50, "width_px": 10,
                "roll_deg": 10.0, "pitch_deg": 10.0, "yaw_deg": 10.0}
_DEFAULT_CAL = {
    "pickup":  dict(_EMPTY_PROFILE),
    "setdown": dict(_EMPTY_PROFILE),
    "marker_labels": {},   # {"<id>": "human label"} — for on-feed debugging labels
    "z_heights": dict(_EMPTY_Z_HEIGHTS),   # lab-measured scoop Z travel for slot operations
    "tolerances": {"pickup": dict(_DEFAULT_TOL), "setdown": dict(_DEFAULT_TOL)},
    "flip_feed": False,    # rotate the camera 180° at the source (for an upside-down mount) — see web_server
    "settle_time_s": _DEFAULT_SETTLE_TIME_S,  # extra post-move wait for vibration to damp (UI-tunable)
    "z_ms_per_mm": _DEFAULT_Z_MS_PER_MM,      # Z speed calibration (UI-tunable; single source of truth)
}


class AutomationLogic:

    # ─── States ───────────────────────────────────────────────────────────────
    STATE_IDLE                = "IDLE"
    STATE_SEARCHING           = "SEARCHING"
    STATE_ALIGNING_XZ         = "ALIGNING_XZ"       # X + Z only, then stop
    STATE_PICKING_UP          = "PICKING_UP"         # vision-guided: Y in → Z lift
    STATE_OUR_PICKING_UP      = "OUR_PICKING_UP"     # our slot: X → (XZ align) → Z lower → Y in → Z lift → Y out
    STATE_OUR_SETTING_DOWN    = "OUR_SETTING_DOWN"   # our slot: X → Y in → Z lower (seating) → Z raise → Y out
    STATE_SETTING_DOWN_TARGET = "SETTING_DOWN_TARGET" # target slot: search → XZ align → Y → Z lower (seating) → Y out
    STATE_DEMO_FOLLOW         = "DEMO_FOLLOW"        # continuous XZ tracking, demo only

    # ─── Camera / physics ─────────────────────────────────────────────────────
    MARKER_PHYSICAL_WIDTH_MM = 40.0
    FOCAL_LENGTH_PX          = 1193.7
    MM_PER_STEP_XY           = 1.005  # (2π × 32mm) / 200 steps — canonical mm/step for X and Y

    # ─── Gantry speeds ────────────────────────────────────────────────────────
    OPERATING_SPEED_SPS      = 85      # steps/s — must match Arduino OPERATING_SPEED
    OPERATING_ACCEL_SPS2     = 80      # steps/s² — mirror of Arduino OPERATING_ACCEL (move-time estimate)
    SETTLE_TIMEOUT_FACTOR    = 2.5     # wait up to (estimate × this + margin) for DONE before giving up

    # ─── Loaded (held-battery) motion profile ─────────────────────────────────
    # A set-down carries a ~30 kg battery cradled on the scoop — a pendulum. Sharp accel makes it
    # swing, loading the arm and threatening dropped steps, so every move WHILE CARRYING uses this
    # gentler accel/speed (applied via the 'A'/'S' serial commands at op start, restored at the end).
    # The op also retracts Z fully first, shortening the pendulum before any horizontal motion.
    # TUNE IN LAB — start conservative.
    LOADED_SPEED_SPS         = 40      # steps/s while holding a battery (clamped 10–200 by firmware)
    LOADED_ACCEL_SPS2        = 30      # steps/s² while holding — the swing matters most here

    # ─── Our robot: slot positions ────────────────────────────────────────────
    SLOT_X_STEPS             = [12, 295, 580]  # X positions for slots 1, 2, 3 (steps ≈ mm)
    Y_DOCKING_STEPS          = 85              # Y depth to engage battery handle

    # ─── Our robot: Z calibration (set in the web UI "Scoop Z Heights" card; persisted to ──
    #      calibration.json and loaded into these instance attrs by _apply_z_heights). The class
    #      defaults below are the unmeasured state — slot operations are gated until they're set.
    Z_HANDLE_DEPTH_MS        = None    # ms to extend from Z-home until the scoop cradles the handle
    Z_CLEAR_LIFT_MS          = None    # ms to lift a cradled battery from handle height fully clear of
                                       #   the slot bracing (also reused as the our-slot set-down depth cap)

    # ─── Target robot: Z pick-up (vision-guided, scoop already at height) ────
    Z_RETRACT_PICKUP_MS      = 13000    # ms to retract after Y-approach — TUNE IN LAB

    # ─── Z alignment (vision-guided correction pulses) ────────────────────────
    # The gain is split into its speed-independent part (mm of Z travel per px of error, at the
    # calibration depth) and the Z speed constant (z_ms_per_mm, persisted/UI-tunable). Keeping the
    # gain in mm means recalibrating Z speed fixes the dashboard readout AND the alignment together,
    # so a drifting actuator speed can't silently desync them. (≈0.126 mm/px ≡ the old 45 ms/px.)
    Z_MM_PER_PX              = 45.0 / _DEFAULT_Z_MS_PER_MM
    MIN_Z_PULSE_MS           = 100
    MAX_Z_PULSE_MS           = 1000    # fine cap: keeps near-target Z pulses from overshooting
    MAX_Z_PULSE_COARSE_MS    = 6000    # coarse cap: far away with a huge unambiguous z_err there's no
                                       #   overshoot risk (pulse-and-verify self-corrects), so move a
                                       #   big chunk per pulse instead of crawling in 1 s capped steps

    # ─── Z seating detection (used by both set-down handlers) ─────────────────
    # Seating is detected by the end-effector grip switch: while carrying, the battery's weight
    # holds the switch engaged; once the battery settles into the slot the scoop drops free and
    # the switch releases (engaged → disengaged). The scoop is then freed by retracting Y, never
    # by raising Z (which would re-cradle the handle and lift the battery back out).
    Z_SET_DOWN_PULSE_MS       = 300    # ms per incremental Z extension pulse
    # Safety fallback for the TARGET set-down only: the most we'll lower past the vision-aligned
    # approach pose before backing off, in case the battery jams without ever seating (the grip
    # switch never releases). RELATIVE to the aligned pose — so it needs no knowledge of the
    # target's (per-visit, vision-found) absolute handle height, and triggering from an unexpected
    # Z can't drive the scoop further than this past where vision parked it. Grip release is the
    # normal stop; this only catches a never-seats jam. Conservative starting value — tune in lab.
    Z_SETDOWN_GUARD_MS        = 6000

    # ─── Orientation reference (idle fallback target only) ────────────────────
    # The LIVE per-operation orientation target is the captured profile aim_orient; these are just
    # the fallback when no profile is loaded (see orient_target). Tolerances are per-profile and
    # live entirely in _DEFAULT_TOL / active_tol — there are intentionally no global tolerance
    # constants here, so there's no dead knob that looks editable but isn't.
    ORIENT_TARGET_ROLL_DEG   = 0.0
    ORIENT_TARGET_PITCH_DEG  = 0.0
    ORIENT_TARGET_YAW_DEG    = 0.0

    # ─── Move limits ──────────────────────────────────────────────────────────
    MAX_STEP_PER_MOVE        = 200     # safety cap per X/Y move command (~200mm)
    MIN_STEP_PER_MOVE        = 5      # minimum step to overcome friction
    MAX_ALIGN_MOVES          = 40      # per-phase cap on corrective moves before aborting — a loop
                                       #   that never converges (camera noise oscillating the error
                                       #   across the tolerance boundary) hits this instead of hanging
    MAX_APPROACH_MOVES       = 30      # per-approach cap on Y depth moves — separate from the XZ
                                       #   align budget so a width that never converges aborts with
                                       #   its own (WIDTH-REF-calibration-pointing) reason
    MAX_APPROACH_STEP        = 40      # per-move cap (steps ≈ mm) on a single Y approach increment.
                                       #   Far smaller than MAX_STEP_PER_MOVE: the approach is open-
                                       #   loop between camera frames, so one big lunge can overshoot
                                       #   the handle and drop the marker out of the downward FOV.
                                       #   Capping it makes the camera re-evaluate every small step.
    COARSE_X_TOL_MAX_SCALE   = 4.0     # cap on how much looser the COARSE-align X tolerance gets at
                                       #   distance (see _coarse_x_tol) — fine align still uses x_px

    # ─── Search behaviour ─────────────────────────────────────────────────────
    SEARCH_STEP_SIZE         = 100     # steps to pulse X during search sweep
    SEARCH_COOLDOWN_S        = 3.5     # s to wait between search pulses (covers move + settle)
    MAX_SEARCH_STEPS         = 10      # give up after this many X pulses

    # ─── Vision timing ────────────────────────────────────────────────────────
    MAX_LOST_FRAMES          = 12      # frames without detection before re-searching

    # ------------------------------------------------------------------ #
    #  Init                                                                #
    # ------------------------------------------------------------------ #

    def __init__(self, camera_object, messenger_object):
        self.eyes      = camera_object
        self.messenger = messenger_object

        self.current_state = self.STATE_IDLE
        self.target_id     = None
        self.is_aligned    = False
        self.last_x_err    = 0
        self.last_z_err    = 0

        self._seq_step      = 0
        self._seq_slot_idx  = 0
        self._active_orient = None        # orientation target of the active operation's profile
        self._active_tol    = None        # alignment tolerances of the active operation's profile
        self._search_prepped = False      # has the current search done its X=0 / Z-up prep?
        self._search_prep_cooldown = self.SEARCH_COOLDOWN_S  # first-pulse wait, sized in prep

        self.last_search_time = 0
        self.search_count     = 0
        self.last_align_time  = 0        # when the last move was commanded (settle-timeout anchor)
        self._move_expected_s = 0.0      # accel-aware estimate of that move's duration
        self._move_done_at    = None     # time the Arduino first reported the move finished
        self._settling        = False    # True while waiting out a move's DONE + settle margin
        self.lost_frames      = 0
        self._align_moves     = 0         # corrective moves issued in the current alignment phase
        self._approach_moves  = 0         # Y depth moves issued in the current approach-to-width phase

        # ─── Step mode (slow-mo, approve-each-move — see _gate) ─────────────────
        # When step_mode is on, the pick-up's physical moves (from coarse align onward) are not
        # executed immediately: each is staged as a proposal the operator must Approve before it
        # runs, so nothing moves without a green light. Search runs autonomously (scoop is up, low
        # collision risk). approve_step() runs the staged move; abort_step() cancels the op.
        self.step_mode         = False
        self.awaiting_approval = False
        self._pending_action   = None     # callable that performs the staged move + its cooldown
        self._pending_desc     = ""        # human-readable description of the staged move

        # Per-operation calibration (pickup / setdown profiles + marker labels + z-heights)
        self.cal = self._load_calibration()
        self._apply_z_heights()           # populate instance Z_* height attrs from cal
        self._apply_z_speed()             # push persisted Z speed into the messenger (single source)

        # Set-down sequencing state (seating is now confirmed by the grip limit switch)
        self._z_set_down_total_ms = 0     # accumulated Z extension during seating step
        self._prev_grip_engaged   = None  # grip state on the previous seating tick

        # Gentle "loaded" motion profile (active while carrying a battery — see _enter/_exit_loaded
        # _motion). Saved values restore whatever speed/accel was in effect before the op, so a UI
        # speed change isn't clobbered.
        self._loaded_motion_active = False
        self._saved_speed = None
        self._saved_accel = None
        self._rehome_started = False      # our-slot set-down: has its re-home been kicked off?

    def profile(self, name):
        """Return the calibration profile dict ('pickup' or 'setdown')."""
        return self.cal[name]

    @property
    def settle_time_s(self):
        """Extra wait (s) after a move finishes before analyzing a new frame (vibration damping)."""
        return self.cal.get('settle_time_s', _DEFAULT_SETTLE_TIME_S)

    def set_settle_time(self, value):
        """Set & persist the post-move settle margin (clamped to a sane 0–5 s)."""
        self.cal['settle_time_s'] = round(max(0.0, min(5.0, float(value))), 2)
        self.save_calibration()
        return self.cal['settle_time_s']

    @property
    def z_ms_per_mm(self):
        """Z speed calibration (ms of run-time per mm). Single source of truth: the dashboard Z,
        move_z_to, and the Z alignment gain all derive from this."""
        return self.cal.get('z_ms_per_mm', _DEFAULT_Z_MS_PER_MM)

    @property
    def z_mm_per_s(self):
        """Z speed in mm/s — the intuitive form shown and edited in the UI."""
        return round(1000.0 / self.z_ms_per_mm, 2)

    def _apply_z_speed(self):
        """Keep the messenger's position-tracking constant in sync with the persisted Z speed."""
        if hasattr(self.messenger, 'set_z_ms_per_mm'):
            self.messenger.set_z_ms_per_mm(self.z_ms_per_mm)

    def set_z_speed(self, mm_per_s):
        """Set & persist Z speed from an intuitive mm/s value (clamped). Recalibrating this one
        number fixes the dashboard readout AND the Z alignment gain together."""
        v = max(0.5, min(200.0, float(mm_per_s)))
        self.cal['z_ms_per_mm'] = round(1000.0 / v, 4)
        self.save_calibration()
        self._apply_z_speed()
        return self.z_mm_per_s

    def cal_ready(self, name):
        """True when a profile has both its crosshair and Y-stop width set."""
        p = self.cal[name]
        return p['crosshair_x'] is not None and p['target_width_px'] is not None

    def seated_ready(self, name):
        """True when a profile's optional seated-pose reference is captured (set-down descent guard).
        When present, the set-down stops lowering once the marker reaches the seated pose; when
        absent it falls back to the relative travel guard (Z_SETDOWN_GUARD_MS)."""
        p = self.cal[name]
        return p.get('seated_y') is not None and p.get('seated_width_px') is not None

    # ─── Gentle loaded-motion profile (active while carrying a battery) ─────────
    def _enter_loaded_motion(self):
        """Drop accel/speed to the gentle held-battery profile so a ~30 kg load doesn't swing.
        Saves the current values first so _exit restores whatever the operator had set."""
        if self._loaded_motion_active:
            return
        self._saved_speed = getattr(self.messenger, 'speed', None)
        self._saved_accel = getattr(self.messenger, 'acceleration', None)
        self.messenger.set_acceleration(self.LOADED_ACCEL_SPS2)
        self.messenger.set_speed(self.LOADED_SPEED_SPS)
        self._loaded_motion_active = True

    @property
    def loaded_motion(self):
        """True while the gentle held-battery motion profile is active (for the UI 'Gentle' pill)."""
        return self._loaded_motion_active

    def _exit_loaded_motion(self):
        """Restore the pre-op accel/speed. Safe to call unconditionally (no-op if not loaded)."""
        if not self._loaded_motion_active:
            return
        if self._saved_accel is not None:
            self.messenger.set_acceleration(self._saved_accel)
        if self._saved_speed is not None:
            self.messenger.set_speed(self._saved_speed)
        self._loaded_motion_active = False

    def reset_profile(self, name):
        """Clear a calibration profile back to empty (crosshair/outlines/width/orientation)."""
        if name not in ('pickup', 'setdown'):
            return False
        self.cal[name] = dict(_EMPTY_PROFILE)
        self.save_calibration()
        return True

    # ─── Scoop Z heights (lab-measured, required for slot operations) ──────────
    def _apply_z_heights(self):
        """Copy persisted z-heights into the instance attrs the handlers read."""
        z = self.cal['z_heights']
        self.Z_HANDLE_DEPTH_MS = z['handle_depth_ms']
        self.Z_CLEAR_LIFT_MS   = z['clear_lift_ms']

    @property
    def z_ready(self):
        """True when the our-slot pick-up / set-down Z travel heights are measured."""
        return self.Z_HANDLE_DEPTH_MS is not None and self.Z_CLEAR_LIFT_MS is not None

    def set_z_height(self, key, value):
        """Set & persist one scoop Z-height ('handle_depth_ms'|'clear_lift_ms')."""
        if key not in self.cal['z_heights']:
            return False
        self.cal['z_heights'][key] = value
        self._apply_z_heights()
        self.save_calibration()
        return True

    # ─── Alignment tolerances (per-operation; how close counts as aligned) ─────
    @property
    def active_tol(self):
        """The tolerance set of the active operation's profile, else the defaults."""
        return self._active_tol if self._active_tol is not None else _DEFAULT_TOL

    def _coarse_x_tol(self, marker_w):
        """Coarse-align X tolerance, widened with distance. Coarse align only needs a rough lineup
        good enough to begin the Y approach — demanding the tight fine tolerance from far away is both
        unachievable (one min gantry step spans many pixels at depth) and unsafe: at distance much of
        the horizontal pixel error is really *forward* offset (perspective), not lateral, so chasing
        it walks the carriage into the X_MAX stop. So when the marker is farther than the pre-engage
        (close) depth we loosen X proportionally, and let the fine-align stage — which runs after the
        Y approach, up close, at the tight x_px tolerance — do the precision. Falls back to the tight
        tolerance when the pre-engage width isn't calibrated (no reference depth to scale from)."""
        base = self.active_tol['x_px']
        aim_w = self.profile('pickup').get('aim_width_px')
        if not aim_w or marker_w <= 0:
            return base
        # marker_w shrinks with distance; aim_w is the width at the close pre-engage pose, so aim_w /
        # marker_w ≈ how many times farther than pre-engage the marker is. Never tighter than base.
        scale = min(self.COARSE_X_TOL_MAX_SCALE, max(1.0, aim_w / marker_w))
        return base * scale

    @property
    def active_profile(self):
        """Which calibration profile the current operation uses ('pickup'/'setdown'), else None.
        Used by the feed to draw that profile's tolerance overlays during an operation."""
        if self.current_state == self.STATE_SETTING_DOWN_TARGET:
            return 'setdown'
        if self.current_state in (self.STATE_SEARCHING, self.STATE_ALIGNING_XZ, self.STATE_PICKING_UP,
                                  self.STATE_OUR_PICKING_UP, self.STATE_DEMO_FOLLOW):
            return 'pickup'
        return None

    # Ordered phase lists per operation (for the on-feed progress banner). The active phase is
    # derived from current_state + _seq_step in op_phase below.
    _PH_TARGET_PICKUP  = ["SEARCH", "COARSE", "APPROACH", "FINE", "ENGAGE", "LIFT"]
    _PH_TARGET_SETDOWN = ["RAISE", "SEARCH", "ALIGN", "LOWER", "RETRACT"]
    _PH_OUR_PICKUP     = ["X-MOVE", "REFINE", "LOWER", "ENGAGE", "LIFT", "RETRACT"]
    _PH_OUR_SETDOWN    = ["RE-HOME", "X-MOVE", "ENGAGE", "LOWER", "RETRACT"]

    @property
    def op_phase(self):
        """For the on-feed banner: (ordered_phase_labels, active_index) for the running op, else
        None when IDLE. Keeps the state→phase mapping here in the state machine, not the view."""
        s, st = self.current_state, self._seq_step
        if s == self.STATE_SEARCHING:
            return (self._PH_TARGET_PICKUP, 0)
        if s == self.STATE_ALIGNING_XZ:
            return (self._PH_TARGET_PICKUP, 1)
        if s == self.STATE_PICKING_UP:
            # step 0 approach, 1 fine, 2 engage, 3/4 lift → indices 2..5 of _PH_TARGET_PICKUP
            return (self._PH_TARGET_PICKUP, min(st + 2, len(self._PH_TARGET_PICKUP) - 1))
        if s == self.STATE_SETTING_DOWN_TARGET:
            return (self._PH_TARGET_SETDOWN, min(st, len(self._PH_TARGET_SETDOWN) - 1))
        if s == self.STATE_OUR_PICKING_UP:
            return (self._PH_OUR_PICKUP, min(st, len(self._PH_OUR_PICKUP) - 1))
        if s == self.STATE_OUR_SETTING_DOWN:
            return (self._PH_OUR_SETDOWN, min(st, len(self._PH_OUR_SETDOWN) - 1))
        if s == self.STATE_DEMO_FOLLOW:
            return (["TRACK"], 0)
        return None

    _TOL_BOUNDS = {"x_px": (2, 400), "z_px": (2, 400), "width_px": (1, 200),
                   "roll_deg": (0.5, 90), "pitch_deg": (0.5, 90), "yaw_deg": (0.5, 90)}

    def set_tolerances(self, profile, values):
        """Set & persist one profile's alignment tolerances (clamped to sane bounds)."""
        if profile not in ('pickup', 'setdown'):
            return False
        tol = self.cal['tolerances'][profile]
        for k, (lo, hi) in self._TOL_BOUNDS.items():
            if k in values and values[k] is not None:
                v = max(lo, min(hi, float(values[k])))
                tol[k] = int(v) if k.endswith('_px') else round(v, 1)
        self.save_calibration()
        return True

    def reset_tolerances(self, profile):
        """Reset one profile's tolerances to the defaults."""
        if profile not in ('pickup', 'setdown'):
            return False
        self.cal['tolerances'][profile] = dict(_DEFAULT_TOL)
        self.save_calibration()
        return True

    def label_for(self, mid):
        """Human-readable role label for a detected marker ID (for on-feed debugging)."""
        if self.target_id is not None and mid == self.target_id:
            base = self._base_label(mid)
            return f"> {base} (active)"   # ASCII '>' — OpenCV's font can't render a unicode triangle
        return self._base_label(mid)

    def _base_label(self, mid):
        custom = self.cal['marker_labels'].get(str(mid))
        if custom:
            return custom
        return f"ID {mid}"

    # ------------------------------------------------------------------ #
    #  Main cycle                                                          #
    # ------------------------------------------------------------------ #

    def run_cycle(self, frame):
        self.eyes.last_corners = []
        self.eyes.last_orientation = None

        if self.messenger.is_limit_switch_pressed():
            self.current_state = self.STATE_IDLE
            return "SAFETY STOP: Limit Hit!"

        # Step mode: a move is staged and waiting for the operator. Don't dispatch (which would
        # recompute and re-stage); just hold, showing what's proposed, until Approve/Abort/Resume.
        if self.awaiting_approval:
            return self._pending_desc

        match self.current_state:
            case self.STATE_IDLE:
                self._active_orient = None   # no operation loaded → targets/tolerances revert to defaults
                self._active_tol = None
                # Any path back to IDLE (completion or abort) restores the normal motion profile, so a
                # gentle loaded profile can never leak into the next (empty) operation.
                self._exit_loaded_motion()
                # Clear the active-target overlays so the feed goes calm between ops: an aborted /
                # completed operation otherwise leaves target_id + the aim crosshair set, making the
                # marker keep its red box, arrow and X/Z/W/TILT chips as if an op were still running.
                # (The sticky abort banner separately preserves *why* the last op ended.)
                if self.target_id is not None:
                    self.target_id = None
                    self.eyes.set_target(None, None)
                return "System Ready."
            case self.STATE_SEARCHING:
                return self._run_searching_logic(frame)
            case self.STATE_ALIGNING_XZ:
                return self._run_aligning_xz_logic(frame)
            case self.STATE_PICKING_UP:
                return self._run_picking_up_logic(frame)
            case self.STATE_OUR_PICKING_UP:
                return self._run_our_picking_up_logic(frame)
            case self.STATE_OUR_SETTING_DOWN:
                return self._run_our_setting_down_logic(frame)
            case self.STATE_SETTING_DOWN_TARGET:
                return self._run_setting_down_target_logic(frame)
            case self.STATE_DEMO_FOLLOW:
                return self._run_demo_follow_logic(frame)

    # ------------------------------------------------------------------ #
    #  Public API (called by Flask routes)                                 #
    # ------------------------------------------------------------------ #

    def demo_follow(self, target_id):
        """Continuously track a marker — corrects X+Z indefinitely, ignores orientation."""
        if not self.homing_ready():
            return
        self.is_aligned = False
        self.current_state = self.STATE_DEMO_FOLLOW
        self.target_id = target_id
        self._active_tol = self.cal['tolerances']['pickup']
        self._reset_search()
        print(f"Logic: Demo Follow for ID {target_id}")

    def homing_ready(self):
        """True only when homing is fully complete — axes referenced and no HOME in flight.
        Autonomous ops gate on this so their moves can't start (or interleave) mid-homing."""
        return self.messenger.has_homed and not self.messenger.is_homing

    def pick_up_target(self, target_id):
        """One-button autonomous pick-up of a target battery by its marker ID.

        Runs the whole staged sequence: search (from X=0, Z up) → coarse XZ align → approach to the
        pre-engage depth → fine XZ re-align + orientation check → approach to full engagement →
        lift → grip confirm. Loads the pick-up crosshair + captured orientation target."""
        if self.current_state != self.STATE_IDLE or not self.homing_ready():
            return
        self.is_aligned = False
        p = self.profile('pickup')
        if self.cal_ready('pickup'):
            self.eyes.set_target(p['crosshair_x'], p['crosshair_y'])
        self._active_orient = p['aim_orient']
        self._active_tol = self.cal['tolerances']['pickup']
        self.current_state = self.STATE_SEARCHING
        self.target_id = target_id
        self._reset_search()
        print(f"Logic: Pick Up Battery (marker ID {target_id})")

    @property
    def orient_target(self):
        """Active orientation target — the captured per-profile pose if set, else global defaults."""
        if self._active_orient:
            return self._active_orient
        return {'roll':  self.ORIENT_TARGET_ROLL_DEG,
                'pitch': self.ORIENT_TARGET_PITCH_DEG,
                'yaw':   self.ORIENT_TARGET_YAW_DEG}

    @property
    def is_orientation_ok(self):
        """True only when the last seen marker is within tolerance of the active orientation target."""
        orient = self.eyes.last_orientation
        if orient is None:
            return False
        t = self.orient_target
        tol = self.active_tol
        return (abs(orient['roll']  - t['roll'])  <= tol['roll_deg']  and
                abs(orient['pitch'] - t['pitch']) <= tol['pitch_deg'] and
                abs(orient['yaw']   - t['yaw'])   <= tol['yaw_deg'])

    def our_slot_pick_up(self, slot_idx):
        """Pick up from our robot's slot (vision XZ+Y align if calibrated, else hardcoded)."""
        if self.current_state == self.STATE_IDLE and self.homing_ready():
            self._seq_slot_idx = slot_idx
            self._seq_step = 0
            self._active_tol = self.cal['tolerances']['pickup']
            self.current_state = self.STATE_OUR_PICKING_UP
            print(f"Logic: Our slot {slot_idx + 1} Pick Up")

    def our_slot_set_down(self, slot_idx):
        """Set down into our robot's slot — no vision (no markers on our robot). We re-home X/Y with
        the battery first, then dead-reckon to the known slot location and lower until the grip
        switch releases. Seating via grip switch (firmware self-aborts the lower the instant it seats).

        Why re-home: X/Y are open-loop and a swinging ~30 kg load is exactly what drops steps, so by
        set-down time the tracked position is least trustworthy. Homing retracts Z first (battery to
        full-up, shortest pendulum, max ground clearance) before any horizontal traverse."""
        if self.current_state == self.STATE_IDLE and self.homing_ready():
            self._seq_slot_idx = slot_idx
            self._seq_step = 0
            self._z_set_down_total_ms = 0
            self._prev_grip_engaged = None
            self._rehome_started = False       # handler step 0 kicks off the re-home (in the brain
            self.current_state = self.STATE_OUR_SETTING_DOWN   # thread, so the IDLE branch can't race
            print(f"Logic: Our slot {slot_idx + 1} Set Down (re-home + dead-reckon)")

    def target_slot_set_down(self, slot_marker_id):
        """Align with target slot marker, lower battery (grip-switch seating), retract."""
        if self.current_state == self.STATE_IDLE and self.homing_ready():
            self.is_aligned = False
            self.target_id = slot_marker_id
            self._seq_step = 0
            self._z_set_down_total_ms = 0
            self._prev_grip_engaged = None
            self._active_orient = self.profile('setdown')['aim_orient']
            self._active_tol = self.cal['tolerances']['setdown']
            self._reset_search()
            self.current_state = self.STATE_SETTING_DOWN_TARGET
            print(f"Logic: Target Slot Set Down for slot marker ID {slot_marker_id}")

    # ------------------------------------------------------------------ #
    #  State handlers                                                      #
    # ------------------------------------------------------------------ #

    def _run_searching_logic(self, frame):
        # Prep once: retract Z (camera needs the scoop up) and return X to 0 so the +X sweep
        # covers the whole rail (otherwise a marker behind the start position is never seen).
        if not self._search_prepped:
            self._search_prepped = True
            self.messenger.move_z_to(0)
            x_travel = abs(self.messenger.x_pos)
            self.messenger.move_gantry(0, self.messenger.y_pos)
            # Hold the +X sweep until this return-to-0 move actually finishes. On real hardware
            # x_pos only commits to 0 when the Arduino replies DONE, so if we pulsed +X now we'd
            # read the stale old X and command G (oldX+100) — driving the gantry away from 0.
            self.last_search_time = time.time()
            self._search_prep_cooldown = max(self.SEARCH_COOLDOWN_S,
                                             x_travel / self.OPERATING_SPEED_SPS + 1.0)
            return "Search: returning to X=0, retracting Z..."

        found, *_ = self.eyes.get_target_error(frame, self.target_id)
        if found:
            self.search_count = 0
            self._align_moves = 0
            self.current_state = self.STATE_ALIGNING_XZ
            return "Target Found! Starting Alignment."

        # First sweep pulse waits for the prep return move; later pulses use the normal cooldown.
        cooldown = self._search_prep_cooldown if self.search_count == 0 else self.SEARCH_COOLDOWN_S
        current_time = time.time()
        if (current_time - self.last_search_time) < cooldown:
            return f"Searching... (Step {self.search_count})"

        if self.search_count < self.MAX_SEARCH_STEPS:
            self.search_count += 1
            self.last_search_time = current_time
            self.messenger.move_gantry_x(self.SEARCH_STEP_SIZE)
            return f"Searching... Pulse X (Step {self.search_count})"

        self.current_state = self.STATE_IDLE
        return "Search Failed: Marker not found."

    def _run_aligning_xz_logic(self, frame):
        """X+Z alignment only — stops when both within tolerance."""
        found, x_err, z_err, marker_w, mx, my = self.eyes.get_target_error(frame, self.target_id)
        self.last_x_err, self.last_z_err = x_err, z_err

        if not found:
            self.lost_frames += 1
            if self.lost_frames > self.MAX_LOST_FRAMES:
                self.is_aligned = False
                self.current_state = self.STATE_SEARCHING
                return "Marker lost. Re-searching..."
            return "Vision Flicker..."
        self.lost_frames = 0

        current_time = time.time()
        if not self._move_settled():
            return "Waiting for move to finish..."

        steps = self._physics_px_to_steps(x_err, marker_w) if abs(x_err) > self._coarse_x_tol(marker_w) else 0
        if steps != 0:
            # If we're jammed at X_MAX and still want +X, don't abort — _x_reach_abort returns 0 steps
            # (abort_on_max=False) so we skip the X move and fall through to Z align + the Y approach.
            # The out-of-reach decision is deferred to fine align at pre-engage distance.
            steps, reach = self._x_reach_abort(steps, "coarse align", abort_on_max=False)
            if reach: return reach
        if steps != 0:
            abort = self._align_move_guard("coarse align")
            if abort: return abort
            staged = self._gate(lambda: (self._set_cooldown_steps(steps), self.messenger.move_gantry_x(steps)),
                                f"COARSE align X: move {steps:+d} steps (x_err={x_err}px)")
            if staged: return staged
            return f"Aligning X (err={x_err}px, move={steps}steps)"

        if abs(z_err) > self.active_tol['z_px']:
            abort = self._align_move_guard("coarse align")
            if abort: return abort
            pulse_ms = self._z_pulse_ms(z_err, max_ms=self.MAX_Z_PULSE_COARSE_MS)
            pulse_ms, reach = self._z_reach_abort(pulse_ms, "coarse align")
            if reach: return reach
            staged = self._gate(lambda: (self._set_cooldown_ms(pulse_ms), self.messenger.pulse_lift_z(pulse_ms)),
                                f"COARSE align Z: pulse {pulse_ms:+d} ms (z_err={z_err}px)")
            if staged: return staged
            return f"Aligning Z (err={z_err}px, pulse={pulse_ms}ms)"

        # Coarse alignment done — flow straight into the staged pick-up (no stop, no orientation
        # gate here; orientation is checked at the close fine-align stage).
        self.is_aligned = True
        self._seq_step = 0
        self._align_moves = 0
        self._approach_moves = 0      # entering the pre-engage Y approach (seq_step 0)
        self.current_state = self.STATE_PICKING_UP
        return "Coarse aligned — approaching target..."

    def _approach_to_width(self, frame, target_width, recenter=True):
        """One Y-approach tick toward a target marker width. Returns a status string while still
        approaching (or while re-centering / briefly lost), or None once within tolerance (arrived).
        Aborts the whole op (→ IDLE, returns 'ABORT:...') if the marker is lost for too long.

        Driving Y translates the marker within the downward FOV, so the pre-engage approach
        (recenter=True) re-centers X/Z each tick to keep the marker in view all the way in. The
        final engagement drive (recenter=False) must NOT: the scoop is already entering the handle,
        the X/Z reference belongs to the pre-engage pose (the marker has legitimately translated by
        the engagement depth), and correcting X here just jams the scoop into the handle wall."""
        found, x_err, z_err, marker_w, _, _ = self.eyes.get_target_error(frame, self.target_id)
        if not found:
            self.lost_frames += 1
            if self.lost_frames > self.MAX_LOST_FRAMES:
                self.lost_frames = 0
                self._seq_step = 0
                self.current_state = self.STATE_IDLE
                return "ABORT: Pick Up — marker lost during approach."
            return "Pick Up: approach, searching..."
        self.lost_frames = 0
        self.last_x_err, self.last_z_err = x_err, z_err

        if recenter:
            msg = self._approach_recenter(x_err, z_err, marker_w)
            if msg is not None:
                return msg

        y_err_w = target_width - marker_w
        if abs(y_err_w) > self.active_tol['width_px']:
            steps = self._physics_depth_to_steps_y_with_target(marker_w, target_width)
            # Early-out: if the move is blocked by the travel limit it's pushing into, the width can
            # never converge — abort now instead of spinning MAX_APPROACH_MOVES no-op moves.
            blocked = self._approach_blocked_abort(steps)
            if blocked: return blocked
            abort = self._approach_move_guard()
            if abort: return abort
            # Cap the increment so one open-loop move can't lunge past the handle and lose the marker.
            steps = max(-self.MAX_APPROACH_STEP, min(self.MAX_APPROACH_STEP, steps))
            staged = self._gate(lambda: (self._set_cooldown_steps(steps), self.messenger.move_gantry_y(steps)),
                                f"APPROACH: drive Y {steps:+d} steps (width_err={y_err_w:.0f}px)")
            if staged: return staged
            return f"Pick Up: approaching (width_err={y_err_w:.0f}px)"
        return None

    def _approach_recenter(self, x_err, z_err, marker_w):
        """Correct X then Z during a Y approach so the marker stays centered (and in-frame) as it
        nears. Returns a status string if it issued a correction move (or an ABORT), else None once
        the marker is centered enough to drive Y. Bounded by the alignment-move guard (its budget,
        reset on each approach entry, is otherwise unused while approaching)."""
        steps = self._physics_px_to_steps(x_err, marker_w) if abs(x_err) > self.active_tol['x_px'] else 0
        if steps != 0:
            abort = self._align_move_guard("approach re-center")
            if abort: return abort
            steps, reach = self._x_reach_abort(steps, "approach re-center")
            if reach: return reach
            staged = self._gate(lambda: (self._set_cooldown_steps(steps), self.messenger.move_gantry_x(steps)),
                                f"APPROACH re-center X: move {steps:+d} steps (x_err={x_err}px)")
            if staged: return staged
            return f"Pick Up: approach re-centering X (err={x_err}px)"
        if abs(z_err) > self.active_tol['z_px']:
            abort = self._align_move_guard("approach re-center")
            if abort: return abort
            pulse_ms = self._z_pulse_ms(z_err)
            pulse_ms, reach = self._z_reach_abort(pulse_ms, "approach re-center")
            if reach: return reach
            staged = self._gate(lambda: (self._set_cooldown_ms(pulse_ms), self.messenger.pulse_lift_z(pulse_ms)),
                                f"APPROACH re-center Z: pulse {pulse_ms:+d} ms (z_err={z_err}px)")
            if staged: return staged
            return f"Pick Up: approach re-centering Z (err={z_err}px)"
        return None

    def _approach_blocked_abort(self, steps):
        """Abort string (→ IDLE) if a Y approach move is blocked by the limit it's pushing into, else
        None. A retract (steps < 0) means the marker reads wider than the calibrated pre-engage width;
        if Y is already fully retracted we can't back off, so it would never converge. A forward move
        (steps > 0) into a pressed Y_MAX means the target is out of reach before the engage width."""
        if steps < 0 and self.messenger.y_pos <= 0:
            self._approach_moves = 0
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return ("ABORT: Pick Up — marker reads wider than the calibrated pre-engage width but Y is "
                    "fully retracted (re-check WIDTH REF calibration / move the target back).")
        if steps > 0 and self.messenger.limits.get('y_max'):
            self._approach_moves = 0
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return ("ABORT: Pick Up — reached Y_MAX before the engage width (target out of reach / "
                    "re-check WIDTH REF calibration).")
        return None

    def _run_picking_up_logic(self, frame):
        """Staged target pick-up (reached automatically after coarse align):
          0. approach Y to the pre-engage (close) width
          1. fine XZ re-align at the calibration depth, then orientation check (abort if out of tol)
          2. approach Y to the full engagement width
          3. lift Z   4. done + grip confirm
        Falls back to a single hardcoded Y engage if the pick-up profile isn't calibrated."""
        current_time = time.time()
        if not self._move_settled():
            return f"Pick Up: step {self._seq_step}, waiting..."

        p = self.profile('pickup')
        use_width = self.cal_ready('pickup') and p['aim_width_px'] is not None

        # Fallback: uncalibrated → one blind Y engage, then jump to the lift step.
        if not use_width:
            if self._seq_step == 0:
                self._seq_step = 3   # advance before gating: the blind engage is this step's action
                staged = self._gate(lambda: (self._set_cooldown_steps(self.Y_DOCKING_STEPS),
                                             self.messenger.move_gantry(self.messenger.x_pos, self.Y_DOCKING_STEPS)),
                                    f"BLIND ENGAGE: drive Y in to {self.Y_DOCKING_STEPS} steps (no width cal)")
                if staged: return staged
                return "Pick Up: Approaching battery (no width cal)..."

        if self._seq_step == 0:
            msg = self._approach_to_width(frame, p['aim_width_px'])
            if msg is not None:
                return msg
            self.lost_frames = 0
            self._seq_step = 1
            self._align_moves = 0
            return "Pick Up: At pre-engage distance — fine aligning..."

        elif self._seq_step == 1:
            found, x_err, z_err, marker_w, _, _ = self.eyes.get_target_error(frame, self.target_id)
            self.last_x_err, self.last_z_err = x_err, z_err
            if not found:
                self.lost_frames += 1
                if self.lost_frames > self.MAX_LOST_FRAMES:
                    self.lost_frames = 0
                    self._seq_step = 0
                    self.current_state = self.STATE_IDLE
                    return "ABORT: Pick Up — marker lost during fine align."
                return "Pick Up: fine align, searching..."
            self.lost_frames = 0
            steps = self._physics_px_to_steps(x_err, marker_w) if abs(x_err) > self.active_tol['x_px'] else 0
            if steps != 0:
                abort = self._align_move_guard("fine align")
                if abort: return abort
                steps, reach = self._x_reach_abort(steps, "fine align")
                if reach: return reach
                staged = self._gate(lambda: (self._set_cooldown_steps(steps), self.messenger.move_gantry_x(steps)),
                                    f"FINE align X: move {steps:+d} steps (x_err={x_err}px)")
                if staged: return staged
                return f"Pick Up: fine X (err={x_err}px)"
            if abs(z_err) > self.active_tol['z_px']:
                abort = self._align_move_guard("fine align")
                if abort: return abort
                pulse_ms = self._z_pulse_ms(z_err)
                pulse_ms, reach = self._z_reach_abort(pulse_ms, "fine align")
                if reach: return reach
                staged = self._gate(lambda: (self._set_cooldown_ms(pulse_ms), self.messenger.pulse_lift_z(pulse_ms)),
                                    f"FINE align Z: pulse {pulse_ms:+d} ms (z_err={z_err}px)")
                if staged: return staged
                return f"Pick Up: fine Z (err={z_err}px)"
            # Centered at the calibration depth — orientation target is valid here.
            if not self.is_orientation_ok:
                self._seq_step = 0
                self.current_state = self.STATE_IDLE
                return "ABORT: Pick Up — marker orientation out of tolerance."
            self._seq_step = 2
            self._align_moves = 0
            self._approach_moves = 0      # entering the full-engagement Y approach (seq_step 2)
            return "Pick Up: aligned — engaging..."

        elif self._seq_step == 2:
            # Engagement drive: no re-center — fine align already centered X/Z and the scoop is
            # entering the handle, so any X correction here jams it against the handle wall.
            msg = self._approach_to_width(frame, p['target_width_px'], recenter=False)
            if msg is not None:
                return msg
            self._seq_step = 3
            return "Pick Up: fully engaged — lifting..."

        elif self._seq_step == 3:
            self._seq_step = 4   # advance before gating: the lift is step 3's whole action
            staged = self._gate(lambda: (self._set_cooldown_ms(self.Z_RETRACT_PICKUP_MS),
                                         self.messenger.pulse_lift_z(-self.Z_RETRACT_PICKUP_MS)),
                                f"LIFT: retract Z up {self.Z_RETRACT_PICKUP_MS} ms (raise battery clear)")
            if staged: return staged
            return "Pick Up: Lifting..."

        else:
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            if self.messenger.grip_engaged:
                return "Pick Up Complete — holding battery (confirmed)."
            return ("ABORT: Pick Up — grip empty after lift, nothing picked up "
                    "(scoop missed the handle, or it slipped off — re-check alignment/calibration).")

    def _run_our_picking_up_logic(self, frame):
        """Pick up a battery from our own carrier slot (hardcoded X + optional vision refine).

        Slots are at known X positions, so this is dead-reckoning with an optional ID-agnostic XZ
        refine (corrects drift since homing) when the pick-up profile is calibrated.

        Steps:
          0. Move X to slot (Y stays 0); load pick-up crosshair if calibrated
          1. Optional XZ refine to the nearest marker (skipped if uncalibrated / no marker)
          2. Lower Z by Z_HANDLE_DEPTH_MS (scoop to handle height)
          3. Drive Y in to Y_DOCKING_STEPS (handle into scoop)
          4. Lift Z by Z_CLEAR_LIFT_MS (raise battery clear of slot)
          5. Back Y to 0   6. done + grip confirm
        """
        if self.Z_HANDLE_DEPTH_MS is None or self.Z_CLEAR_LIFT_MS is None:
            print("Our Slot Pick Up: scoop Z heights not set — set them in the web UI first")
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return "Our Slot Pick Up: scoop Z heights not set (Setup → Scoop Z Heights)"

        current_time = time.time()
        if not self._move_settled():
            return f"Our Slot Pick Up: step {self._seq_step}, waiting..."

        slot_idx  = self._seq_slot_idx
        slot_x    = self.SLOT_X_STEPS[slot_idx]
        p         = self.profile('pickup')
        refine_ok = self.cal_ready('pickup')

        if self._seq_step == 0:
            x_travel = abs(slot_x - self.messenger.x_pos)
            self.messenger.move_gantry(slot_x, 0)
            self._set_cooldown_steps(x_travel if x_travel > 0 else 1)
            if refine_ok:
                self.eyes.set_target(p['crosshair_x'], p['crosshair_y'])
            self.lost_frames = 0
            self._align_moves = 0
            self._seq_step = 1
            return f"Our Slot {slot_idx + 1} Pick Up: Moving X to slot..."

        elif self._seq_step == 1:
            # Optional ID-agnostic XZ refine: align the marker nearest the crosshair (whatever
            # battery is in this slot). Corrects drift since homing. Skips if uncalibrated/no marker.
            if not refine_ok:
                self._seq_step = 2
                return "Our Slot Pick Up: no cal — using hardcoded position"
            found, x_err, z_err, marker_w, _, _ = self.eyes.get_nearest_error(
                frame, p['crosshair_x'], p['crosshair_y'])
            if not found:
                self.lost_frames += 1
                if self.lost_frames > self.MAX_LOST_FRAMES:
                    self.lost_frames = 0
                    self._seq_step = 2
                    return "Our Slot Pick Up: no marker — using hardcoded position"
                return "Our Slot Pick Up: refine, searching..."
            self.lost_frames = 0
            self.last_x_err, self.last_z_err = x_err, z_err
            steps = self._physics_px_to_steps(x_err, marker_w) if abs(x_err) > self.active_tol['x_px'] else 0
            if steps != 0:
                abort = self._align_move_guard("our-slot refine")
                if abort: return abort
                steps, reach = self._x_reach_abort(steps, "our-slot refine")
                if reach: return reach
                self._set_cooldown_steps(steps)
                self.messenger.move_gantry_x(steps)
                return f"Our Slot Pick Up: refine X (err={x_err}px)"
            if abs(z_err) > self.active_tol['z_px']:
                abort = self._align_move_guard("our-slot refine")
                if abort: return abort
                pulse_ms = self._z_pulse_ms(z_err)
                pulse_ms, reach = self._z_reach_abort(pulse_ms, "our-slot refine")
                if reach: return reach
                self._set_cooldown_ms(pulse_ms)
                self.messenger.pulse_lift_z(pulse_ms)
                return f"Our Slot Pick Up: refine Z (err={z_err}px)"
            self._seq_step = 2
            return "Our Slot Pick Up: refined — lowering scoop..."

        elif self._seq_step == 2:
            self.messenger.pulse_lift_z(self.Z_HANDLE_DEPTH_MS)
            self._set_cooldown_ms(self.Z_HANDLE_DEPTH_MS)
            self._seq_step = 3
            return "Our Slot Pick Up: Lowering scoop to handle height..."

        elif self._seq_step == 3:
            self.messenger.move_gantry(slot_x, self.Y_DOCKING_STEPS)
            self._set_cooldown_steps(self.Y_DOCKING_STEPS)
            self._seq_step = 4
            return "Our Slot Pick Up: Engaging handle..."

        elif self._seq_step == 4:
            self.messenger.pulse_lift_z(-self.Z_CLEAR_LIFT_MS)
            self._set_cooldown_ms(self.Z_CLEAR_LIFT_MS)
            self._seq_step = 5
            return "Our Slot Pick Up: Lifting battery clear of slot..."

        elif self._seq_step == 5:
            self.messenger.move_gantry(slot_x, 0)
            self._set_cooldown_steps(self.Y_DOCKING_STEPS)
            self._seq_step = 6
            return "Our Slot Pick Up: Retracting Y..."

        else:
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            if self.messenger.grip_engaged:
                return f"Our Slot {slot_idx + 1} Pick Up Complete — holding battery (confirmed)."
            return (f"ABORT: Our Slot {slot_idx + 1} Pick Up — grip empty after lift, nothing "
                    f"picked up (scoop missed the handle, or it slipped off).")

    def _run_our_setting_down_logic(self, frame):
        """Place a held battery into our own carrier slot — no vision (our robot has no markers).

        Re-homes X/Y with the battery first (fresh switch reference — open-loop steps drift, worst
        under a swinging load), then dead-reckons to the known slot location and lowers until the
        grip switch releases. The whole op runs under the gentle loaded motion profile. The scoop is
        freed by retracting Y — NOT by raising Z, which would re-cradle the handle and lift it back out.

        Steps:
          0. Re-home X/Y with the battery (kicked off here, in the brain thread, then waited out)
          1. Move X to slot (Y stays 0)
          2. Drive Y in to Y_DOCKING_STEPS
          3. Extend Z in short pulses down toward the absolute seat depth (Z_HANDLE_DEPTH_MS); stop
             the instant the grip switch releases (the firmware self-aborts the pulse on seat), or
             at that depth as the floor
          4. Back Y to 0 (slides the scoop off the handle, battery stays seated)
        """
        if self.Z_HANDLE_DEPTH_MS is None or self.Z_CLEAR_LIFT_MS is None:
            print("Our Slot Set Down: scoop Z heights not measured — measure in lab first")
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return "Our Slot Set Down: Z heights not calibrated (Setup → Scoop Z Heights)"

        # Step 0: re-home with the battery. Kicked off here (not in the public method) so all motion
        # side-effects run on the brain thread — the IDLE branch can't race the loaded-profile setup.
        if self._seq_step == 0:
            if not self._rehome_started:
                self._enter_loaded_motion()        # gentle accel/speed while carrying
                self.messenger.start_home()        # non-blocking (Z up → X → Y); waited out below
                self._rehome_started = True
                return "Our Slot Set Down: re-homing with battery (Z up, then X, Y)..."
            if self.messenger.is_homing:
                return "Our Slot Set Down: re-homing with battery (Z up, then X, Y)..."
            if not self.messenger.has_homed:
                self._rehome_started = False
                self._seq_step = 0
                self.current_state = self.STATE_IDLE
                return ("ABORT: Our Slot Set Down — re-home failed "
                        f"({self.messenger.last_home_error or 'E-stop / interrupted'}).")
            self._rehome_started = False
            self._seq_step = 1
            return "Our Slot Set Down: re-homed — moving to slot..."

        current_time = time.time()
        if not self._move_settled():
            return f"Our Slot Set Down: step {self._seq_step}, waiting..."

        slot_x   = self.SLOT_X_STEPS[self._seq_slot_idx]

        if self._seq_step == 1:
            x_travel = abs(slot_x - self.messenger.x_pos)
            self.messenger.move_gantry(slot_x, 0)
            self._set_cooldown_steps(x_travel if x_travel > 0 else 1)
            self._seq_step = 2
            return f"Our Slot {self._seq_slot_idx + 1} Set Down: Moving X to slot..."

        elif self._seq_step == 2:
            self.messenger.move_gantry(slot_x, self.Y_DOCKING_STEPS)
            self._set_cooldown_steps(self.Y_DOCKING_STEPS)
            self._seq_step = 3
            self._z_set_down_total_ms = 0
            self._prev_grip_engaged = None
            return "Our Slot Set Down: Over slot — lowering battery..."

        elif self._seq_step == 3:
            # Lower toward the ABSOLUTE seat depth (handle height), stopping early the instant the
            # grip switch releases (battery seated — the firmware also self-aborts the pulse). Lowering
            # to an absolute Z depth — not a fixed pulse budget from wherever Z happens to be — makes
            # this start-position-independent: from full retract it still lands at the same depth, so
            # it can't drive the scoop past the seat into the frame.
            if self._grip_seated():
                self._seq_step = 4
                return "Our Slot Set Down: Battery seated (grip released)!"
            cur_ms   = self.messenger.z_pos_mm * self.messenger.z_ms_per_mm   # current depth from Z-home
            floor_ms = self.Z_HANDLE_DEPTH_MS                                 # absolute seat depth
            if cur_ms >= floor_ms - self.Z_SET_DOWN_PULSE_MS:
                self._seq_step = 4
                return "Our Slot Set Down: Reached seat depth without release — proceeding"

            pulse_ms = min(self.Z_SET_DOWN_PULSE_MS, floor_ms - cur_ms)
            self.messenger.pulse_lift_z(int(pulse_ms))
            self._set_cooldown_ms(pulse_ms)
            return f"Our Slot Set Down: Lowering to seat depth ({int(cur_ms)}/{int(floor_ms)}ms)..."

        elif self._seq_step == 4:
            self.messenger.move_gantry(slot_x, 0)
            self._set_cooldown_steps(self.Y_DOCKING_STEPS)
            self._seq_step = 5
            return "Our Slot Set Down: Retracting Y (freeing scoop)..."

        else:
            self._seq_step = 0
            self.current_state = self.STATE_IDLE   # IDLE restores the normal motion profile
            return f"Our Slot {self._seq_slot_idx + 1} Set Down Complete."

    def _run_setting_down_target_logic(self, frame):
        """Place a held battery into a target-robot slot, keyed by the slot's flag marker ID.

        Unlike a pick-up there is no handle to thread, so this does NOT stage a Y engage or fish in
        Z. It commits to a safe shape for carrying a ~30 kg load: retract Z fully first (raise the
        battery — max ground clearance, shortest pendulum), align over the slot in X and Y only
        (NOT Z — Z is purely the descent), then lower straight down until the grip switch releases.
        Runs under the gentle loaded motion profile throughout. Seating is the grip switch releasing
        (the firmware self-aborts the lowering pulse the instant it seats); the scoop is freed by
        retracting Y, never by raising Z.

        Steps:
          0. RAISE  — enter loaded profile, retract Z fully to carry height, prep the search
          1. SEARCH — +X sweep for the slot flag marker
          2. ALIGN  — X to the aim point + Y to the target width (at full retract; Z untouched)
          3. LOWER  — descend in short pulses, re-centering X between pulses; stop on grip release
             (seated). Guards: the calibrated seated pose (stop if the marker reaches it) and a
             relative travel backstop (Z_SETDOWN_GUARD_MS) for a never-seats jam.
          4. RETRACT Y to 0 (frees the scoop, battery stays seated)
        """
        setdown = self.profile('setdown')
        cal_ok  = self.cal_ready('setdown')

        if not cal_ok:
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return "Target Set Down: Set-down profile not calibrated — run calibration first"

        # Step 0 issues the Z raise then waits via _move_settled like any other move.
        if self._seq_step == 0:
            self._enter_loaded_motion()
            retract_ms = abs(self.messenger.z_pos_mm * self.messenger.z_ms_per_mm)
            self.messenger.move_z_to(0)            # full retract: raise battery to carry height
            self._set_cooldown_ms(retract_ms if retract_ms > 0 else 1)
            self._reset_search()
            self._seq_step = 1
            return "Target Set Down: Raising battery to carry height..."

        current_time = time.time()
        if not self._move_settled():
            return f"Target Set Down: step {self._seq_step}, waiting..."

        if self._seq_step == 1:
            # Search for the slot flag marker (Z already retracted, so the camera sees the flags).
            found, *_ = self.eyes.get_target_error(frame, self.target_id)
            if found:
                self.search_count = 0
                self.lost_frames = 0
                self.eyes.set_target(setdown['crosshair_x'], setdown['crosshair_y'])
                self._align_moves = 0
                self._seq_step = 2
                return "Target Set Down: Slot found — aligning over it..."

            if (current_time - self.last_search_time) >= self.SEARCH_COOLDOWN_S:
                if self.search_count < self.MAX_SEARCH_STEPS:
                    self.search_count += 1
                    self.last_search_time = current_time
                    self.messenger.move_gantry_x(self.SEARCH_STEP_SIZE)
                    return f"Target Set Down: Searching... (Step {self.search_count})"
                self._seq_step = 0
                self.current_state = self.STATE_IDLE
                return "Target Set Down: Slot marker not found"
            return f"Target Set Down: Searching... (Step {self.search_count})"

        elif self._seq_step == 2:
            # Align over the slot in X and Y only. Z stays fully retracted — there is no handle to
            # match a height to, and fishing in Z under load risks snagging the battery's underside.
            found, x_err, _, marker_w, _, _ = self.eyes.get_target_error(frame, self.target_id)
            self.last_x_err, self.last_z_err = x_err, 0

            if not found:
                self.lost_frames += 1
                if self.lost_frames > self.MAX_LOST_FRAMES:
                    self.lost_frames = 0
                    self._seq_step = 1
                    self._reset_search()
                    return "Target Set Down: Slot lost — re-searching"
                return "Target Set Down: Vision flicker..."
            self.lost_frames = 0

            steps = self._physics_px_to_steps(x_err, marker_w) if abs(x_err) > self.active_tol['x_px'] else 0
            if steps != 0:
                abort = self._align_move_guard("set-down align")
                if abort: return abort
                steps, reach = self._x_reach_abort(steps, "set-down align")
                if reach: return reach
                self._set_cooldown_steps(steps)
                self.messenger.move_gantry_x(steps)
                return f"Target Set Down: Aligning X (err={x_err}px)"

            y_err_w = setdown['target_width_px'] - marker_w
            if abs(y_err_w) > self.active_tol['width_px']:
                abort = self._align_move_guard("set-down Y approach")
                if abort: return abort
                steps = self._physics_depth_to_steps_y_with_target(marker_w, setdown['target_width_px'])
                steps = max(-self.MAX_APPROACH_STEP, min(self.MAX_APPROACH_STEP, steps))
                self._set_cooldown_steps(steps)
                self.messenger.move_gantry_y(steps)
                return f"Target Set Down: Y approach (width_err={y_err_w:.0f}px)"

            self._align_moves = 0
            self._z_set_down_total_ms = 0
            self._prev_grip_engaged = None
            self._seq_step = 3
            return "Target Set Down: Over slot — lowering battery..."

        elif self._seq_step == 3:
            # Descend straight down. PRIMARY stop = grip switch releasing (seated); the firmware
            # self-aborts the active pulse the instant that happens, so overshoot is one loop tick.
            if self._grip_seated():
                self._seq_step = 4
                return "Target Set Down: Battery seated (grip released)!"

            # Re-center X between pulses (lowering translates the marker in the frame). Bounded: once
            # the budget is spent, stop correcting and keep lowering — grip + guards still bound it.
            found, x_err, _, marker_w, _, _ = self.eyes.get_target_error(frame, self.target_id)
            if found:
                self.last_x_err = x_err
                # Vision guard: marker reached the calibrated seated depth (≈ as close as it gets when
                # seated) but the grip hasn't released — stop before driving the battery past its seat.
                if self.seated_ready('setdown') and \
                        marker_w >= setdown['seated_width_px'] - self.active_tol['width_px']:
                    self._seq_step = 4
                    return "Target Set Down: Reached seated pose without grip release — stopping"
                if abs(x_err) > self.active_tol['x_px'] and self._align_moves <= self.MAX_ALIGN_MOVES:
                    steps = self._physics_px_to_steps(x_err, marker_w)
                    if steps != 0:
                        self._align_moves += 1
                        steps, reach = self._x_reach_abort(steps, "set-down descend re-center")
                        if reach: return reach
                        self._set_cooldown_steps(steps)
                        self.messenger.move_gantry_x(steps)
                        return f"Target Set Down: re-centering X while lowering (err={x_err}px)"

            # Relative travel backstop: never lower more than the guard past the aligned pose (the
            # grip switch is the real stop; this only catches a never-seats jam).
            if self._z_set_down_total_ms >= self.Z_SETDOWN_GUARD_MS:
                self._seq_step = 4
                return "Target Set Down: Travel guard reached without seating — backing off"

            remaining_ms = self.Z_SETDOWN_GUARD_MS - self._z_set_down_total_ms
            pulse_ms = min(self.Z_SET_DOWN_PULSE_MS, remaining_ms)
            self.messenger.pulse_lift_z(int(pulse_ms))
            self._z_set_down_total_ms += pulse_ms
            self._set_cooldown_ms(pulse_ms)
            return f"Target Set Down: Lowering ({self._z_set_down_total_ms}/{self.Z_SETDOWN_GUARD_MS}ms)..."

        elif self._seq_step == 4:
            self.messenger.move_gantry(self.messenger.x_pos, 0)
            self._set_cooldown_steps(self.Y_DOCKING_STEPS)
            self._seq_step = 5
            return "Target Set Down: Retracting Y (freeing scoop)..."

        else:
            self._seq_step = 0
            self.current_state = self.STATE_IDLE   # IDLE restores the normal motion profile
            return "Target Set Down Complete."

    def _run_demo_follow_logic(self, frame):
        """Continuously correct X+Z to keep marker centered. Never stops on its own."""
        found, x_err, z_err, marker_w, _, _ = self.eyes.get_target_error(frame, self.target_id)
        self.last_x_err, self.last_z_err = x_err, z_err

        if not found:
            return "Demo Follow: waiting for marker..."

        current_time = time.time()
        if not self._move_settled():
            return "Demo Follow: moving..."

        steps = self._physics_px_to_steps(x_err, marker_w) if abs(x_err) > self.active_tol['x_px'] else 0
        if steps != 0:
            self._set_cooldown_steps(steps)
            self.messenger.move_gantry_x(steps)
            return f"Demo Follow: X (err={x_err}px)"

        if abs(z_err) > self.active_tol['z_px']:
            pulse_ms = self._z_pulse_ms(z_err)
            self._set_cooldown_ms(pulse_ms)
            self.messenger.pulse_lift_z(pulse_ms)
            return f"Demo Follow: Z (err={z_err}px)"

        return "Demo Follow: centered."

    # ------------------------------------------------------------------ #
    #  Calibration persistence                                             #
    # ------------------------------------------------------------------ #

    def _fresh_cal(self):
        return {
            "pickup":  dict(_EMPTY_PROFILE),
            "setdown": dict(_EMPTY_PROFILE),
            "marker_labels": {},
            "z_heights": dict(_EMPTY_Z_HEIGHTS),
            "tolerances": {"pickup": dict(_DEFAULT_TOL), "setdown": dict(_DEFAULT_TOL)},
            "flip_feed": False,
            "settle_time_s": _DEFAULT_SETTLE_TIME_S,
            "z_ms_per_mm": _DEFAULT_Z_MS_PER_MM,
        }

    def _load_calibration(self):
        try:
            with open(_CAL_FILE) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._fresh_cal()

        cal = self._fresh_cal()

        # Per-operation profiles
        for name in ('pickup', 'setdown'):
            if isinstance(data.get(name), dict):
                cal[name].update({k: data[name][k] for k in _EMPTY_PROFILE if k in data[name]})

        # Legacy flat format → migrate the single global crosshair/width into the pickup profile
        if 'crosshair_x' in data or 'target_marker_width_px' in data:
            cal['pickup']['crosshair_x']    = data.get('crosshair_x')
            cal['pickup']['crosshair_y']    = data.get('crosshair_y')
            cal['pickup']['target_width_px'] = data.get('target_marker_width_px')

        if isinstance(data.get('marker_labels'), dict):
            cal['marker_labels'] = {str(k): v for k, v in data['marker_labels'].items()}
        if isinstance(data.get('z_heights'), dict):
            zh = data['z_heights']
            cal['z_heights'].update({k: zh[k] for k in _EMPTY_Z_HEIGHTS if k in zh})
            # Legacy keys → intuitive absolutes: lower_ms→handle_depth_ms,
            # (lower_ms+clear_ms)→clear_lift_ms. (The old setdown_max_ms / target_setdown_max_ms
            # depth cap is retired — the target set-down now uses a relative travel guard.)
            if 'lower_ms' in zh and cal['z_heights']['handle_depth_ms'] is None:
                cal['z_heights']['handle_depth_ms'] = zh['lower_ms']
            if cal['z_heights']['clear_lift_ms'] is None and zh.get('lower_ms') is not None \
                    and zh.get('clear_ms') is not None:
                cal['z_heights']['clear_lift_ms'] = zh['lower_ms'] + zh['clear_ms']
        if isinstance(data.get('tolerances'), dict):
            for name in ('pickup', 'setdown'):
                if isinstance(data['tolerances'].get(name), dict):
                    cal['tolerances'][name].update(
                        {k: data['tolerances'][name][k] for k in _DEFAULT_TOL if k in data['tolerances'][name]})
        if isinstance(data.get('flip_feed'), bool):
            cal['flip_feed'] = data['flip_feed']
        if isinstance(data.get('settle_time_s'), (int, float)):
            cal['settle_time_s'] = max(0.0, min(5.0, float(data['settle_time_s'])))
        if isinstance(data.get('z_ms_per_mm'), (int, float)) and data['z_ms_per_mm'] > 0:
            cal['z_ms_per_mm'] = float(data['z_ms_per_mm'])
        return cal

    def save_calibration(self):
        with open(_CAL_FILE, 'w') as f:
            json.dump(self.cal, f, indent=2)

    # ------------------------------------------------------------------ #
    #  Math helpers                                                        #
    # ------------------------------------------------------------------ #

    def _physics_px_to_steps(self, pixel_error, marker_width_px):
        depth_mm  = (self.FOCAL_LENGTH_PX * self.MARKER_PHYSICAL_WIDTH_MM) / marker_width_px
        offset_mm = (pixel_error * depth_mm) / self.FOCAL_LENGTH_PX
        raw = int(offset_mm / self.MM_PER_STEP_XY)
        # If the true correction is below the minimum executable step, DON'T round it up to
        # MIN_STEP_PER_MOVE: a min-step move spans more pixels than the tolerance band, so the
        # carriage jumps clean over the target and lands on the far side every cycle (+10/-10/+10…
        # limit cycle). Return 0 instead — we're already as close as the gantry can usefully get,
        # and the caller treats a 0-step X correction as aligned. (Z/Y are unaffected.)
        if abs(raw) < self.MIN_STEP_PER_MOVE:
            return 0
        return self._clamp_steps(raw)

    def _physics_depth_to_steps_y_with_target(self, marker_width_px, target_width_px):
        """Y steps to reach a specific target marker width (engagement depth)."""
        current_depth_mm = (self.FOCAL_LENGTH_PX * self.MARKER_PHYSICAL_WIDTH_MM) / marker_width_px
        target_depth_mm  = (self.FOCAL_LENGTH_PX * self.MARKER_PHYSICAL_WIDTH_MM) / target_width_px
        return self._clamp_steps(int((current_depth_mm - target_depth_mm) / self.MM_PER_STEP_XY))

    def _grip_seated(self):
        """True on the engaged→disengaged transition of the end-effector grip switch.

        While carrying, the battery's weight holds the switch engaged; when the battery settles
        into the slot the scoop drops free and the switch releases. That transition = seated."""
        g = self.messenger.grip_engaged
        seated = (self._prev_grip_engaged is True and not g)
        self._prev_grip_engaged = g
        return seated

    def _z_pulse_ms(self, z_err, max_ms=None):
        # mm of Z correction (geometry, speed-independent) → ms via the single Z speed constant.
        # max_ms overrides the per-pulse cap (coarse align passes the larger MAX_Z_PULSE_COARSE_MS).
        cap = max_ms if max_ms is not None else self.MAX_Z_PULSE_MS
        raw = int(self.Z_MM_PER_PX * abs(z_err) * self.z_ms_per_mm)
        ms  = max(self.MIN_Z_PULSE_MS, min(raw, cap))
        return ms if z_err > 0 else -ms

    def _clamp_steps(self, steps):
        if abs(steps) > self.MAX_STEP_PER_MOVE:
            return self.MAX_STEP_PER_MOVE if steps > 0 else -self.MAX_STEP_PER_MOVE
        if 0 < abs(steps) < self.MIN_STEP_PER_MOVE:
            return self.MIN_STEP_PER_MOVE if steps > 0 else -self.MIN_STEP_PER_MOVE
        return steps

    def _xy_move_time_s(self, steps):
        """Real X/Y travel time including the accel/decel ramp. The Arduino ramps at OPERATING_ACCEL,
        so short moves never reach top speed (triangular profile) — a plain steps/speed estimate
        badly under-counts them and the camera would re-analyze mid-move."""
        d = abs(steps)
        if d == 0:
            return 0.0
        a = float(getattr(self.messenger, 'acceleration', None) or self.OPERATING_ACCEL_SPS2)
        a = max(1.0, a)
        # Use the live commanded speed (the gentle loaded profile lowers it), not the constant — else
        # a loaded move's cooldown is sized for full speed and the camera re-analyzes mid-move.
        vmax = float(getattr(self.messenger, 'speed', None) or self.OPERATING_SPEED_SPS)
        d_ramp = vmax * vmax / a                       # steps spent ramping up to vmax and back down
        if d <= d_ramp:
            return 2.0 * (d / a) ** 0.5                 # triangular: never reaches vmax
        return 2.0 * (vmax / a) + (d - d_ramp) / vmax   # trapezoidal: ramp up + cruise + ramp down

    def _arm_settle(self, move_time_s):
        """Start the post-move wait: remember when the move was commanded and its expected duration
        (the timeout fallback), and reset the 'motion finished' marker. _move_settled() takes over."""
        self.last_align_time  = time.time()
        self._move_expected_s = max(0.0, move_time_s)
        self._move_done_at    = None
        self._settling        = True

    def _move_settled(self):
        """True once it's safe to analyze a new frame: the Arduino has reported the move finished
        (is_busy cleared) AND the tunable settle margin has since elapsed — OR a hard timeout
        (estimated move time × SETTLE_TIMEOUT_FACTOR + margin) has passed, so a missed DONE can
        never hang the loop. Returns True immediately when no move is in flight."""
        if not self._settling:
            return True
        now = time.time()
        if not self.messenger.is_busy and self._move_done_at is None:
            self._move_done_at = now                  # motion just finished — start the settle timer
        settled = (self._move_done_at is not None and
                   (now - self._move_done_at) >= self.settle_time_s)
        timed_out = (now - self.last_align_time) >= \
            (self._move_expected_s * self.SETTLE_TIMEOUT_FACTOR + self.settle_time_s + 1.0)
        if settled or timed_out:
            self._settling = False
            return True
        return False

    def _set_cooldown_steps(self, steps):
        self._arm_settle(self._xy_move_time_s(steps))

    def _set_cooldown_ms(self, ms):
        self._arm_settle(abs(ms) / 1000.0)   # a Z pulse's duration is exactly its ms

    def _reset_search(self):
        self.lost_frames = 0
        self.search_count = 0
        self.last_search_time = 0
        self._align_moves = 0
        self._approach_moves = 0
        self._search_prep_cooldown = self.SEARCH_COOLDOWN_S  # first-pulse wait, sized in prep
        self._search_prepped = False   # re-run the X=0 / Z-up prep at the start of each search

    def _align_move_guard(self, phase):
        """Count one corrective move in the current alignment phase; return an ABORT message
        (and drop to IDLE) if the phase has issued MAX_ALIGN_MOVES without converging — i.e. it
        is oscillating around the tolerance boundary and would otherwise loop forever. Returns
        None while still under the cap. Call once per commanded corrective X/Z/Y move; reset
        self._align_moves = 0 when entering each alignment phase."""
        self._align_moves += 1
        if self._align_moves > self.MAX_ALIGN_MOVES:
            self._align_moves = 0
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return (f"ABORT: {phase} — {self.MAX_ALIGN_MOVES}+ correction moves without "
                    f"converging (likely oscillating; loosen tolerance or re-check calibration).")
        return None

    def _approach_move_guard(self):
        """Count one Y depth move in the current approach-to-width phase; return an ABORT message
        (and drop to IDLE) if it has issued MAX_APPROACH_MOVES without reaching the target width —
        the marker width never converges (usually a stale WIDTH REF or lighting/detection drift).
        Separate from _align_move_guard so it has its own budget and a calibration-pointing reason.
        Returns None while under the cap; reset self._approach_moves = 0 when entering an approach."""
        self._approach_moves += 1
        if self._approach_moves > self.MAX_APPROACH_MOVES:
            self._approach_moves = 0
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return (f"ABORT: Pick Up approach — {self.MAX_APPROACH_MOVES}+ Y moves without reaching "
                    f"the engage width (re-check the WIDTH REF calibration / lighting).")
        return None

    def _x_reach_abort(self, steps, phase, abort_on_max=True):
        """Keep an X correction inside the gantry's travel. Both ends are guarded:

        - HOME end (X=0): if the marker wants the gantry *past* X=0 while it's already home there,
          the target is unreachable → IDLE + abort. The step counter is dead-reckoned (a normal G/R
          move never consults X_MIN), so it can read ~0 while the carriage is physically short of
          home; before declaring "out of travel" we check the live X_MIN switch — if it isn't pressed
          there's real travel left, so we let the move run (the firmware hard-stops and re-zeros at
          the switch, self-healing the drift). Otherwise clamp to home and continue.

        - FAR end (X_MAX): there is no hardcoded max step (the open-loop counter drifts), so the
          X_MAX switch is the only trustworthy bound. If a +X correction is requested while X_MAX is
          pressed, the carriage is jammed against the far stop and can't go further.
          (arduino_link latches x_max from the firmware's async `LIMIT X_MAX HIT` so this is seen
          within a cycle of the first clip, not after the whole 40-move budget.)
          • `abort_on_max=True` (default): treat that as out of travel → IDLE + abort.
          • `abort_on_max=False` (coarse align): don't abort — return 0 steps so the caller *skips*
            the X move and proceeds (Z align, then the Y approach). The out-of-reach call is deferred
            to the fine-align stage at pre-engage distance, where the marker may have come into the
            tolerance window as the camera closed in (perspective), and where abort_on_max is True.

        Returns (steps_to_use, abort_message_or_None)."""
        x = self.messenger.x_pos
        if x is not None and steps < 0 and x + steps < 0:
            if x < self.MIN_STEP_PER_MOVE:   # step counter says we're at the X=0 home end
                if not self.messenger.limits.get('x_min', False):
                    # Counter reads home but X_MIN isn't pressed → drift left travel toward home.
                    # Allow the move; the switch hard-stops and re-zeros if the carriage reaches it.
                    return steps, None
                self._seq_step = 0
                self.current_state = self.STATE_IDLE
                return 0, (f"ABORT: {phase} — marker needs the gantry past X=0 (X_MIN switch pressed, "
                           f"truly out of travel), so it's outside reach. The camera may have shifted "
                           f"— re-aim or re-calibrate.")
            steps = -x   # only as far as home
        if steps > 0 and self.messenger.limits.get('x_max', False):
            if not abort_on_max:
                return 0, None   # jammed at X_MAX — skip the X move, defer the decision to fine align
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return 0, (f"ABORT: {phase} — marker needs the gantry past X_MAX (switch pressed, truly "
                       f"out of travel), so it's beyond the arm's reach. Reposition the battery/target "
                       f"closer in X, or re-aim the camera.")
        return steps, None

    def _z_reach_abort(self, pulse_ms, phase):
        """Block a Z retract (negative pulse, scoop up) when the scoop is already at its top stop
        (z_pos_mm <= 0) — it can't retract further, so a marker above that is unreachable. Returns
        (pulse_ms_to_use, abort_message_or_None)."""
        if pulse_ms < 0 and self.messenger.z_pos_mm <= 0:
            self._seq_step = 0
            self.current_state = self.STATE_IDLE
            return 0, (f"ABORT: {phase} — marker needs the scoop to retract above its top stop "
                       f"(already home), so it's outside reach. The camera may have shifted — "
                       f"re-aim or re-calibrate.")
        return pulse_ms, None

    # ─── Step mode (approve-each-move) ─────────────────────────────────────────
    def _gate(self, action_fn, desc):
        """Decide how a single physical move happens. In normal mode, run it now and return None
        (caller proceeds). In step mode, stage it for operator approval and return the operator-
        facing message the caller should return — so the staged frame, every hold frame, and the UI
        step-bar all show the same text. `action_fn` must perform the move AND set its cooldown, so
        timing starts when the move actually runs (on approval)."""
        if not self.step_mode:
            action_fn()
            return None
        self._pending_action = action_fn
        self._pending_desc = f"STEP ▸ {desc} — press Approve (A)"
        self.awaiting_approval = True
        return self._pending_desc

    @property
    def pending_step(self):
        """The staged-move description while awaiting approval, else None (for the UI)."""
        return self._pending_desc if self.awaiting_approval else None

    def approve_step(self):
        """Operator approved the staged move: run it (it sets its own cooldown) and clear the gate."""
        if not (self.awaiting_approval and self._pending_action):
            return False
        fn = self._pending_action
        self._pending_action = None
        self.awaiting_approval = False
        fn()
        return True

    def abort_step(self):
        """Operator aborted from step mode: drop any staged move and return to IDLE."""
        self._pending_action = None
        self.awaiting_approval = False
        self._seq_step = 0
        self._reset_search()
        self.current_state = self.STATE_IDLE
        return True

    def set_step_mode(self, on):
        """Toggle step mode. Turning it off mid-op ('Resume autonomous') drops any staged move so
        the operation continues on its own from the next cycle."""
        self.step_mode = bool(on)
        if not self.step_mode:
            self.awaiting_approval = False
            self._pending_action = None
        return self.step_mode
