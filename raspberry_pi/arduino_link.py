import time
import threading
import queue

# Calibrated in lab: 56 mm measured over 20 000 ms
_Z_MS_PER_MM = 20000.0 / 56.0    # ≈ 357.14 ms per mm

# Homing wall-clock cap — if HOMING COMPLETE never arrives (motor PSU off, a limit switch fault,
# or a dead USB cable), home() returns False with a reason instead of blocking its thread forever.
# Generous: a full Z retract to the Z_TOP switch is the slow leg.
HOMING_TIMEOUT_S = 120

# Seconds without any line from the Arduino before the link is considered dead. The heartbeat ping
# (web_server) sends P every ~2 s so a healthy idle link keeps refreshing _last_rx well inside this.
LINK_STALE_S = 5

class ArduinoLink:
    def __init__(self, port='/dev/ttyUSB0', use_simulation=False):
        self.is_simulated = use_simulation
        self._x_pos = 0
        self._y_pos = 0
        self._z_pos_mm = 0.0
        self._z_busy = False        # True while a Z pulse is in flight
        self._z_pending_mm = 0.0   # position delta to apply when Z DONE arrives
        self._pending_x = None      # target x committed to _x_pos only when DONE arrives
        self._pending_y = None      # target y committed to _y_pos only when DONE arrives
        self._serial = None
        # Bounded so the steady-state heartbeat (P) replies, which only home() ever drains, can't
        # accumulate unboundedly between homing runs — _read_loop drops the oldest line on overflow.
        self._response_queue = queue.Queue(maxsize=256)
        self.load_detected = False
        self.has_homed = False      # True only after a completed HOME (axes referenced to switches)
        self.is_homing = False      # True while a HOME is in flight (axes mid-reference)
        self.last_home_error = None # reason string when the last HOME failed (timeout / E-stop), else None
        # Link health: _last_rx is bumped on every received line; _link_error is set if the read
        # thread dies. link_alive combines both (see property). Seeded now so a never-connected
        # link reads stale rather than falsely alive.
        self._last_rx = time.time()
        self._link_error = None
        self.grip_engaged = False   # end-effector limit switch: True = battery weight on scoop
        # Live travel-limit states, refreshed from the P heartbeat reply (True = pressed). Lets the
        # UI show them and lets the controller tell a real "out of travel" from open-loop drift.
        self._limits = {'x_min': False, 'x_max': False, 'y_min': False, 'y_max': False, 'z_top': False}
        self.acceleration = 80      # last commanded/echoed accel (steps/s²); matches Arduino OPERATING_ACCEL
        self.accel_confirmed = True # True once the Arduino echoes "ACCEL SET TO:" (the boot default is known)
        self.speed = 60             # last commanded/echoed max speed (steps/s); matches Arduino OPERATING_SPEED
        self.speed_confirmed = True # True once the Arduino echoes "SPEED SET TO:" (the boot default is known)
        # ms of actuator run-time per mm of Z travel. Lab-calibratable (the DC actuator's speed
        # drifts with load/wear), so it's an instance attr the controller can update at runtime
        # via set_z_ms_per_mm — single source of truth, shared with the alignment gain.
        self._z_ms_per_mm = _Z_MS_PER_MM

        if not use_simulation:
            import serial as pyserial
            self._serial = pyserial.Serial(port, 9600, timeout=1)
            time.sleep(2)
            threading.Thread(target=self._read_loop, daemon=True).start()
        else:
            print("ArduinoLink: Simulation mode — no hardware needed.")

    @property
    def x_pos(self):
        return self._x_pos

    @property
    def y_pos(self):
        return self._y_pos

    @property
    def z_pos_mm(self):
        return self._z_pos_mm

    @property
    def limits(self):
        """Live travel-limit states {x_min,x_max,y_min,y_max,z_top}; True = switch pressed.
        Real hardware: parsed from the P heartbeat reply (≤ a couple seconds stale, refreshed only
        while idle). Simulation: derived from the tracked position so the UI still shows something."""
        if self.is_simulated:
            return {'x_min': self._x_pos <= 0, 'x_max': False,
                    'y_min': self._y_pos <= 0, 'y_max': False,
                    'z_top': self._z_pos_mm <= 0}
        return dict(self._limits)

    def _read_loop(self):
        while self._serial.is_open:
            try:
                line = self._serial.readline().decode('utf-8', errors='replace').strip()
                if line:
                    self._last_rx = time.time()   # any line = link alive (heartbeat refreshes this)
                    print(f"[ARDUINO]: {line}")
                    self._enqueue(line)
                    if "HOMING COMPLETE" in line:
                        # Homing finished — referenced to the switches. Handled here (not only in the
                        # blocking home() loop) so a non-blocking start_home() also updates the flags.
                        self._x_pos = 0
                        self._y_pos = 0
                        self._z_pos_mm = 0.0
                        self._z_pending_mm = 0.0
                        self._z_busy = False
                        self._pending_x = None
                        self._pending_y = None
                        self.has_homed = True
                        self.is_homing = False
                        self.last_home_error = None
                    elif "DONE" in line and "Z" not in line:
                        # Commit pending XY position now that the move actually completed
                        if self._pending_x is not None:
                            self._x_pos = self._pending_x
                        if self._pending_y is not None:
                            self._y_pos = self._pending_y
                        self._pending_x = None
                        self._pending_y = None
                    elif "LIMIT X_MIN HIT" in line:
                        self._x_pos = 0
                        self._pending_x = 0        # X confirmed at 0
                        self._limits['x_min'] = True   # latch now; next P reply refreshes it
                    elif "LIMIT Y_MIN HIT" in line:
                        self._y_pos = 0
                        self._pending_y = 0        # Y confirmed at 0
                        self._limits['y_min'] = True
                    elif "LIMIT X_MAX HIT" in line:
                        self._pending_x = None     # X position unknown — don't update
                        # Latch the switch state immediately. The P heartbeat that normally fills
                        # _limits is only sent while idle, so without this the controller's far-end
                        # reach guard wouldn't see the max until ~seconds later — long enough to burn
                        # the whole 40-move align budget against the stop. Self-clears on the next P.
                        self._limits['x_max'] = True
                    elif "LIMIT Y_MAX HIT" in line:
                        self._pending_y = None     # Y position unknown — don't update
                        self._limits['y_max'] = True
                    elif "LOAD DETECTED" in line:
                        self.load_detected = True
                    elif line.startswith("ACCEL SET TO:"):
                        try:
                            self.acceleration = int(line.split(":", 1)[1].strip())
                            self.accel_confirmed = True   # Arduino echoed it back — applied
                        except (IndexError, ValueError):
                            pass
                    elif line.startswith("SPEED SET TO:"):
                        try:
                            self.speed = int(line.split(":", 1)[1].strip())   # clamped value from firmware
                            self.speed_confirmed = True   # Arduino echoed it back — applied
                        except (IndexError, ValueError):
                            pass
                    elif line == "GRIP 1":
                        self.grip_engaged = True
                    elif line == "GRIP 0":
                        self.grip_engaged = False
                    elif line == "Z DONE":
                        self._z_pos_mm += self._z_pending_mm
                        self._z_pending_mm = 0.0
                        self._z_busy = False
                    elif line == "Z HOME":
                        self._z_pos_mm = 0.0
                        self._z_pending_mm = 0.0
                        self._z_busy = False
                    elif line == "Z SEATED":
                        # A set-down lower self-aborted on grip release (battery seated mid-pulse).
                        # Treat like Z DONE: clear busy and commit the pending delta. The real travel
                        # was a bit less than commanded (the pulse stopped early), so z_pos_mm runs
                        # marginally deep — harmless: Z is re-homed each cycle and we never lower from
                        # this tracked value (the next op re-references it).
                        self._z_pos_mm += self._z_pending_mm
                        self._z_pending_mm = 0.0
                        self._z_busy = False
                    elif line.startswith("X=") and "X_MIN=" in line:
                        # P-reply: "X=.. Y=.. Z=.. G=.. X_MIN=0 X_MAX=0 Y_MIN=0 Y_MAX=0 Z_TOP=0"
                        p_x = p_y = None
                        for tok in line.split():
                            key, _, val = tok.partition('=')
                            kl = key.lower()
                            if kl in self._limits:
                                self._limits[kl] = (val == '1')
                            elif kl == 'x':
                                try: p_x = int(val)
                                except ValueError: pass
                            elif kl == 'y':
                                try: p_y = int(val)
                                except ValueError: pass
                        # Resync tracked position to the Arduino's own step counter, but ONLY when no
                        # XY move is in flight (else we'd clobber the optimistic target mid-move). This
                        # heals open-loop dead-reckoning drift — e.g. steps the firmware took into a
                        # limit clip that were never committed here — so the dashboard stops lying.
                        if self._pending_x is None and p_x is not None:
                            self._x_pos = p_x
                        if self._pending_y is None and p_y is not None:
                            self._y_pos = p_y
            except Exception as e:
                # Don't die silently: record why so link_alive can report the break to the operator
                # (e.g. the USB cable was pulled). The Pi side stops getting fresh data from here on.
                self._link_error = repr(e)
                print(f"[ARDUINO] read loop stopped: {self._link_error}")
                break

    def _enqueue(self, line):
        """Put a line on the response queue, dropping the oldest if full (heartbeat replies that
        nothing drains between homing runs must not grow without bound)."""
        try:
            self._response_queue.put_nowait(line)
        except queue.Full:
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._response_queue.put_nowait(line)
            except queue.Full:
                pass

    @property
    def link_alive(self):
        """True if the Arduino link is healthy. Always True in simulation. Otherwise requires the
        read thread to be alive (no recorded error) and a line received within LINK_STALE_S — the
        heartbeat ping keeps this fresh even when idle."""
        if self.is_simulated:
            return True
        return self._link_error is None and (time.time() - self._last_rx) < LINK_STALE_S

    @property
    def link_error(self):
        return self._link_error

    def ping(self):
        """Heartbeat: ask for position so a healthy-but-idle link keeps producing RX lines."""
        self._send("P\n")

    def _send(self, command):
        if self.is_simulated:
            print(f"[SIM -> ARDUINO]: {command.strip()}")
        else:
            self._serial.write(command.encode('utf-8'))

    def start_home(self):
        """Send H and return immediately (non-blocking). The read loop sets has_homed/is_homing as
        the Arduino reports progress (HOMING COMPLETE / EMERGENCY STOP). Use this from the brain's
        per-tick state machine — e.g. an our-slot set-down that re-homes a held battery — so the
        camera/brain thread isn't blocked for the whole (multi-second) homing run. Poll homing_ready
        / is_homing to know when it's done. Flask's HOME button uses the blocking home() wrapper."""
        self._z_busy = False
        self._z_pending_mm = 0.0
        self.has_homed = False
        self.is_homing = True
        self.last_home_error = None
        self._send("H\n")
        if self.is_simulated:
            self._x_pos = 0
            self._y_pos = 0
            self._z_pos_mm = 0.0
            self.has_homed = True
            self.is_homing = False

    def home(self):
        """Blocking home: start it, then wait until HOMING COMPLETE (or E-stop/timeout). Returns True
        on success. Completion is detected via the read-loop flags (set on the async serial lines),
        so this just polls has_homed / last_home_error. Used by Flask's HOME route (its own thread)."""
        self.start_home()
        if self.is_simulated:
            return True
        deadline = time.time() + HOMING_TIMEOUT_S
        while time.time() < deadline:
            if self.has_homed and not self.is_homing:
                return True
            # An E-stop during homing flips is_homing off via emergency_stop(); surface it as a failure.
            if not self.is_homing and not self.has_homed:
                self.last_home_error = self.last_home_error or "Homing aborted (E-stop)"
                return False
            time.sleep(0.05)
        # Fell through the deadline without HOMING COMPLETE — don't hang the thread forever.
        self.is_homing = False
        self.last_home_error = (f"Homing timed out after {HOMING_TIMEOUT_S}s — no HOMING COMPLETE "
                                f"(check motor PSU is on, limit switches, USB cable)")
        print(f"[ARDUINO] {self.last_home_error}")
        return False

    @property
    def is_busy(self):
        """True while a move or Z pulse is in flight (cleared by the Arduino's DONE / Z DONE)."""
        return self._z_busy or self._pending_x is not None or self._pending_y is not None

    def _sim_commit_xy(self):
        """In simulation there is no DONE reply, so commit the move immediately."""
        if self.is_simulated:
            self._x_pos = self._pending_x
            self._y_pos = self._pending_y
            self._pending_x = None
            self._pending_y = None

    def move_gantry_x(self, delta_steps):
        """Relative X move. Sent as a single-axis relative R command so the
        other axis is never re-commanded (avoids dragging Y to a stale tracked
        value when Pi/Arduino positions are desynced — e.g. before homing).

        No soft clamp at the tracked zero: before homing that zero is fictional
        (just the boot default), so clamping would refuse a legitimate negative
        jog when the carriage isn't actually at X_MIN. The firmware's limit
        switches are the real bound — a min-switch hit stops the move and resets
        the tracked position to 0."""
        self._pending_x = self._x_pos + delta_steps
        self._pending_y = self._y_pos
        self._send(f"R {delta_steps} 0\n")
        self._sim_commit_xy()

    def move_gantry_y(self, delta_steps):
        """Relative Y move. Sent as a single-axis relative R command (X is left
        untouched via dx=0). See move_gantry_x for rationale (no soft clamp;
        limit switches bound the travel)."""
        self._pending_x = self._x_pos
        self._pending_y = self._y_pos + delta_steps
        self._send(f"R 0 {delta_steps}\n")
        self._sim_commit_xy()

    def move_gantry(self, x, y):
        """Absolute move to (x, y) in steps."""
        self._pending_x = x
        self._pending_y = y
        self._send(f"G {x} {y}\n")
        self._sim_commit_xy()

    def pulse_lift_z(self, ms):
        """Pulse Z actuator. Positive ms = extend (arm down), negative = retract (arm up).
        Drops the command if a pulse is already in flight to prevent position drift.
        In simulation mode, updates position immediately (no Z DONE feedback)."""
        if ms > 0:
            self.load_detected = False
        if self.is_simulated:
            self._z_pos_mm += ms / self._z_ms_per_mm
            self._send(f"Z {ms}\n")
            return
        if self._z_busy:
            return  # previous pulse still running — drop to avoid drift
        self._z_pending_mm = ms / self._z_ms_per_mm
        self._z_busy = True
        self._send(f"Z {ms}\n")

    def move_z_to(self, target_mm):
        """Absolute Z move. Calculates the delta from current tracked position and pulses."""
        delta_ms = int(round((target_mm - self._z_pos_mm) * self._z_ms_per_mm))
        if delta_ms == 0:
            return
        self.pulse_lift_z(delta_ms)

    def set_z_ms_per_mm(self, value):
        """Update the Z speed calibration (ms of run-time per mm). Kept in sync with the
        controller's persisted value so position tracking and the alignment gain agree."""
        self._z_ms_per_mm = max(1.0, float(value))

    @property
    def z_ms_per_mm(self):
        return self._z_ms_per_mm

    def set_acceleration(self, steps_per_s2):
        self.acceleration = steps_per_s2           # optimistic; corrected by the Arduino echo
        # Real hardware confirms via the "ACCEL SET TO:" echo; sim has no echo so treat as confirmed.
        self.accel_confirmed = self.is_simulated
        self._send(f"A {steps_per_s2}\n")

    def set_speed(self, steps_per_s):
        self.speed = steps_per_s                   # optimistic; corrected by the Arduino echo
        # Real hardware confirms via the "SPEED SET TO:" echo (which carries the clamped value);
        # sim has no echo so treat as confirmed.
        self.speed_confirmed = self.is_simulated
        self._send(f"S {steps_per_s}\n")

    def emergency_stop(self):
        self._pending_x = None
        self._pending_y = None
        self._z_busy = False
        self._z_pending_mm = 0.0
        # An E-stop (incl. mid-homing) leaves the axes unreferenced — force a re-home before operating.
        self.has_homed = False
        self.is_homing = False
        self._send("E\n")

    def query_position(self):
        self._send("P\n")

    def is_limit_switch_pressed(self):
        return False
