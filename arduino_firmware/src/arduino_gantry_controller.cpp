#include <Arduino.h>
#include <AccelStepper.h>

// SERIAL PROTOCOL (9600 baud, newline-terminated):
//   H           — Home all axes (Z retract → X → Y); replies "HOMING COMPLETE"
//   G <x> <y>   — Move X to x steps, Y to y steps (absolute, sequential: X first then Y)
//   R <dx> <dy> — Move X by dx steps, Y by dy steps (relative, sequential: X first then Y)
//   A <val>     — Set acceleration for both axes (steps/s^2)
//   Z <ms>      — Pulse Z actuator: +ms = extend down, -ms = retract up
//                 During a DOWNWARD pulse that began while the grip switch was holding a battery,
//                 the pulse self-aborts the instant the switch releases (battery seated), replying
//                 "Z SEATED" instead of "Z DONE" — so a set-down stops within a loop tick of the
//                 weight transferring, not a whole pulse later (minimal crush risk on the battery).
//   E           — Emergency stop (all axes, including homing)
//   P           — Query current position; replies
//                 "X=<x> Y=<y> Z=<z_mm> G=<0|1> X_MIN=<0|1> X_MAX=<0|1> Y_MIN=<0|1> Y_MAX=<0|1> Z_TOP=<0|1>"
//                 (limit fields: 1 = pressed)
// Async output (unsolicited):
//   GRIP 1 / GRIP 0 — end-effector grip switch changed (1 = holding, 0 = empty/seated)

// --- PIN DEFINITIONS ---
const int LIMIT_X_MIN = 6; // Active Low (in idle state the switch is open and in HIGH state, when switch closes, the pin connects to ground and becomes LOW)
const int LIMIT_Y_MIN = 7;
const int LIMIT_X_MAX = 8;
const int LIMIT_Y_MAX = 11;
const int LIMIT_Z_TOP = 12; // Active Low; triggers when arm is fully retracted (home position)

// End-effector grip switch on the scoop. Active Low (INPUT_PULLUP): closes to GND (LOW) when a
// battery's weight is resting on the scoop ("holding"); open (HIGH) when empty / seated.
// A0/A1 are wired to the IBT_2 current sense (R_IS/L_IS, informational only, not read in firmware);
// pin 13 is avoided (onboard LED interferes with pull-up). A2 is the first free digital-capable pin.
const int LIMIT_GRIP = A2;

// Z-axis linear actuator (IBT_2 driver).
// RPWM drives the motor forward (extend = arm moves DOWN toward battery).
// LPWM drives the motor in reverse (retract = arm moves UP to safe position).
// Pins 9 and 10 are used because they are the only free PWM-capable pins on the Nano.
const int Z_RPWM = 9;
const int Z_LPWM = 10;

// Z position tracking. Calibrated in lab: 56 mm measured over 20 000 ms → 0.0028 mm/ms.
// z_position_ms: 0 = fully retracted (home), positive = extended down.
const float Z_MM_PER_MS = 56.0 / 20000.0;

// --- MOTION SETTINGS ---
const int HOMING_SPEED = 85;       // Steps/s during homing — slow for safety near physical stops
const int OPERATING_SPEED = 60;    // Steps/s during normal operation (boot default; live-tunable via 'S')
const int OPERATING_ACCEL = 80;    // Steps/s^2 during normal operation

// Speed clamp for the runtime 'S' command. Below ~10 sps moves crawl; above ~200 sps a full-step
// NEMA 23 loses torque and skips — so a fat-fingered value can't stall or freeze an axis.
const int MIN_OPERATING_SPEED = 10;
const int MAX_OPERATING_SPEED = 200;

// Define stepper objects for X and Y axes
AccelStepper stepperX(AccelStepper::DRIVER, 2, 3);
AccelStepper stepperY(AccelStepper::DRIVER, 4, 5);

// Live operating speed (steps/s) for normal X/Y moves. Starts at the boot default and is updated
// by the 'S' command; applied in setup() and restored after homing. Homing itself uses HOMING_SPEED.
int currentOperatingSpeed = OPERATING_SPEED;

const byte MSG_MAX_LEN = 24; // Max characters allowed in a single Serial command
char inputBuffer[MSG_MAX_LEN]; // Array to store the incoming Serial characters
int bufferIndex = 0; // Current position in the input buffer
bool isMovingSequential = false;  // Tracks if a G-code sequence is currently active
bool zActive = false;             // True while a Z pulse is in progress
unsigned long zEndTime = 0;       // millis() timestamp when the current Z pulse should stop
unsigned long zStartTime = 0;     // millis() when the current Z pulse began (for partial-pulse tracking)
unsigned long zLastBeat = 0;      // millis() of the last "Z RUN" heartbeat emitted during the pulse
int zDirection = 0;               // +1 = extending down, -1 = retracting up
bool zSeatAbortArmed = false;     // set when a DOWN pulse starts while holding a battery; while armed,
                                  //   runZ() stops the pulse the instant the grip switch releases
                                  //   (battery seated) and replies "Z SEATED" — see runZ()/handleSerial
const unsigned long Z_BEAT_MS = 1000;  // emit a heartbeat this often during a Z pulse so the Pi's
                                       // link-staleness timer doesn't trip on long (multi-second) moves
long z_position_ms = 0;           // Signed accumulated Z travel in ms; 0 = home (fully retracted)

enum HomingState { HOMING_IDLE, HOMING_Z, HOMING_X, HOMING_Y };
HomingState homingState = HOMING_IDLE;
bool hasHomed = false;            // True after a successful H command completes

// Grip switch debounce/reporting state
const unsigned long GRIP_DEBOUNCE_MS = 30;
int gripRawState        = HIGH;   // last raw read
int gripStableState     = HIGH;   // debounced state
int gripReportedState   = -1;     // last state announced over serial (-1 forces first report)
unsigned long gripLastChangeMs = 0;

// Forward declarations — required in C++ so the compiler knows these functions
// exist before loop() calls them. The actual definitions come further below.
void handleSerial();
void checkLimits();
void checkGrip();
void runMotors();
void runZ();
void runHoming();
void beginHomingX();
void beginHomingY();
void stopAll();

void setup() {
  Serial.begin(9600);

  // Configure limit switches with internal pull-up resistors (Active low)
  pinMode(LIMIT_X_MIN, INPUT_PULLUP);
  pinMode(LIMIT_Y_MIN, INPUT_PULLUP);
  pinMode(LIMIT_X_MAX, INPUT_PULLUP);
  pinMode(LIMIT_Y_MAX, INPUT_PULLUP);
  pinMode(LIMIT_Z_TOP, INPUT_PULLUP);
  pinMode(LIMIT_GRIP, INPUT_PULLUP);

  // Configure Z-axis actuator pins and ensure motor starts in the off state
  pinMode(Z_RPWM, OUTPUT);
  pinMode(Z_LPWM, OUTPUT);
  digitalWrite(Z_RPWM, LOW);
  digitalWrite(Z_LPWM, LOW);

  // Initial default values
  stepperX.setMaxSpeed(currentOperatingSpeed);
  stepperX.setAcceleration(OPERATING_ACCEL);
  stepperY.setMaxSpeed(currentOperatingSpeed);
  stepperY.setAcceleration(OPERATING_ACCEL);

  Serial.println(F("SYSTEM READY"));
} 

void loop() {
  handleSerial();
  if (homingState != HOMING_IDLE) {
    runHoming();     // Homing state machine — handles its own limit detection
  } else {
    checkLimits();   // Safety: stop motors if a limit switch is triggered
    runMotors();     // Advance steppers one step toward their targets (non-blocking)
  }
  runZ();            // Check if the active Z pulse has elapsed and stop if so
  checkGrip();       // Report end-effector grip switch state changes over serial
}

// Reads incoming serial characters one at a time into a buffer.
// When a full line arrives (newline or carriage return), it identifies the
// command letter and dispatches to the appropriate handler.
// Called every loop() tick so no serial data is ever missed.
void handleSerial() {
  while (Serial.available() > 0) {   // While there is data waiting in the Serial buffer
    char inChar = Serial.read();  // Read the next character from Serial
    if (inChar == '\n' || inChar == '\r') {
      if (bufferIndex > 0) { // Safety: prevents processing empty lines
        inputBuffer[bufferIndex] = '\0';
        char cmd = inputBuffer[0];

        // If it is a Homing command
        if (cmd == 'H') {
          Serial.println(F("HOMING: Retracting Z..."));
          analogWrite(Z_LPWM, 255);
          analogWrite(Z_RPWM, 0);
          homingState = HOMING_Z;
        }

        // If it is a Movement command
        else if (cmd == 'G') {
          long targetX, targetY;   // Variables to store the target coordinates
          if (sscanf(inputBuffer + 1, "%ld %ld", &targetX, &targetY) == 2) { // Extract two numbers
            if (!hasHomed) Serial.println(F("WARN: not homed — position may be incorrect"));
            Serial.print(F("Moving to X=")); Serial.print(targetX);
            Serial.print(F(", Y=")); Serial.println(targetY);

            // Position 0 is defined by the limit switch, not step count.
            // Drive past 0 so checkLimits() stops and zeroes on the physical switch.
            stepperX.moveTo(targetX == 0 ? -99999L : targetX);
            stepperY.moveTo(targetY == 0 ? -99999L : targetY);
            isMovingSequential = true; // Set flag to track the movement sequence
          } else {
            // sscanf returns the number of items successfully parsed.
            // If it's not 2, the command was malformed (e.g. "G 100" with no Y value).
            // We print an error so the Pi knows to re-send rather than waiting forever.
            Serial.println(F("ERROR: Bad G command. Format: G <x> <y>"));
          }
        }

        // If it is a Relative movement command (testing/jogging only — Pi always uses G)
        else if (cmd == 'R') {
          long dx, dy;
          if (sscanf(inputBuffer + 1, "%ld %ld", &dx, &dy) == 2) {
            if (!hasHomed) Serial.println(F("WARN: not homed — position may be incorrect"));
            long targetX = stepperX.currentPosition() + dx;
            long targetY = stepperY.currentPosition() + dy;
            Serial.print(F("Moving to X=")); Serial.print(targetX);
            Serial.print(F(", Y=")); Serial.println(targetY);
            stepperX.moveTo(targetX);
            stepperY.moveTo(targetY);
            isMovingSequential = true;
          } else {
            Serial.println(F("ERROR: Bad R command. Format: R <dx> <dy>"));
          }
        }

        // If it is a Position query command
        else if (cmd == 'P') {
          Serial.print(F("X=")); Serial.print(stepperX.currentPosition());
          Serial.print(F(" Y=")); Serial.print(stepperY.currentPosition());
          Serial.print(F(" Z=")); Serial.print(z_position_ms * Z_MM_PER_MS, 1);
          Serial.print(F(" G=")); Serial.print(digitalRead(LIMIT_GRIP) == LOW ? 1 : 0);
          // Live limit-switch states (1 = pressed / LOW). Lets the Pi surface them in the UI and
          // distinguish a real "out of travel" from open-loop drift (the step counter can read 0
          // while X_MIN isn't actually pressed, since a normal G/R move never consults the switch).
          Serial.print(F(" X_MIN=")); Serial.print(digitalRead(LIMIT_X_MIN) == LOW ? 1 : 0);
          Serial.print(F(" X_MAX=")); Serial.print(digitalRead(LIMIT_X_MAX) == LOW ? 1 : 0);
          Serial.print(F(" Y_MIN=")); Serial.print(digitalRead(LIMIT_Y_MIN) == LOW ? 1 : 0);
          Serial.print(F(" Y_MAX=")); Serial.print(digitalRead(LIMIT_Y_MAX) == LOW ? 1 : 0);
          Serial.print(F(" Z_TOP=")); Serial.println(digitalRead(LIMIT_Z_TOP) == LOW ? 1 : 0);
        }

        // If it is an Acceleration command
        else if (cmd == 'A') {
          long newAccel = atol(&inputBuffer[1]); // Convert text after 'A' to a long integer
          stepperX.setAcceleration(newAccel);  // Update acceleration for X (at the moment x and y accel are always equal)
          stepperY.setAcceleration(newAccel);  // Update acceleration for Y (at the moment x and y accel are always equal)
          Serial.print(F("ACCEL SET TO: "));
          Serial.println(newAccel);
        }

        // If it is a Speed command (max steps/s for normal X/Y moves; mirrors 'A')
        else if (cmd == 'S') {
          long newSpeed = atol(&inputBuffer[1]); // Convert text after 'S' to a long integer
          if (newSpeed < MIN_OPERATING_SPEED) newSpeed = MIN_OPERATING_SPEED;
          if (newSpeed > MAX_OPERATING_SPEED) newSpeed = MAX_OPERATING_SPEED;
          currentOperatingSpeed = (int)newSpeed;       // persists for all later moves + homing restore
          stepperX.setMaxSpeed(currentOperatingSpeed); // X and Y always share one speed
          stepperY.setMaxSpeed(currentOperatingSpeed);
          Serial.print(F("SPEED SET TO: "));
          Serial.println(currentOperatingSpeed);       // echo the clamped value so the Pi syncs to truth
        }

        // If it is a Z-axis pulse command
        else if (cmd == 'Z') {
          long duration = atol(&inputBuffer[1]); // Signed ms: positive = extend down, negative = retract up

          if (duration > 0) {
            analogWrite(Z_RPWM, 255); // Full power extend
            analogWrite(Z_LPWM, 0);
            zStartTime = millis();
            zEndTime = zStartTime + (unsigned long)duration;
            zLastBeat = zStartTime;
            zDirection = 1;
            zActive = true;
            // Arm the seat-abort ONLY if a battery is currently on the scoop (grip holding = LOW).
            // A downward pulse while holding only ever happens during a set-down, so a grip release
            // mid-pulse means the battery just seated → stop now. Pickups lower with an empty scoop
            // (grip HIGH), so this stays disarmed for them and can't cut a pickup descent short.
            zSeatAbortArmed = (gripStableState == LOW);
          } else if (duration < 0) {
            analogWrite(Z_RPWM, 0);
            analogWrite(Z_LPWM, 255); // Full power retract
            zStartTime = millis();
            zEndTime = zStartTime + (unsigned long)(-duration);
            zLastBeat = zStartTime;
            zDirection = -1;
            zActive = true;
            zSeatAbortArmed = false;  // retracts never seat
          }
          // "Z DONE" is sent by runZ() once the pulse duration has elapsed.
          // If E arrives before then, stopAll() kills the pins immediately.
        }

        else if (cmd == 'E') {
          stopAll();
          Serial.println(F("EMERGENCY STOP"));
        }

        bufferIndex = 0;
      }
    }
    else if (bufferIndex < MSG_MAX_LEN - 1) { // Safety: prevents buffer overflow
      inputBuffer[bufferIndex++] = inChar;
    }
  }
}

// Checks all four limit switches every loop() tick.
// distanceToGo() = targetPosition - currentPosition, so:
//   negative → motor is moving toward the MIN end
//   positive → motor is moving toward the MAX end
// The directional guard prevents a false stop: if the system boots with
// an axis already touching a switch, it can still move *away* from it.
void checkLimits() {
  // setCurrentPosition() zeros AccelStepper's velocity immediately — unlike stop()+moveTo()
  // which only sets a deceleration target and lets the motor keep running.
  if (digitalRead(LIMIT_X_MIN) == LOW && stepperX.distanceToGo() < 0) {
    stepperX.setCurrentPosition(0);
    Serial.println(F("LIMIT X_MIN HIT - X STOPPED"));
  }
  if (digitalRead(LIMIT_X_MAX) == LOW && stepperX.distanceToGo() > 0) {
    stepperX.setCurrentPosition(stepperX.currentPosition());
    Serial.println(F("LIMIT X_MAX HIT - X STOPPED"));
  }
  if (digitalRead(LIMIT_Y_MIN) == LOW && stepperY.distanceToGo() < 0) {
    stepperY.setCurrentPosition(0);
    Serial.println(F("LIMIT Y_MIN HIT - Y STOPPED"));
  }
  if (digitalRead(LIMIT_Y_MAX) == LOW && stepperY.distanceToGo() > 0) {
    stepperY.setCurrentPosition(stepperY.currentPosition());
    Serial.println(F("LIMIT Y_MAX HIT - Y STOPPED"));
  }
}

// Debounced read of the end-effector grip switch. Emits "GRIP 1" (battery weight on scoop /
// holding) or "GRIP 0" (empty / seated) over serial only when the debounced state changes, so
// the Pi can track holding/seating without polling. Called every loop() tick.
void checkGrip() {
  int r = digitalRead(LIMIT_GRIP);
  if (r != gripRawState) {
    gripRawState = r;
    gripLastChangeMs = millis();
  }
  if ((millis() - gripLastChangeMs) >= GRIP_DEBOUNCE_MS) {
    gripStableState = r;
  }
  if (gripStableState != gripReportedState) {
    gripReportedState = gripStableState;
    // LOW = switch closed = battery weight on the scoop = engaged
    Serial.println(gripStableState == LOW ? F("GRIP 1") : F("GRIP 0"));
  }
}

// Advances the motors one step per loop() tick (non-blocking).
// X runs first; Y only starts once X has reached its target.
// When both finish, "DONE" is sent so the Pi knows it can issue the next command.
void runMotors() {
  if (stepperX.distanceToGo() != 0) {
    // run() checks whether enough time has elapsed (based on speed and acceleration)
    // to take a single step. If not yet, it returns immediately with no step taken.
    stepperX.run();
  }
  else if (stepperY.distanceToGo() != 0) {
    stepperY.run();
  }
  else if (isMovingSequential) {
    Serial.println(F("DONE"));
    isMovingSequential = false;
  }
}

// Checks whether the active Z pulse has elapsed (or the Z_TOP limit fired). Cuts power,
// updates z_position_ms, and sends "Z DONE". Called every loop() tick.
void runZ() {
  if (!zActive) return;

  // Seat-abort: while lowering a held battery (armed at pulse start), the grip switch releasing
  // (gripStableState went HIGH = weight off the scoop) means the battery has settled into the slot.
  // Stop immediately and report "Z SEATED" so the Pi knows it seated mid-pulse (vs. ran to time).
  // gripStableState is the debounced read maintained by checkGrip(), so this won't trip on noise.
  if (zDirection == 1 && zSeatAbortArmed && gripStableState == HIGH) {
    analogWrite(Z_RPWM, 0);
    analogWrite(Z_LPWM, 0);
    z_position_ms += (long)(millis() - zStartTime) * zDirection;  // credit the partial extend
    zActive = false;
    zSeatAbortArmed = false;
    Serial.println(F("Z SEATED"));
    return;
  }

  // Z_TOP limit switch: if retracting and the arm reaches home, stop and zero position.
  // Send "Z HOME" (not "Z DONE") so the Pi knows to sync its own position to 0.
  if (zDirection == -1 && digitalRead(LIMIT_Z_TOP) == LOW) {
    analogWrite(Z_RPWM, 0);
    analogWrite(Z_LPWM, 0);
    z_position_ms = 0;
    zActive = false;
    Serial.println(F("Z HOME"));
    return;
  }

  if (millis() >= zEndTime) {
    analogWrite(Z_RPWM, 0);
    analogWrite(Z_LPWM, 0);
    z_position_ms += (long)(zEndTime - zStartTime) * zDirection;
    zActive = false;
    Serial.println(F("Z DONE"));
    return;
  }

  // Still pulsing: emit a periodic heartbeat so the Pi keeps receiving lines during a long Z move.
  // Without this the Pi sees no serial traffic for the whole pulse (its position heartbeat is
  // suppressed while busy) and trips its link-staleness timer. The Pi treats any line as "alive";
  // "Z RUN" matches none of its parse branches, so it's a harmless keep-alive.
  if (millis() - zLastBeat >= Z_BEAT_MS) {
    zLastBeat = millis();
    Serial.println(F("Z RUN"));
  }
}

// Non-blocking homing state machine. Called every loop() tick while homingState != HOMING_IDLE.
// Stages: HOMING_Z (limit switch) → HOMING_X (limit switch) → HOMING_Y (limit switch).
// E-stop aborts it at any point by setting homingState = HOMING_IDLE in stopAll().
void runHoming() {
  if (homingState == HOMING_Z) {
    if (digitalRead(LIMIT_Z_TOP) == LOW) {
      analogWrite(Z_LPWM, 0);
      analogWrite(Z_RPWM, 0);
      z_position_ms = 0;
      beginHomingX();
    }
  } else if (homingState == HOMING_X) {
    if (digitalRead(LIMIT_X_MIN) == LOW) {
      stepperX.setCurrentPosition(0);
      stepperX.setMaxSpeed(currentOperatingSpeed);
      beginHomingY();
    } else {
      stepperX.run();
    }
  } else if (homingState == HOMING_Y) {
    if (digitalRead(LIMIT_Y_MIN) == LOW) {
      stepperY.setCurrentPosition(0);
      stepperY.setMaxSpeed(currentOperatingSpeed);
      hasHomed = true;
      homingState = HOMING_IDLE;
      Serial.println(F("HOMING COMPLETE"));
    } else {
      stepperY.run();
    }
  }
}

// Checks X_MIN before starting X homing. If already pressed, zeros X and skips straight to Y.
// Prevents the stutter caused by moveTo() firing step pulses into an already-triggered switch.
void beginHomingX() {
  if (digitalRead(LIMIT_X_MIN) == LOW) {
    stepperX.setCurrentPosition(0);
    Serial.println(F("HOMING: X already home, Homing Y..."));
    beginHomingY();
  } else {
    Serial.println(F("HOMING: Homing X..."));
    stepperX.setMaxSpeed(HOMING_SPEED);
    stepperX.moveTo(-99999);
    homingState = HOMING_X;
  }
}

// Checks Y_MIN before starting Y homing. If already pressed, zeros Y and completes homing.
void beginHomingY() {
  if (digitalRead(LIMIT_Y_MIN) == LOW) {
    stepperY.setCurrentPosition(0);
    stepperY.setMaxSpeed(currentOperatingSpeed);
    hasHomed = true;
    homingState = HOMING_IDLE;
    Serial.println(F("HOMING COMPLETE"));
  } else {
    Serial.println(F("HOMING: Homing Y..."));
    stepperY.setMaxSpeed(HOMING_SPEED);
    stepperY.moveTo(-99999);
    homingState = HOMING_Y;
  }
}

// Helper function for E-Stop
void stopAll() {
  // setCurrentPosition() zeros velocity immediately; stop()+moveTo() only decelerates,
  // letting the motor keep running and potentially reverse direction after overshooting.
  stepperX.setCurrentPosition(stepperX.currentPosition());
  stepperY.setCurrentPosition(stepperY.currentPosition());
  if (zActive) {
    // Credit however much of the pulse actually ran before the stop
    z_position_ms += (long)(millis() - zStartTime) * zDirection;
  }
  analogWrite(Z_RPWM, 0);
  analogWrite(Z_LPWM, 0);
  zActive = false;
  zSeatAbortArmed = false;
  homingState = HOMING_IDLE; // abort homing if in progress; hasHomed stays false
  isMovingSequential = false;
}
