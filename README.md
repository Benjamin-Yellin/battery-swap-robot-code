# battery-swap-robot-code

Control and computer-vision software for an autonomous three-axis Cartesian robot that
performs on-site battery swaps for field agricultural robots. Developed for a mechanical
engineering capstone project at Ben-Gurion University of the Negev (2026), in collaboration
with the Volcani Institute. See the project report for the full design, theory, and results.

The system is split across two controllers communicating over USB serial:

- **`raspberry_pi/`** — Python master. Computer vision (OpenCV / ArUco marker detection and
  pose estimation), the operation state machine, image-to-gantry geometry, and a Flask web
  dashboard for calibration, manual control, and monitoring.
- **`arduino_firmware/`** — Arduino Nano slave. Real-time stepper control for the X/Y gantry,
  Z linear-actuator pulses, limit-switch safety, homing, and the end-effector grip switch.

## Layout

```
raspberry_pi/
├── web_server.py         Flask entry point, background vision/decision thread, video feed
├── automation_logic.py   Operation state machine and per-profile vision calibration
├── vision_module.py      ArUco detection, pixel error, pose estimation
├── arduino_link.py       Serial layer; gantry position and grip-switch tracking
└── templates/index.html  Web dashboard

arduino_firmware/
├── src/arduino_gantry_controller.cpp
└── platformio.ini
```

## Running

The Pi software runs with `python3 web_server.py` (dependencies in
`raspberry_pi/requirements.txt`; the dashboard is served on port 5000). It falls back to a
simulation mode on a machine without the camera and serial hardware. The firmware builds and
uploads with [PlatformIO](https://platformio.org/) (`pio run --target upload`).

## License

MIT — see [LICENSE](LICENSE).
