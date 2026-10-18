# Robot hardware

Code for the real-robot platform: a Realhand A7 dual arm with an L6 dexterous hand on each
arm and a head-mounted RealSense D435i. It covers the 3D viewer, hand-eye calibration,
teleoperation with Wuji data gloves and Vive Trackers, and the L6 forward kinematics used to
map the hand to 21 MANO keypoints. The A7 and L6 URDF and mesh files are from
[Realhand](https://github.com/RealHand-Robotics).

```
drivers/arm.py          arm connection (wraps the SDK)
drivers/robot/          joint conventions, L6 CAN driver, dual_a7.urdf
drivers/tools/          viewer, calibration and projection tools
drivers/calib/          ACTIVE.json: the calibration in use
teleop/                 teleoperation console, control loops, recording
glove/                  Wuji glove driver, glove -> L6 calibration
vive/                   Vive Tracker server, SteamVR set-up
hand/l6_fk.py           L6 channels -> 21 keypoints (numpy only)
models/                 arm and hand meshes, L6 URDFs
calib_target/           printable ChArUco board
```

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/RealHand-Robotics/realbot-python-sdk.git
sudo ip link set can0 up type can bitrate 1000000   # one CAN bus per arm / hand
```

Joint angles are in radians and in motor order (shoulder pitch, roll, yaw, elbow pitch,
wrist yaw, pitch, roll). Joint limits and the hand mounting transform are defined in
`drivers/robot/conventions.py`.

## 3D viewer

```bash
cd drivers && python tools/sim_viewer.py
```

Shows both arms and hands in the browser; type commands such as `r4 45`, `rh 255 255 80 80 80 80`
or `fist` in the terminal (full list in the file docstring). No dynamics, no collision checking.

## Hand-eye calibration

Estimates the camera pose and the joint zero offsets of the arm from a ChArUco board held by
the arm.

1. Print `calib_target/charuco_5x5_28mm/charuco_5x5_28mm_A4.pdf` at 100 % and mount it on the
   end of the arm.
2. Start the camera service and the calibration panel, then record 30 or more samples at
   poses spread over the workspace (button "record sample"):
   ```bash
   cd drivers
   python tools/cam_server.py               # http://127.0.0.1:8850/
   python tools/arm_jog.py --side right     # http://127.0.0.1:8840/  (--sim-only without hardware)
   ```
3. Solve, then set the path of `joint_result.json` in `drivers/calib/ACTIVE.json`:
   ```bash
   python tools/handeye_solve.py       calib/handeye_<timestamp>           # initial guess
   python tools/handeye_joint_solve.py calib/handeye_<timestamp> --scale   # final result
   ```

To check a calibration on a recorded video, run `tools/project_arm.py` / `project_hand.py`
and then `overlay_arm.py` / `overlay_hand.py` on the recording directory.

## Teleoperation

Each Wuji glove drives the L6 hand on its side; a Vive Tracker on each wrist drives the arm
(1:1 displacement and orientation, joint-space IK). Space bar is the clutch.

1. Vive: install SteamVR, run `python vive/setup_steamvr.py` once (no headset needed), start
   SteamVR. `teleop/app.py` starts `vive/tracker_server.py` itself (set `VIVE_PYTHON` if
   `openvr` lives in another environment).
2. Glove calibration, once per glove: `python glove/l6_calibrate.py --hand right` (six hand
   poses), or `python teleop/hand_panel.py --hand right --can <can>` (robot/human pose pairs).
3. The Vive frame is assumed to be aligned with the robot base frame: give the rotation as
   `R_base_from_world` (3x3) in `vive/calib/arm_vive_extrinsics.json`.
4. Console:
   ```bash
   python teleop/app.py
   ```
   Move the arms to the ready pose, tick "enable clutch", start, then press space to engage.
   A hand moving faster than the lock threshold (1.5 m/s) locks both sides. Recordings go to
   `recordings/<time>/` (set `CAMERA_PYTHON` if `pyrealsense2` lives in another environment).

## L6 hand keypoints

```python
from l6_fk import l6_fk
pts = l6_fk([255, 255, 255, 255, 255, 255], side="right")  # (21, 3) m, hand base frame
```

Channels are 0-255 (255 = extended). `python hand/l6_fk.py --selftest` checks it.

## Safety

There is no collision checking. Watch the robot while it moves. `arm.close()` leaves the
motors enabled.
