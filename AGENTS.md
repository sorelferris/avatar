# avatar

Hand visual teleoperation for the SO101 robot arm (5-DOF) with optional SO102 (6-DOF) support. Camera-based hand tracking → Pinocchio IK → MuJoCo simulation.

## Quick start

```bash
uv sync          # install dependencies
uv run pytest    # run all tests
python main.py   # launch teleoperation
```

## Project structure

| Path | Purpose |
|---|---|
| `main.py` | Entrypoint — wires camera, detector, IK, sim, visualizer |
| `src/` | Library modules (CameraThread, HandDetector, IKSolver, SimEnvironment, CoordinateProcessor, Visualizer, TeleopState, JoyconManager) |
| `tests/` | pytest suite per module |
| `assets/SO101/` | SO101 (5-DOF) URDF + XML + STL files |
| `assets/SO102/` | SO102 (6-DOF, adds wrist_yaw) URDF + XML |

## Key facts

- **Python 3.12+ required** (`.python-version`)
- **Package manager**: `uv`. `uv.lock` is gitignored — run `uv sync` after pulling.
- **Import paths**: `from src.xxx import YYY` — resolved via `pythonpath = ["."]` in pyproject.toml.
- **No linter/formatter/typechecker configured** — install and configure ruff if you add one.
- **No CI pipeline** in the repo.

## Robot model variants

| Variant | DOF | URDF | Extra joints |
|---|---|---|---|
| SO101 (default) | 5 | `so101_new_calib.urdf` | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll |
| SO102 | 6 | `so102.urdf` | + wrist_yaw |

Both `IKSolver` and `SimEnvironment` auto-detect the variant by checking if `so102` is in the path.

## Model auto-download

On first run, the following are downloaded automatically:
- **MediaPipe Hand Landmarker** → `~/.cache/mediapipe/models/hand_landmarker.task`
- **MiDaS depth model** → `~/.cache/torch/hub/intel-isl_MiDaS_master` (cloned from GitHub)

These are one-time operations. No manual model download needed.

## Running tests

```bash
uv run pytest                   # all tests
uv run pytest tests/ -k so102   # SO102-specific tests
uv run pytest tests/test_state.py -v  # single file, verbose
```

Tests use URDF files from `assets/` — no hardware required.

## Hardware dependencies

| Hardware | Default | Notes |
|---|---|---|
| Camera | `camera_id=4` | Set in `main.py:27` — likely different on your machine |
| Joycon controller | Bluetooth | Via `pyjoycon`; test with `python test_joycon.py` |

## Architecture notes

- **IK is independent of MuJoCo**: `IKSolver` uses only Pinocchio. `SimEnvironment` wraps MuJoCo for simulation. Either can be swapped without touching the other.
- **Threading model**: `CameraThread` captures on a daemon thread; `TeleopState` provides thread-safe read/write with copies. Main loop runs detection → IK → simulation → rendering sequentially.
- **Gesture detection**: HandDetector returns `"fist"` (close gripper), `"open"` (open gripper), `"palm_closed"` (reset), or `None`.
- **Joycon module** (`src/joycon.py`): local import of `pyjoycon` inside `scan()` so missing hardware doesn't break imports. Works with both left/right Joycons via `PythonicJoyCon`.
- **Workspace clipping**: CoordinateProcessor clamps target to a spherical workspace (default `max_radius=0.36m`).
