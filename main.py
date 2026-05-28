"""Hand visual teleoperation for SO101 robot arm.

Single-hand tracking with Pinocchio-based IK (independent of MuJoCo).
"""

import time

import cv2

from src.camera import CameraThread
from src.coordinate_processor import CoordinateProcessor
from src.hand_detector import HandDetector
from src.ik_solver import IKSolver
from src.sim_env import SimEnvironment
from src.visualizer import Visualizer

URDF = "assets/SO101/so101_new_calib.urdf"
XML = "assets/SO101/scene.xml"


def main() -> None:
    camera = CameraThread(camera_id=0, fps=30)
    detector = HandDetector(max_hands=1)
    processor = CoordinateProcessor(
        hand_range=0.3, robot_range=0.4, ema_alpha=0.3, max_radius=0.36
    )
    ik = IKSolver(URDF, damping=0.1, max_delta=0.1)
    sim = SimEnvironment(XML)
    viz = Visualizer(sim.model, sim.data)

    camera.start()

    print("Calibrating... Show your hand to the camera.")
    anchored = False
    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    try:
        viz.start_viewer()
    except Exception:
        print("MuJoCo viewer not available, running headless")

    running = True
    while running:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Detect single hand
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        left_lm, right_lm, left_tip, right_tip = detector.detect(frame_rgb, frame)

        # Use whichever hand is detected
        hand_tip = left_tip if left_tip is not None else right_tip
        hand_lm = left_lm if left_lm is not None else right_lm

        # Anchor on first detection
        if not anchored and hand_tip is not None:
            processor.set_anchor(hand_tip)
            anchored = True
            print("Anchored!")

        # Process coordinates
        target, detected = processor.process(hand_tip, hand_tip is not None)

        # IK + control
        if detected:
            q_current = sim.get_joint_positions()
            q_new = ik.solve(q_current, target)
            sim.set_control(q_new)

        # Step simulation (50Hz control = every 10 physics steps)
        for _ in range(10):
            sim.step()

        # FPS
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = frame_count / (time.time() - fps_timer)
            frame_count = 0
            fps_timer = time.time()

        # Visualize
        annotated = viz.draw_opencv(
            frame,
            hand_lm,
            None,
            target if detected else None,
            None,
            fps,
        )
        cv2.imshow("Hand Teleoperation", annotated)
        viz.update_viewer(
            target if detected else None,
            None,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

    camera.stop()
    detector.close()
    viz.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
