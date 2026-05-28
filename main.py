# main.py
import time
import cv2

from src.camera import CameraThread
from src.hand_detector import HandDetector
from src.coordinate_processor import CoordinateProcessor
from src.sim_env import SimEnvironment
from src.ik_solver import IKSolver
from src.visualizer import Visualizer


def main() -> None:
    # Initialize modules
    camera = CameraThread(camera_id=4, fps=30)
    detector = HandDetector(max_hands=2)
    left_proc = CoordinateProcessor(
        hand_range=0.3, robot_range=0.4, ema_alpha=0.3, max_radius=0.36
    )
    right_proc = CoordinateProcessor(
        hand_range=0.3, robot_range=0.4, ema_alpha=0.3, max_radius=0.36
    )
    sim = SimEnvironment("assets/g1_23dof_fixed.xml")
    ik = IKSolver(sim, damping=0.1, max_delta=0.1)
    viz = Visualizer(sim.model, sim.data)

    camera.start()

    # Anchor calibration
    print("Calibrating... Show your hands to the camera.")
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

        # Detect hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        left_lm, right_lm, left_tip, right_tip = detector.detect(frame_rgb, frame)

        # Anchor on first detection
        if not anchored and (left_tip is not None or right_tip is not None):
            if left_tip is not None:
                left_proc.set_anchor(left_tip)
            if right_tip is not None:
                right_proc.set_anchor(right_tip)
            anchored = True
            print("Anchored!")

        # Process coordinates
        left_target, left_det = left_proc.process(left_tip, left_tip is not None)
        right_target, right_det = right_proc.process(right_tip, right_tip is not None)

        # IK + control
        if left_det:
            q_left = ik.solve("left", left_target)
            torque_left = ik.compute_pd_torque("left", q_left)
            sim.set_control("left", torque_left)
        if right_det:
            q_right = ik.solve("right", right_target)
            torque_right = ik.compute_pd_torque("right", q_right)
            sim.set_control("right", torque_right)

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
            left_lm,
            right_lm,
            left_target if left_det else None,
            right_target if right_det else None,
            fps,
        )
        cv2.imshow("Hand Teleoperation", annotated)
        viz.update_viewer(
            left_target if left_det else None,
            right_target if right_det else None,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

    camera.stop()
    detector.close()
    viz.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
