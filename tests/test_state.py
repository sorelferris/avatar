import numpy as np
from src.state import TeleopState


def test_initial_state():
    state = TeleopState()
    assert state.left_hand_detected is False
    assert state.right_hand_detected is False
    assert state.left_target.shape == (3,)
    assert state.right_target.shape == (3,)
    assert np.allclose(state.left_target, 0.0)
    assert np.allclose(state.right_target, 0.0)


def test_thread_safe_update():
    state = TeleopState()
    target = np.array([0.1, 0.2, 0.3])
    state.update_left(target, detected=True)
    result, detected = state.get_left()
    assert detected is True
    assert np.allclose(result, target)


def test_hand_lost_keeps_last_position():
    state = TeleopState()
    target = np.array([0.1, 0.2, 0.3])
    state.update_left(target, detected=True)
    state.update_left(None, detected=False)
    result, detected = state.get_left()
    assert detected is False
    assert np.allclose(result, target)  # frozen
