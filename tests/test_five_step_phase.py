import numpy as np

from microlens_defects.features.five_step_phase import compute_phase


def test_compute_phase_reconstructs_known_phase():
    h, w = 8, 8
    phi_true = np.full((h, w), 0.7, dtype=np.float32)  # radians
    amp = 100.0
    dc = 128.0
    shifts = [0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5]
    frames = []
    for s in shifts:
        frames.append(dc + amp * np.cos(phi_true + s))
    stack = np.stack(frames, axis=2).astype(np.float32)

    result = compute_phase(stack, amp_threshold=0.1)
    # phase difference modulo 2pi
    diff = np.angle(np.exp(1j * (result.phase - phi_true)))
    assert np.max(np.abs(diff)) < 1e-2
    assert np.all(result.mask == 255)
