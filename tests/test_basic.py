import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import A, B, AD, numerical_jacobians
from controllers import LQR, MPC, CascadePD, pd_gains, simulate


def test_linearisation_matches_nonlinear_model():
    An, Bn = numerical_jacobians()
    assert np.abs(An - A).max() < 1e-6 and np.abs(Bn - B).max() < 1e-6


def test_hover_is_controllable_and_marginal():
    Ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(6)])
    assert np.linalg.matrix_rank(Ctrb) == 6
    assert np.allclose(np.linalg.eigvals(A), 0, atol=1e-6)


def test_lqr_stabilises():
    X, _ = simulate(LQR(), np.zeros(6), [(1.0, 0.0)], steps_per_target=400)
    assert abs(X[-1, 0] - 1.0) < 1e-3


def test_tilt_limit_is_respected():
    X, _ = simulate(MPC(tilt_max_deg=15.0), np.zeros(6), [(1.0, 0.0)], steps_per_target=100)
    assert np.rad2deg(np.abs(X[:, 2]).max()) < 15.05


def test_wall_is_respected_from_rest():
    X, _ = simulate(MPC(tilt_max_deg=15.0, y_max=0.8), np.zeros(6), [(1.0, 0.0)], steps_per_target=150)
    assert X[:, 0].max() < 0.801


def test_cascade_gains_place_the_position_poles():
    k = pd_gains()
    roots = np.roots([1, 32, 400, 3924 * k["Kdy"], 3924 * k["Kpy"]])
    expected = np.array([-2.25 + 1.0897j, -2.25 - 1.0897j, -13.75 + 8.9967j, -13.75 - 8.9967j])
    assert np.allclose(np.sort_complex(roots), np.sort_complex(expected), atol=1e-2)


def test_cascade_pd_reaches_the_target():
    X, _ = simulate(CascadePD(), np.zeros(6), [(0.5, 0.3)], steps_per_target=300)
    assert abs(X[-1, 0] - 0.5) < 2e-3 and abs(X[-1, 1] - 0.3) < 2e-3
