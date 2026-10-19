"""Planar (2D) quadrotor model, linearised around hover.

States:  [y, z, theta, vy, vz, omega]  (sideways position, height, tilt, and their rates)
Inputs:  [dT, tau]                      (thrust deviation from hover thrust m*g, torque)

Parameters come from my own modelling notes (illustrative values, not a specific vehicle):
m = 1.0 kg, J = 0.01 kg m^2, arm length l = 0.25 m, so u2 = (F2 - F1) l and F1, F2 = (u1 -/+ u2/l)/2.
Sign convention: y to the right, z up, theta positive anticlockwise (a positive theta gives a negative y'').
"""
import numpy as np
from scipy.signal import cont2discrete

M_QUAD, I_QUAD, L_ARM, G = 1.0, 0.01, 0.25, 9.81   # kg, kg m^2, m, m/s^2
TS = 0.02                                # sampling time (s)


def f(x, u):
    """Nonlinear dynamics. u = [T, tau] with T the TOTAL thrust."""
    y, z, th, vy, vz, om = x
    T, tau = u
    return np.array([vy, vz, om,
                     -T * np.sin(th) / M_QUAD,
                     T * np.cos(th) / M_QUAD - G,
                     tau / I_QUAD])


# Linearisation at hover (theta = 0, T = m g): sin(theta) ~ theta, cos(theta) ~ 1
A = np.zeros((6, 6))
A[0, 3] = A[1, 4] = A[2, 5] = 1.0
A[3, 2] = -G
B = np.zeros((6, 2))
B[4, 0] = 1.0 / M_QUAD
B[5, 1] = 1.0 / I_QUAD


def discretize(ts=TS):
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(6), np.zeros((6, 2))), ts)
    return Ad, Bd


AD, BD = discretize()


def numerical_jacobians(h=1e-6):
    """Central-difference Jacobians of the nonlinear model at hover (to check A and B)."""
    xe, ue = np.zeros(6), np.array([M_QUAD * G, 0.0])
    An = np.zeros((6, 6)); Bn = np.zeros((6, 2))
    for j in range(6):
        d = np.zeros(6); d[j] = h
        An[:, j] = (f(xe + d, ue) - f(xe - d, ue)) / (2 * h)
    for j in range(2):
        d = np.zeros(2); d[j] = h
        Bn[:, j] = (f(xe, ue + d) - f(xe, ue - d)) / (2 * h)
    return An, Bn


if __name__ == "__main__":
    An, Bn = numerical_jacobians()
    print("max |A - A_numerical| =", np.abs(A - An).max())
    print("max |B - B_numerical| =", np.abs(B - Bn).max())
    print("eigenvalues of A:", np.round(np.linalg.eigvals(A), 6))
    Ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(6)])
    print("controllability rank:", np.linalg.matrix_rank(Ctrb), "of 6")
