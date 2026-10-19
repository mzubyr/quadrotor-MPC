"""Checks the cascaded PD design for the 2D quadrotor (m = 1.0 kg, J = 0.01 kg m^2, l = 0.25 m).

States x = [y, vy, z, vz, theta, omega]; inputs u1 (total thrust), u2 = (F2 - F1) l.
Control: altitude PD -> du1; position PD -> theta_ref; attitude PD -> u2.
The position gains come from exact coefficient matching on the full cascade polynomial.
Run: python quadrotor_pd_design_check.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp

m, J, l, g = 1.0, 0.01, 0.25, 9.81

# ---- design: match s^2 + 2 zeta wn s + wn^2 in each loop ----
wz, zz = 3.0, 0.9                      # altitude
Kpz, Kdz = m * wz**2, 2 * zz * wz * m
wth, zth = 20.0, 0.8                   # attitude (inner loop, fast)
Kpth, Kdth = J * wth**2, 2 * zth * wth * J
wy, zy = 2.5, 0.9                      # position (outer loop): exact coefficient matching on the cascade
# Cascade characteristic polynomial: s^4 + 2 zeta_th w_th s^3 + w_th^2 s^2 + g w_th^2 Kdy s + g w_th^2 Kpy = 0
# Desired: (s^2 + 2 zy wy s + wy^2)(s^2 + p s + q); solve p, q from the s^3 and s^2 coefficients, then Kdy, Kpy.
a1, a2 = 2 * zth * wth, wth**2
p_ = a1 - 2 * zy * wy
q_ = a2 - wy**2 - 2 * zy * wy * p_
Kdy = (2 * zy * wy * q_ + p_ * wy**2) / (g * wth**2)
Kpy = wy**2 * q_ / (g * wth**2)
print("Kpz = %.2f  Kdz = %.2f  Kpth = %.2f  Kdth = %.3f  Kpy = %.4f  Kdy = %.4f"
      % (Kpz, Kdz, Kpth, Kdth, Kpy, Kdy))

# ---- linear model and the closed loop: u = F x + Gr r, r = [y_ref, z_ref] ----
A = np.zeros((6, 6)); A[0, 1] = 1; A[1, 4] = -g; A[2, 3] = 1; A[4, 5] = 1
B = np.zeros((6, 2)); B[3, 0] = 1 / m; B[5, 1] = 1 / J
F = np.zeros((2, 6)); Gr = np.zeros((2, 2))
F[0, 2], F[0, 3], Gr[0, 1] = -Kpz, -Kdz, Kpz                    # du1 = Kpz (z_ref - z) - Kdz vz
# theta_ref = -Kpy (y_ref - y) + Kdy vy ;  u2 = Kpth (theta_ref - theta) - Kdth omega
F[1, 0], F[1, 1], F[1, 4], F[1, 5] = Kpth * Kpy, Kpth * Kdy, -Kpth, -Kdth
Gr[1, 0] = -Kpth * Kpy
Acl, Bcl = A + B @ F, B @ Gr
print("closed-loop poles:", np.round(np.linalg.eigvals(Acl), 2))

t = np.linspace(0, 4, 4001)
yr, zr = 0.5, 0.3
r = np.column_stack([yr * np.ones_like(t), zr * np.ones_like(t)])
_, _, x = signal.lsim(signal.StateSpace(Acl, Bcl, np.eye(6), np.zeros((6, 2))), r, t)

def metrics(sig, ref):
    bad = np.where(np.abs(sig - ref) > 0.02 * ref)[0]
    return max(0.0, (sig.max() - ref) / ref * 100), (t[bad[-1]] if len(bad) else 0.0)

os_y, ts_y = metrics(x[:, 0], yr); os_z, ts_z = metrics(x[:, 2], zr)
print("y: overshoot %.2f %%, settling %.2f s | z: overshoot %.2f %%, settling %.2f s" % (os_y, ts_y, os_z, ts_z))
du1 = -Kpz * (x[:, 2] - zr) - Kdz * x[:, 3]
th_ref = Kpy * x[:, 0] + Kdy * x[:, 1] - Kpy * yr
u2 = Kpth * (th_ref - x[:, 4]) - Kdth * x[:, 5]
u1 = m * g + du1
F1, F2 = (u1 - u2 / l) / 2, (u1 + u2 / l) / 2
print("peak tilt %.1f deg | peak du1 %.2f N | peak torque %.2f N m | F1 %.1f..%.1f N, F2 %.1f..%.1f N"
      % (np.rad2deg(np.abs(x[:, 4]).max()), np.abs(du1).max(), np.abs(u2).max(), F1.min(), F1.max(), F2.min(), F2.max()))

# ---- same controller on the NONLINEAR model ----
def f(_, s):
    y, vy, z, vz, th, om = s
    u1 = m * g + Kpz * (zr - z) - Kdz * vz
    thr = -Kpy * (yr - y) + Kdy * vy
    u2_ = Kpth * (thr - th) - Kdth * om
    return [vy, -u1 / m * np.sin(th), vz, u1 / m * np.cos(th) - g, om, u2_ / J]

s = solve_ivp(f, [0, 4], np.zeros(6), t_eval=t, rtol=1e-8, atol=1e-10)
print("nonlinear: y = %.5f, z = %.5f, peak tilt %.1f deg" % (s.y[0, -1], s.y[2, -1], np.rad2deg(np.abs(s.y[4]).max())))

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7))
ax[0].plot(t, x[:, 0], label="linear"); ax[0].plot(t, s.y[0], "--", label="nonlinear")
ax[0].axhline(yr, color="gray", ls=":"); ax[0].set_ylabel("y (m)"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(t, x[:, 2]); ax[1].plot(t, s.y[2], "--"); ax[1].axhline(zr, color="gray", ls=":")
ax[1].set_ylabel("z (m)"); ax[1].grid(True)
ax[2].plot(t, np.rad2deg(x[:, 4])); ax[2].plot(t, np.rad2deg(s.y[4]), "--")
ax[2].set_ylabel("tilt (deg)"); ax[2].set_xlabel("time (s)"); ax[2].grid(True)
plt.tight_layout()
plt.savefig("results/pd_design_check.png", dpi=150)
plt.show()
