"""Runs all quadrotor experiments; writes figures and a summary to results/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import AD, BD, TS
from controllers import LQR, MPC, CascadePD, pd_gains, simulate

OUT = "results"
os.makedirs(OUT, exist_ok=True)
lines = []


def log(s=""):
    print(s); lines.append(s)


def tilt_deg(X): return np.rad2deg(np.abs(X[:, 2]).max())


# 1. Open loop: 1 degree tilt, hover thrust, no torque
x = np.array([0, 0, np.deg2rad(1), 0, 0, 0.0]); ys = []
for _ in range(150):
    x = AD @ x; ys.append(x[0])
log("## Open loop (1 degree tilt, no controller)")
log(f"Sideways drift after 1 s, 2 s, 3 s: {ys[49]:.3f} m, {ys[99]:.3f} m, {ys[149]:.3f} m; tilt stays at 1 degree")

# 2. LQR and the analytical cascade PD: move 1 m sideways (no constraints)
log("\n## 1 m sideways move without constraints")
log("| Controller | Peak tilt (deg) | Peak torque (N m) | Settles within 2 cm (s) |")
log("|---|---|---|---|")
lqr_X = None
for name, ctrl in (("LQR", LQR()), ("Analytical cascade PD", CascadePD())):
    X, U = simulate(ctrl, np.zeros(6), [(1.0, 0.0)], steps_per_target=400)
    settle = next(i for i in range(len(X)) if np.all(np.abs(X[i:, 0] - 1.0) < 0.02)) * TS
    log(f"| {name} | {tilt_deg(X):.1f} | {np.abs(U[:, 1]).max():.2f} | {settle:.2f} |")
    if name == "LQR":
        lqr_X = X
k = pd_gains()
log("Cascade PD gains: " + ", ".join(f"{n} = {v:.4g}" for n, v in k.items()))

# 3. MPC waypoint flight: three variants
waypoints = [(1.0, 0.0), (1.0, 0.5), (0.0, 0.5)]
variants = {
    "Analytical cascade PD (no limits)": CascadePD(),
    "MPC, input limits only": MPC(),
    "MPC + tilt limit 15 deg": MPC(tilt_max_deg=15.0),
    "MPC + tilt limit and wall y <= 0.8 m": MPC(tilt_max_deg=15.0, y_max=0.8),
}
log("\n## MPC waypoint flight: (1,0) -> (1,0.5) -> (0,0.5), 3 s per leg")
log("| Variant | Peak tilt (deg) | Max y (m) | Leg 1: time within 2 cm of target (s) | End of legs (y, z) |")
log("|---|---|---|---|---|")
runs = {}
for name, ctrl in variants.items():
    X, U = simulate(ctrl, np.zeros(6), waypoints)
    runs[name] = X
    leg1 = X[:151]
    t1 = next((i for i in range(151) if np.all(np.abs(leg1[i:, 0] - 1.0) < 0.02)), None)
    ends = [tuple(float(round(v, 3)) for v in X[150 * (k + 1)][:2]) for k in range(3)]
    t1s = "not reached (target behind wall)" if t1 is None else f"{t1 * TS:.2f}"
    log(f"| {name} | {tilt_deg(X):.1f} | {X[:, 0].max():.3f} | {t1s} | {ends} |")

# 4. Stress test: already moving toward the wall
log("\n## Stress test: wall at y = 0.8 m, drone already moving at 1 m/s toward it (target: origin)")
log("| Start y (m) | Max y reached (m) | Wall violation (mm) | Peak tilt (deg) |")
log("|---|---|---|---|")
for y0 in (0.0, 0.5, 0.7):
    ctrl = MPC(tilt_max_deg=15.0, y_max=0.8)
    X, U = simulate(ctrl, np.array([y0, 0, 0, 1.0, 0, 0.0]), [(0.0, 0.0)], steps_per_target=200)
    viol = max(0.0, X[:, 0].max() - 0.8) * 1000
    log(f"| {y0} | {X[:, 0].max():.3f} | {viol:.0f} | {tilt_deg(X):.1f} |")
log("\nStopping from 1 m/s needs at least v^2/(2 a) with a = g tan(15 deg), i.e. about 0.19 m in the ideal case.")

# Figures
t = np.arange(len(lqr_X)) * TS
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, lqr_X[:, 0]); ax[0].set_ylabel("y (m)"); ax[0].grid(True)
ax[1].plot(t, np.rad2deg(lqr_X[:, 2])); ax[1].set_ylabel("tilt (deg)"); ax[1].set_xlabel("time (s)"); ax[1].grid(True)
fig.suptitle("LQR 1 m move (no constraints)")
plt.savefig(f"{OUT}/lqr_move.png", dpi=150); plt.close()

fig, ax = plt.subplots(2, 1, figsize=(6.4, 7))
for name, X in runs.items():
    ax[0].plot(X[:, 0], X[:, 1], label=name)
    ax[1].plot(np.arange(len(X)) * TS, np.rad2deg(X[:, 2]), label=name)
ax[0].axvline(0.8, color="r", ls="--"); ax[0].plot(*zip(*waypoints), "ro")
ax[0].set_xlabel("y (m)"); ax[0].set_ylabel("z (m)"); ax[0].legend(fontsize=7); ax[0].grid(True)
ax[1].axhline(15, color="gray", ls="--"); ax[1].axhline(-15, color="gray", ls="--")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("tilt (deg)"); ax[1].grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/waypoints.png", dpi=150); plt.close()

open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
