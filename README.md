# 2D quadrotor: analytical cascade PD and constrained MPC (simulation)

A planar quadrotor flies between waypoints. Three controllers are compared on the same linear
model: LQR, an **analytical cascade PD** (derived by hand, see `docs/`), and a **linear MPC** with
thrust and torque limits, a hard tilt limit and a restricted zone it must not cross.
Everything is simulation of a linear model in Python (NumPy, SciPy, Matplotlib). No hardware,
noise or disturbances are used.

## Model (from my own modelling notes)

Parameters: m = 1.0 kg, J = 0.01 kg m^2, arm length l = 0.25 m, g = 9.81 m/s^2 (illustrative values,
not a specific vehicle). Inputs: total thrust u1 and torque u2 = (F2 - F1) l. Conventions: y to the
right, z up, theta positive anticlockwise.

```
y''     = -(u1/m) sin(theta)      a drone must tilt before it can move sideways
z''     =  (u1/m) cos(theta) - g
theta'' =  u2 / J
```

Linearised at hover (u1 = m g, sin(theta) ~ theta), the only non-trivial entry of A is -g. All six
eigenvalues of A are zero (repeated, so the system is unstable with polynomial growth) and the system
is controllable. `python model.py` checks A and B against a numerical linearisation of the nonlinear
model. The code orders the states as `[y, z, theta, vy, vz, omega]`; the hand-written notes use
`[y, vy, z, vz, theta, omega]`.

Transfer functions: Z/dU1 = 1/(m s^2), Theta/U2 = 1/(J s^2) = 100/s^2, Y/Theta = -g/s^2.

## Controllers (`controllers.py`)

- **LQR**: unconstrained state feedback, baseline.
- **CascadePD**: analytical design, no constraints (details below).
- **MPC**: prediction horizon 40 steps (0.8 s), 10 free input moves (then held), cost on
  position/velocity/tilt errors and on inputs. Limits |thrust deviation| <= 4 N and |torque| <= 0.2 N m
  (illustrative choices). Optional constraints at every predicted step: hard tilt limit
  |theta| <= 15 degrees, and restricted zone y <= 0.8 m. Without state constraints the problem is solved
  exactly (bounded least squares). With state constraints it is solved with SLSQP, with a slack variable
  and a very large penalty on each constraint so the problem is always solvable. The cost is scaled by
  1e-3: without scaling, SLSQP returned a worse (but "successful") solution at some steps, which I found
  by comparing it step by step with the exact solver.

## Analytical cascade PD (`docs/derivation_inner_outer_loops.pdf`)

Same method as for my maglev PID: closed loop 1 + G K = 0, characteristic polynomial, Routh-Hurwitz,
coefficient matching.

- Inner loop (attitude): J s^2 + Kdth s + Kpth = 0, with wn = 20 rad/s, zeta = 0.8:
  **Kpth = 4.0, Kdth = 0.32**.
- Outer loop (position): with the inner loop included, the characteristic polynomial is
  s^4 + 32 s^3 + 400 s^2 + 3924 Kdy s + 3924 Kpy = 0. Routh-Hurwitz gives 0 < Kdy < 3.26 and
  Kpy < 12.5 Kdy - 3.83 Kdy^2. Matching the dominant poles at wn = 2.5 rad/s, zeta = 0.9 gives
  **Kpy = 0.430, Kdy = 0.353**; poles at -2.25 +/- 1.09j and -13.75 +/- 9.0j.
- Altitude loop: m s^2 + Kdz s + Kpz = 0, with wn = 3 rad/s, zeta = 0.9: **Kpz = 9.0, Kdz = 5.4**.

`python pd_design_check.py` reproduces the verification in the PDF (0.5 m step, linear and
nonlinear model: peak tilt about 10 degrees, no overshoot, rotor thrusts stay positive).
A shortcut that assumes the inner loop is instant puts the position poles in the wrong place, so the
full cascade polynomial is used.

## Results (reproduce with `python run_experiments.py`)

**Open loop**, 1 degree tilt and no controller: the drone drifts 0.09 m, 0.34 m and 0.77 m
sideways after 1, 2 and 3 s. The growth is polynomial, not exponential, because the eigenvalues are zero.

**1 m sideways move, no constraints:**

| Controller | Peak tilt (deg) | Peak torque (N m) | Settles within 2 cm (s) |
|---|---|---|---|
| LQR | 27.6 | 0.82 | 1.44 |
| Analytical cascade PD | 21.4 | 1.72 | 1.98 |

Both ask for tilts well beyond the small-angle model and for torques above the MPC limit.
The PD gains were designed for a 0.5 m step (peak tilt about 10 degrees).

**Waypoint flight** (1,0) -> (1,0.5) -> (0,0.5), 3 s per leg:

| Variant | Peak tilt (deg) | Max y (m) | Leg 1: within 2 cm of target (s) |
|---|---|---|---|
| Analytical cascade PD (no limits) | 21.4 | 1.002 | 1.98 |
| MPC, input limits only | 23.7 | 1.000 | 1.68 |
| MPC + tilt limit 15 deg | 15.0 | 1.000 | 1.78 |
| MPC + tilt limit and wall y <= 0.8 m | 15.0 | 0.800 | target is behind the wall |

Limiting the tilt to 15 degrees costs about 0.1 s on the first leg. With the wall, the drone flies
to y = 0.8 m and holds there, because the targets at y = 1.0 are unreachable.

**Stress test**: wall at y = 0.8 m, drone already moving at 1 m/s toward it, target at the origin:

| Start y (m) | Max y reached (m) | Wall violation (mm) | Peak tilt (deg) |
|---|---|---|---|
| 0.0 | 0.311 | 0 | 15.0 |
| 0.5 | 0.806 | 6 | 15.2 |
| 0.7 | 1.001 | 201 | 25.7 |

With tilt limited to 15 degrees the drone decelerates at about g tan(15 deg) = 2.6 m/s^2, so stopping
from 1 m/s takes at least 0.19 m (0.31 m in the simulation, because it must rotate first). If it starts
too close and too fast, no controller can respect the wall. The soft constraints then violate both the
wall and the tilt limit, which compete.

Figures: `results/lqr_move.png`, `results/waypoints.png`, `results/pd_design_check.png`. Tables:
`results/summary.md`.

## Run

```
pip install -r requirements.txt
python model.py
python run_experiments.py
python pd_design_check.py
python -m pytest tests        # optional
```

## Limitations

- Linear model, valid for small angles only; no noise, disturbances or model mismatch.
- A single half-plane wall, not a true obstacle (a box is non-convex).
- SLSQP is a general-purpose solver. It took about 5 ms per step on a laptop in my tests, below the
  20 ms sample time, but it is not suited to embedded hardware and gives no feasibility guarantee.
- Recursive feasibility and stability are not proven; the slack variables only keep the optimisation solvable.
- Mass, inertia and arm length are illustrative; the input limits are my own choice.

## Next steps

- Nonlinear dynamics and nonlinear MPC.
- Model mismatch and noise, and a data-driven model update (as in my maglev MPC project).
- A dedicated QP solver for embedded use.

## Author

Muhammad Ibrahim Zubair, Kano, Nigeria.
