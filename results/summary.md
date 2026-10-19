## Open loop (1 degree tilt, no controller)
Sideways drift after 1 s, 2 s, 3 s: -0.086 m, -0.342 m, -0.770 m; tilt stays at 1 degree

## 1 m sideways move without constraints
| Controller | Peak tilt (deg) | Peak torque (N m) | Settles within 2 cm (s) |
|---|---|---|---|
| LQR | 27.6 | 0.82 | 1.44 |
| Analytical cascade PD | 21.4 | 1.72 | 1.98 |
Cascade PD gains: Kpz = 9, Kdz = 5.4, Kpth = 4, Kdth = 0.32, Kpy = 0.43, Kdy = 0.3534

## MPC waypoint flight: (1,0) -> (1,0.5) -> (0,0.5), 3 s per leg
| Variant | Peak tilt (deg) | Max y (m) | Leg 1: time within 2 cm of target (s) | End of legs (y, z) |
|---|---|---|---|---|
| Analytical cascade PD (no limits) | 21.4 | 1.002 | 1.98 | [(1.002, 0.0), (1.0, 0.5), (-0.002, 0.5)] |
| MPC, input limits only | 23.7 | 1.000 | 1.68 | [(1.0, 0.0), (1.0, 0.5), (0.0, 0.5)] |
| MPC + tilt limit 15 deg | 15.0 | 1.000 | 1.78 | [(0.999, 0.0), (1.0, 0.5), (0.001, 0.5)] |
| MPC + tilt limit and wall y <= 0.8 m | 15.0 | 0.800 | not reached (target behind wall) | [(0.8, 0.0), (0.8, 0.5), (0.0, 0.5)] |

## Stress test: wall at y = 0.8 m, drone already moving at 1 m/s toward it (target: origin)
| Start y (m) | Max y reached (m) | Wall violation (mm) | Peak tilt (deg) |
|---|---|---|---|
| 0.0 | 0.311 | 0 | 15.0 |
| 0.5 | 0.806 | 6 | 15.2 |
| 0.7 | 1.001 | 201 | 25.7 |

Stopping from 1 m/s needs at least v^2/(2 a) with a = g tan(15 deg), i.e. about 0.19 m in the ideal case.
