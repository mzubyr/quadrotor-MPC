"""LQR baseline and a linear MPC with input limits, a tilt limit and a restricted zone."""
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import lsq_linear, minimize
from model import AD, BD, M_QUAD, I_QUAD, G

NU = 2


def lqr_gain(Q=(100, 100, 10, 10, 10, 1.0), R=(1.0, 100.0)):
    Q, R = np.diag(Q), np.diag(R)
    P = solve_discrete_are(AD, BD, Q, R)
    return np.linalg.solve(R + BD.T @ P @ BD, BD.T @ P @ AD)


class LQR:
    """State feedback u = -K (x - x_ref). No constraints."""
    def __init__(self, **kw):
        self.K = lqr_gain(**kw)

    def control(self, x, target):
        xr = np.array([target[0], target[1], 0, 0, 0, 0.0])
        return -self.K @ (x - xr)


class MPC:
    """Condensed linear MPC.

    cost = sum q-weighted (x_k - x_ref)^2 + r-weighted u^2, inputs held after Nc moves.
    Always: |dT| <= umax[0] (N), |tau| <= umax[1] (N m).
    Optional: |theta| <= tilt_max_deg at every predicted step, y <= y_max (restricted zone).

    Without state constraints the problem is solved exactly (bounded least squares).
    With state constraints it is solved with SLSQP; each constraint group has a slack
    variable with a very large penalty, so the problem is always solvable and the
    constraint is violated only when it cannot be satisfied.
    """

    def __init__(self, Np=40, Nc=10, q=(100, 100, 10, 10, 10, 1.0), r=(1.0, 100.0),
                 umax=(4.0, 0.2), tilt_max_deg=None, y_max=None, rho=1e9, scale=1e-3):
        self.Np, self.Nc = Np, Nc
        self.umax = np.array(umax, float)
        self.tilt_max = None if tilt_max_deg is None else np.deg2rad(tilt_max_deg)
        self.y_max = y_max
        self.rho, self.scale = rho, scale

        self.F = np.vstack([np.linalg.matrix_power(AD, i) for i in range(1, Np + 1)])
        G = np.zeros((6 * Np, NU * Nc))
        for i in range(1, Np + 1):
            for j in range(i):
                c = min(j, Nc - 1)          # after Nc moves the last input is held
                G[6*(i-1):6*i, NU*c:NU*c+NU] += np.linalg.matrix_power(AD, i-1-j) @ BD
        self.Qh = np.kron(np.eye(Np), np.diag(np.sqrt(q)))
        self.Rh = np.kron(np.eye(Nc), np.diag(np.sqrt(r)))
        self.M = np.vstack([self.Qh @ G, self.Rh])
        self.lb = np.tile(-self.umax, Nc)
        self.ub = -self.lb
        self.G_theta, self.G_y = G[2::6, :], G[0::6, :]

        self.n_slack = (self.tilt_max is not None) + (self.y_max is not None)
        self.constrained = self.n_slack > 0
        if self.constrained:
            nz = NU * Nc + self.n_slack
            self.bounds = [(self.lb[k], self.ub[k]) for k in range(NU * Nc)] + [(0, None)] * self.n_slack
            rows, k = [], 0
            ones = np.ones((Np, 1)); zeros = np.zeros((Np, self.n_slack - 1))
            def slack_cols(k):
                cols = np.zeros((Np, self.n_slack)); cols[:, k] = 1.0; return cols
            if self.tilt_max is not None:
                rows.append(np.hstack([-self.G_theta, slack_cols(k)]))
                rows.append(np.hstack([ self.G_theta, slack_cols(k)])); k += 1
            if self.y_max is not None:
                rows.append(np.hstack([-self.G_y, slack_cols(k)]))
            self.J = np.vstack(rows)
            self.z_prev = np.zeros(nz)

    def reset(self):
        if self.constrained:
            self.z_prev = np.zeros(NU * self.Nc + self.n_slack)

    def control(self, x, target):
        Xref = np.tile(np.array([target[0], target[1], 0, 0, 0, 0.0]), self.Np)
        rhs = np.concatenate([self.Qh @ (Xref - self.F @ x), np.zeros(NU * self.Nc)])
        if not self.constrained:
            return lsq_linear(self.M, rhs, bounds=(self.lb, self.ub), method="bvls").x[:NU]

        s, nU = self.scale, NU * self.Nc
        Ms, rs = self.M * np.sqrt(s), rhs * np.sqrt(s)
        f_th, f_y = (self.F @ x)[2::6], (self.F @ x)[0::6]

        def cost(z):
            e = Ms @ z[:nU] - rs
            return e @ e + self.rho * s * np.sum(z[nU:] ** 2)

        def grad(z):
            return np.concatenate([2 * Ms.T @ (Ms @ z[:nU] - rs), 2 * self.rho * s * z[nU:]])

        def cons(z):
            parts, k = [], 0
            if self.tilt_max is not None:
                th = f_th + self.G_theta @ z[:nU]
                parts += [self.tilt_max + z[nU + k] - th, self.tilt_max + z[nU + k] + th]; k += 1
            if self.y_max is not None:
                y = f_y + self.G_y @ z[:nU]
                parts.append(self.y_max + z[nU + k] - y)
            return np.concatenate(parts)

        res = minimize(cost, self.z_prev.copy(), jac=grad, bounds=self.bounds,
                       constraints=[{"type": "ineq", "fun": cons, "jac": lambda z: self.J}],
                       method="SLSQP", options={"maxiter": 100, "ftol": 1e-10})
        self.z_prev = res.x
        return res.x[:NU]


def pd_gains(wz=3.0, zz=0.9, wth=20.0, zth=0.8, wy=2.5, zy=0.9):
    """Gains of the analytical cascade PD design (see docs/derivation_inner_outer_loops.pdf).

    Altitude:  m s^2 + Kdz s + Kpz = 0                       -> Kpz = m wz^2,  Kdz = 2 zz wz m
    Attitude:  J s^2 + Kdth s + Kpth = 0                     -> Kpth = J wth^2, Kdth = 2 zth wth J
    Position:  s^4 + a1 s^3 + a2 s^2 + g w^2 Kdy s + g w^2 Kpy = 0, with a1 = 2 zth wth, a2 = wth^2,
               matched to (s^2 + 2 zy wy s + wy^2)(s^2 + p s + q).
    """
    Kpz, Kdz = M_QUAD * wz**2, 2 * zz * wz * M_QUAD
    Kpth, Kdth = I_QUAD * wth**2, 2 * zth * wth * I_QUAD
    a1, a2 = 2 * zth * wth, wth**2
    p = a1 - 2 * zy * wy
    q = a2 - wy**2 - 2 * zy * wy * p
    Kdy = (2 * zy * wy * q + p * wy**2) / (G * wth**2)
    Kpy = wy**2 * q / (G * wth**2)
    return dict(Kpz=Kpz, Kdz=Kdz, Kpth=Kpth, Kdth=Kdth, Kpy=Kpy, Kdy=Kdy)


class CascadePD:
    """Analytical baseline: altitude PD, position PD (outer) and attitude PD (inner). No constraints.

    State order [y, z, theta, vy, vz, omega]. Derivative terms act on the measurement.
    """
    def __init__(self, **kw):
        self.k = pd_gains(**kw)

    def control(self, x, target):
        y, z, th, vy, vz, om = x
        k = self.k
        dT = k["Kpz"] * (target[1] - z) - k["Kdz"] * vz
        th_ref = -k["Kpy"] * (target[0] - y) + k["Kdy"] * vy
        tau = k["Kpth"] * (th_ref - th) - k["Kdth"] * om
        return np.array([dT, tau])


def simulate(controller, x0, targets, steps_per_target=150):
    """Closed loop on the discretised linear model. targets: list of (y, z) waypoints."""
    if hasattr(controller, "reset"):
        controller.reset()
    x = np.array(x0, float)
    X, U = [x.copy()], []
    for tgt in targets:
        for _ in range(steps_per_target):
            u = controller.control(x, tgt)
            x = AD @ x + BD @ u
            X.append(x.copy()); U.append(u)
    return np.array(X), np.array(U)
