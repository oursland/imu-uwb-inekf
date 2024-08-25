import numpy as np
from scipy import linalg
import math

GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD  = math.pi/180.0

def SO3_wedge(phi):
    return np.array([
        [       0, -phi[2],  phi[1] ],
        [  phi[2],       0, -phi[0] ],
        [ -phi[1],  phi[0],       0 ],
    ])

def SO3_vee(Phi):
    return np.array([
        Phi[2, 1],
        Phi[0, 2],
        Phi[1, 0],
    ])

def SO3_exp(phi):
    angle = np.linalg.norm(phi)

    if np.isclose(angle, 0.):
        return np.eye(3) + SO3_wedge(phi)

    axis = phi / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * SO3_wedge(axis)

def SO3_log(Phi):
    cos_angle = np.clip(0.5 * np.trace(Phi) - 0.5, -1., 1.)
    angle = np.arccos(cos_angle)

    if np.isclose(angle, 0.):
        return SO3_vee(Phi - np.eye(3))

    return SO3_vee((0.5 * angle / np.sin(angle)) * (Phi - Phi.T))

def SO3_left_jacobian(phi):
    angle = np.linalg.norm(phi)

    # near 0, use first order Taylor expansion
    if np.isclose(angle, 0):
        J = np.eye(3) - 0.5 * SO3_wedge(phi)
        return J

    axis = phi / angle
    s = np.sin(angle)
    c = np.cos(angle)
    J = (s / angle)*np.eye(3) + (1 - s / angle)*np.outer(axis, axis) + ((1 - c) / angle)*SO3_wedge(axis)
    return J

def SO3_inv_left_jacobian(phi):
    angle = np.linalg.norm(phi)

    # near 0, use first order Taylor expansion
    if np.isclose(angle, 0):
        J = np.eye(3) - 0.5 * SO3_wedge(phi)
        return J

    axis = phi / angle
    a2 = angle / 2
    cot = 1 / np.tan(a2)
    J = (a2 * cot)*np.eye(3) + (1 - a2 * cot)*np.outer(axis, axis) - a2*SO3_wedge(axis)
    return J

def SE3_2_wedge(xi):
    """SE_2(3) wedge operation"""
    Xi = np.zeros([5, 5])
    Xi[:3, :3] = SO3_wedge(xi[:3])
    Xi[:3, 3] = xi[3:6]
    Xi[:3, 4] = xi[6:9]
    return Xi

def SE3_2_vee(Xi):
    """SE_2(3) vee operation"""
    xi = np.concatenate((SO3_vee(Xi[:3, :3]), Xi[:3, 3], Xi[:3, 4]))
    return xi

def SE3_2_vec(Xi):
    """SE_2(3) to vector of [ R_x, R_y, R_z, vx, vy, vz, x, y, z ] operation"""
    xi = np.concatenate((SO3_log(Xi[:3, :3]), Xi[:3, 3], Xi[:3, 4]))
    return xi

def SE3_2_exp(xi):
    """SE_2(3) exp() operation"""
    T = np.zeros([5, 5])
    T[3, 3] = 1
    T[4, 4] = 1
    T[:3, :3] = SO3_exp(xi[:3])
    tmp = np.zeros([3, 2])
    tmp[:, 0] = xi[3:6]
    tmp[:, 1] = xi[6:9]
    T[:3, 3:] = SO3_left_jacobian(xi[:3])@tmp
    return T

def SE3_2_inv(Xi):
    """SE_2(3) inv() operation"""
    T = np.zeros([5, 5])
    T[:3, :3] = Xi[:3,:3].T
    T[:3, 3] = -T[:3, :3]@Xi[:3, 3]
    T[:3, 4] = -T[:3, :3]@Xi[:3, 4]
    T[3, 3] = 1
    T[4, 4] = 1
    return T

def SE3_2_log(T):
    """SE_2(3) log() operation"""
    phi = SO3_log(T[:3, :3])
    Xi = SO3_inv_left_jacobian(phi)@T[:3, 3:5]
    xi = np.zeros(9)
    xi[:3] = phi
    xi[3:6] = Xi[:, 0]
    xi[6:9] = Xi[:, 1]
    return xi

def SE3_2_adjoint(T):
    """SE_2(3) adjoint operation"""
    Adjoint = np.zeros([9, 9])
    R = T[:3, :3]
    Adjoint[0:3, 0:3] = R
    Adjoint[3:6, 3:6] = R
    Adjoint[6:9, 6:9] = R
    Adjoint[3:6, 0:3] = SO3_wedge(T[:3, 3])@R
    Adjoint[6:9, 0:3] = SO3_wedge(T[:3, 4])@R
    return Adjoint

# exponentially-weighted variance
def ewv(avg, var, alpha, val):
    avg = alpha*val + (1 - alpha)*avg
    diff = val - avg
    var = alpha*diff**2 + (1 - alpha)*var
    return avg, var

class InEKF:
    def __init__(self, X0, P0, bias):
        # nominal-state X = SE3_2_exp([phi, theta, psi, vx, vy, vz, x, y, z])
        if X0.shape == (9,):
            self.X = SE3_2_exp(X0)
        else:
            self.X = X0
        self.X_hat = self.X

        self.P = P0
        self.P_hat = P0

        self.bias = bias

        # noise params (variance)
        self.w_gyr = np.ones(3) * 0.1**2
        self.w_acc = np.ones(3) * 2.0**2
        # self.w_gyr = np.ones(3) * 1e-8
        # self.w_acc = np.ones(3) * 1e-4
        self.w_tdoa = np.array([ np.sqrt(0.05) ])
        self.w_gyr_b = np.ones(3) * 0.000001**2
        self.w_acc_b = np.ones(3) * 0.0001**2
        # self.w_gyr_b = np.ones(3) * 1e-12
        # self.w_acc_b = np.ones(3) * 1e-8

        # print("IEKF Params: ")
        # print("    w_gyr = ", self.w_gyr)
        # print("    w_acc = ", self.w_acc)
        # print("    w_tdoa  = ", self.w_tdoa)
        # print("    w_gyr_b = ", self.w_gyr_b)
        # print("    w_acc_b = ", self.w_acc_b)
        # print()

        # external calibration: translation vector from the quadcopter to UWB tag
        self.t_uv = np.array([-0.01245, 0.00127, 0.0908])

        self.rej_cnt = 0

    # def predict(X, P, u, dt):
    #     X_hat = F(X, u, dt, np.zeros([2, 3]))
    #     A = A(X)
    #     Phi = SE3_2_exp(A*dt)
    #     L = Phi@SE3_2_adjoint(X)
    #     P_hat = Phi@P@Phi.T + L@Q@L.T*dt
    #     return X_hat, P_hat

    def predict(self, u, dt):
        """
        InEKF Prediction

        Args:
            u     (6 ndarray)     : IMU measurements, [acc, gyr]
            bias  (2,3 ndarray)   : estimated bias
            dt    float           : time delta

        Returns:
            X_hat (5,5 ndarray)   : X_{n+1}+ predicted state
            P_hat (15,15 ndarray) : P_{n+1}+ predicted state covariance
            bias  (6 ndarray)     : estimated bias
        """

        g = np.array([0, 0, -GRAVITY_MAGNITUDE])
        R = self.X[:3,:3]
        v = self.X[:3,3]
        p = self.X[:3,4]

        a     = u[0:3]*GRAVITY_MAGNITUDE - self.bias[0:3]
        omega = u[3:6]*DEG_TO_RAD - self.bias[3:6]

        Rnew = R@SO3_exp(omega*dt)
        vnew = v + (R@a + g)*dt
        pnew = p + v*dt + 0.5*(R@a + g)*dt**2

        # if CF is on the ground
        if pnew[2] < 0:
            pnew[2] = 0
            vnew = np.zeros(3)

        self.X_hat = np.block([
            [ Rnew, vnew.reshape(-1,1), pnew.reshape(-1,1) ],
            [ 0, 0, 0,               1,                  0 ],
            [ 0, 0, 0,               0,                  1 ],
        ])

        # make adjoint with biases
        I = np.eye(6)
        zero = np.zeros((9,6))
        adj_X = np.block([
            [ SE3_2_adjoint(self.X_hat), zero ],
            [                    zero.T,    I ],
        ])

        I       = np.eye(3)
        zero    = np.zeros((3,3))
        g_cross = SO3_wedge(g)
        R       = self.X_hat[:3,:3]
        A = np.block([
            [    zero, zero, zero,              -R, zero ],
            [ g_cross, zero, zero, -adj_X[3:6,0:3],   -R ],
            [    zero,    I, zero, -adj_X[6:9,0:3], zero ],
            [    zero, zero, zero,            zero, zero ],
            [    zero, zero, zero,            zero, zero ],
        ])
        Phi = linalg.expm(A*dt)

        # construct Q
        Q = np.block([
            [ np.diag(self.w_gyr),     np.zeros((3,3)), np.zeros((3,3)),       np.zeros((3,3)),       np.zeros((3,3)) ],
            [     np.zeros((3,3)), np.diag(self.w_acc), np.zeros((3,3)),       np.zeros((3,3)),       np.zeros((3,3)) ],
            [     np.zeros((3,3)),     np.zeros((3,3)), np.zeros((3,3)),       np.zeros((3,3)),       np.zeros((3,3)) ],
            [     np.zeros((3,3)),     np.zeros((3,3)), np.zeros((3,3)), np.diag(self.w_acc_b),       np.zeros((3,3)) ],
            [     np.zeros((3,3)),     np.zeros((3,3)), np.zeros((3,3)),       np.zeros((3,3)), np.diag(self.w_gyr_b) ],
        ])*dt**2

        self.P_hat = Phi@self.P@Phi.T + Phi@adj_X@Q@adj_X.T@Phi.T

        # print("X = ", self.X)
        # print("P = ", self.P)
        # print("Phi = ", Phi)
        # print("adj_X = ", adj_X)
        # print("Q = ", Q)
        # print("dt = ", dt)
        # print("X_hat = ", self.X_hat)
        # print("P_hat = ", self.P_hat)

        self.X = self.X_hat
        self.P = self.P_hat

        return self.X_hat, self.P_hat, self.bias

    # def correct(X_hat, P_hat, y):
    #     y_hat = G(X)
    #     z = X_hat.T@(y - y_hat) # Z = X@Y - b
    #     S = H@P@H.T + M@R@M.T
    #     K = P_hat@H.T@np.linalg.inv(S)
    #     P = (I - K@H)@P_hat@(I-K@H).T + K@M@R@M.T@K.T
    #     X = SEK3_exp(-SEK3_wedge(K@z))@X_hat
    #     return X, P

    def correct(self, uwb, anchor_position):
        """
        InEKF Correction

        Args:
            uwb             ([3] ndarray)    : UWB measurements, [ idA, idB, TDoA ]
            anchor_position ([n, 3] ndarray) : array of anchor positions

        Returns:
            X_hat (5,5 ndarray)   : X_{n+1} estimated state
            P_hat (15,15 ndarray) : P_{n+1} estimated state covariance
            bias  (6 ndarray)     : estimated bias
        """
        an_A = anchor_position[int(uwb[0]),:]   # idA
        an_B = anchor_position[int(uwb[1]),:]   # idB

        g = np.array([0, 0, -GRAVITY_MAGNITUDE])
        R = self.X_hat[:3,:3]
        v = self.X_hat[:3,3]
        p = self.X_hat[:3,4]

        p_uwb = R@self.t_uv + p         # uwb tag position
        d_A = linalg.norm(an_A - p_uwb) # distance to anchor A
        d_B = linalg.norm(an_B - p_uwb) # distance to anchor B

        y_hat = d_B - d_A               # predicted difference of distance
        y = uwb[2]                      # measured difference of distance
        z = np.array([y - y_hat])       # error

        # -------------------- Statistical Validation -------------------- #
        H = np.block([[ np.zeros(3), np.zeros(3), (an_B - p_uwb)/d_B - (an_A - p_uwb)/d_A, np.zeros(3), np.zeros(3) ]])
        M = H@self.P_hat@H.T + self.w_tdoa**2
        d_m = math.sqrt(z**2 / M)
        if d_m >= 5:
            # print("InEKF: rejecting z = ", z, ", d_m = ", d_m)
            # the measurement failed the Chi-squared test, keep the previous state
            self.rej_cnt += 1
            self.X = self.X_hat
            self.P = self.P_hat
            return self.X, self.P, self.bias, 1

        # convert to right-invariant measurement model
        I = np.eye(6)
        zero = np.zeros((9,6))
        adj_X = np.block([
            [ SE3_2_adjoint(SE3_2_inv(self.X_hat)), zero ],
            [                               zero.T,    I ],
        ])
        H = H@adj_X

        I = np.eye(15)
        N = self.w_tdoa**2
        S = H@self.P_hat@H.T + N
        K = self.P_hat@H.T@np.linalg.inv(S)
        self.P = (I - K@H)@self.P_hat@(I - K@H).T + K*N@K.T
        self.X = SE3_2_exp(-(K@z))@self.X_hat
        self.bias = self.bias + -(K@z)[9:15]

        return self.X, self.P, self.bias, 0
