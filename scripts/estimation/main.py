#!/usr/bin/env python3

'''
    The main file for eskf estimation
'''
import argparse
import os, sys
import math
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from scipy import interpolate
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, './../'))
from utility.praser import extract_gt, extract_tdoa, extract_acc, extract_gyro, interp_meas, extract_tdoa_meas

from eskf_class import ESKF
from inekf import InEKF, SO3_exp, SO3_log, SE3_2_vec

from plot_eskf import *
from plot_inekf import *
from plot_traj import *

np.set_printoptions(precision=6)
np.set_printoptions(floatmode='fixed')
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(sign=' ')

def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float64)
    return np.array(*args, **kwargs)

def zeros(*args, **kwargs):
    kwargs.setdefault("dtype", np.float64)
    return np.zeros(*args, **kwargs)

def ones(*args, **kwargs):
    kwargs.setdefault("dtype", np.float64)
    return np.ones(*args, **kwargs)

def eye(*args, **kwargs):
    kwargs.setdefault("dtype", np.float64)
    return np.eye(*args, **kwargs)

def block(*args, **kwargs):
    return np.block(*args, **kwargs).astype(np.float64)

def concatenate(*args, **kwargs):
    return np.concatenate(*args, **kwargs)

def diag(*args, **kwargs):
    return np.diag(*args, **kwargs).astype(np.float64)

def isin(t_np, t_k):
    '''
        help function for timestamp
    '''
    # check if t_k is in the numpy array t_np.
    # If t_k is in t_np, return the index and bool = True.
    # else return 0 and bool = False
    if t_k in t_np:
        res = np.where(t_np == t_k)
        return res[0][0], True
    return 0, False

def downsamp(data):
    '''
        down-sample uwb data
    '''
    data_ds = data
    # data_ds = data_ds[0::2,:]           # downsample by half
    # data_ds = data_ds[0::2,:]           # downsample by half
    # data_ds = data_ds[0::2,:]           # downsample by half
    # data_ds = data_ds[0::2,:]           # downsample by half
    return data_ds

def from_quaternion(quat, ordering='wxyz'):
    if ordering == 'xyzw':
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]
    elif ordering == 'wxyz':
        qw = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]

    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    # Form the matrix
    mat = zeros((3, 3))

    mat[0, 0] = 1. - 2. * (qy2 + qz2)
    mat[0, 1] = 2. * (qx * qy - qw * qz)
    mat[0, 2] = 2. * (qw * qy + qx * qz)

    mat[1, 0] = 2. * (qw * qz + qx * qy)
    mat[1, 1] = 1. - 2. * (qx2 + qz2)
    mat[1, 2] = 2. * (qy * qz - qw * qx)

    mat[2, 0] = 2. * (qx * qz - qw * qy)
    mat[2, 1] = 2. * (qw * qx + qy * qz)
    mat[2, 2] = 1. - 2. * (qx2 + qy2)
    return mat

if __name__ == "__main__":
    # load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', nargs=2)
    args = parser.parse_args()

    # access the survey results
    anchor_npz = args.i[0]
    anchor_survey = np.load(anchor_npz)
    # select anchor constellations
    anchor_position = anchor_survey['an_pos']
    # print out
    anchor_file = os.path.split(sys.argv[-2])[1]
    print("\nselecting anchor constellation " + str(anchor_file) + "\n")

    # access csv
    csv_file = args.i[1]
    df = pd.read_csv(csv_file)
    csv_name = os.path.split(sys.argv[-1])[1]
    print("ESKF estimation with: " + str(csv_name) + "\n")

    TDOA2 = True

    # --------------- extract csv file --------------- #
    gt_pose = extract_gt(df)
    tdoa    = extract_tdoa(df)
    acc     = extract_acc(df)
    gyr     = extract_gyro(df)
    #
    t_vicon = gt_pose[:,0]
    pos_vicon = gt_pose[:,1:4]
    ori_vicon = gt_pose[:,4:8]
    # t_tdoa = tdoa[:,0]
    # uwb_tdoa  = tdoa[:,1:]

    t_imu = acc[:,0]
    gyr_x_syn = interp_meas(gyr[:,0], gyr[:,1], t_imu).reshape(-1,1)
    gyr_y_syn = interp_meas(gyr[:,0], gyr[:,2], t_imu).reshape(-1,1)
    gyr_z_syn = interp_meas(gyr[:,0], gyr[:,3], t_imu).reshape(-1,1)

    imu = concatenate((acc[:,1:], gyr_x_syn, gyr_y_syn, gyr_z_syn), axis = 1)

    min_t = min(tdoa[0,0], t_imu[0], t_vicon[0])
    # get the vicon information from min_t
    t_vicon = array(t_vicon)
    idx = np.argwhere(t_vicon > min_t)
    t_vicon = np.squeeze(t_vicon[idx])
    pos_vicon = np.squeeze(array(pos_vicon)[idx,:])

    # reset time base
    t_vicon = (t_vicon - min_t)
    t_imu = (t_imu - min_t).reshape(-1,1)
    tdoa[:,0] = tdoa[:,0] - min_t

    # ------ downsample the raw data
    t_imu = downsamp(t_imu)
    imu   = downsamp(imu)

    if TDOA2:
        # extract tdoa meas.
        tdoa_70, tdoa_01, tdoa_12, tdoa_23, tdoa_34, tdoa_45, tdoa_56, tdoa_67 = extract_tdoa_meas(tdoa[:,0], tdoa[:,1:4])

        # downsample uwb tdoa data
        tdoa_70_ds = downsamp(tdoa_70)
        tdoa_01_ds = downsamp(tdoa_01)
        tdoa_12_ds = downsamp(tdoa_12)
        tdoa_23_ds = downsamp(tdoa_23)
        tdoa_34_ds = downsamp(tdoa_34)
        tdoa_45_ds = downsamp(tdoa_45)
        tdoa_56_ds = downsamp(tdoa_56)
        tdoa_67_ds = downsamp(tdoa_67)

        # convert back to tdoa
        tdoa_c = concatenate((
            tdoa_70_ds, tdoa_01_ds, tdoa_12_ds, tdoa_23_ds,
            tdoa_34_ds, tdoa_45_ds, tdoa_56_ds, tdoa_67_ds,
            ), axis = 0)
    else:
        tdoa_c = downsamp(tdoa)

    sort_id=np.argsort(tdoa_c[:,0])
    t_uwb = tdoa_c[sort_id, 0].reshape(-1,1)
    uwb   = tdoa_c[sort_id, 1:4]

    # ----------------------- INITIALIZATION OF EKF -------------------------#
    # initial estimate for the covariance matrix
    std_xy0 = 0.1
    std_z0 = 0.1
    std_vel0 = 0.01
    std_rp0 = 0.01
    std_yaw0 = 1.0
    std_b_acc = 0.0001   # accel bias std
    std_b_gyr = 0.000001 # gyro bias std
    var_tdoa = 0.13
    std_tdoa = np.sqrt(var_tdoa)

    print("Initial Covariance Params:")
    print("    std_xy0 = ", std_xy0)
    print("    std_z0 = ", std_z0)
    print("    std_vel0  = ", std_vel0 )
    print("    std_rp0 = ", std_rp0)
    print("    std_yaw0  = ", std_yaw0 )
    print("    std_b_acc = ", std_b_acc)
    print("    std_b_gyr = ", std_b_gyr)
    print()

    delta_pos = array([ 0.0, 0.0, 0.0 ])
    delta_yaw = 0*np.pi/16

    print("Initial State Offset:")
    print("    delta_pos = ", delta_pos)
    print("    delta_yaw = ", delta_yaw/(np.pi/16), "/16 𝜋")
    print()

    pos0 = pos_vicon[0] + delta_pos
    vel0 = array([ 0.0, 0.0, 0.0 ])
    ori0 = SO3_log(from_quaternion(ori_vicon[0], 'xyzw'))
    ori0q = Quaternion(block([ori_vicon[0][3], ori_vicon[0][0:3]]))

    # Create a compound vector t with a sorted merge of all the sensor time bases
    time = np.sort(concatenate((t_imu, t_uwb)))
    t = np.unique(time)
    K = t.shape[0]
    # Initial estimate for the state vector
    X0 = zeros((6,1))
    X0[0:3,0] = pos0
    X0[3:6,0] = vel0

    # introduce yaw error
    q0 = ori0q # initial quaternion
    q0 = ori0q*Quaternion(axis=(0.0, 0.0, 1.0), radians=delta_yaw)

    # Initial posterior covariance
    P0 = diag([
        std_xy0**2,   std_xy0**2,   std_z0**2,
        std_vel0**2,  std_vel0**2,  std_vel0**2,
        std_rp0**2,   std_rp0**2,   std_yaw0**2,
    ])

    # create the object of ESKF
    eskf = ESKF(X0, q0, P0, K)

    # x = array([ Rx, Ry, Rz, Vx, Vy, Vz,   Px,  Py,   Pz ])
    X0 = eye(5)
    X0[:3,:3] = SO3_exp(ori0 + [ 0, 0, delta_yaw ])
    X0[:3,3] = vel0
    X0[:3,4] = pos0

    P0 = diag([
        std_rp0**2,  std_rp0**2,  std_yaw0**2,
        std_vel0**2, std_vel0**2, std_vel0**2,
        std_xy0**2,  std_xy0**2,  std_z0**2,
        std_b_acc**2, std_b_acc**2, std_b_acc**2,
        std_b_gyr**2, std_b_gyr**2, std_b_gyr**2,
    ])
    # bias = np.mean(imu[0:100], axis=0) - array([0, 0, 1, 0, 0, 0])
    bias = zeros(6)
    bias[0:3] *= 9.81
    bias[3:6] *= math.pi/180.0
    inekf = InEKF(X0, P0, bias)

    eskf.std_uwb_tdoa = std_tdoa
    inekf.w_tdoa = np.array([std_tdoa])

    inekf_X = [ SE3_2_vec(X0) ]
    inekf_P = [ P0 ]

    t_imu_prev = 0

    print('timestep: %f' % K)
    print('\nStart state estimation')
    for k in range(1,K):
        # Find what measurements are available at the current time (help function: isin() )
        imu_k, imu_check = isin(t_imu, t[k-1])
        uwb_k, uwb_check = isin(t_uwb, t[k-1])
        dt = t[k]-t[k-1]

        imu_dt = t[k] - t_imu_prev
        if imu_check:
            t_imu_prev = t[k]

        # ESKF Prediction
        eskf.predict(imu[imu_k,:], dt, imu_check, k)

        # InEKF Prediction
        if imu_check:
            X_hat, P_hat, bias = inekf.predict(imu[imu_k,:] + array([ 0, 0/9.81, 0, 0, 0, 0*(180/np.pi) ]), imu_dt)
            inekf_X.append(SE3_2_vec(X_hat))
            inekf_P.append(P_hat)

        if uwb_check:
            eskf.correct(uwb[uwb_k,:], anchor_position, k)

            X, P, bias, _ = inekf.correct(uwb[uwb_k,:], anchor_position)
            inekf_X.append(SE3_2_vec(X))
            inekf_P.append(P)

    print('ESKF rej_cnt = ', eskf.rej_cnt)
    print('IEKF rej_cnt = ', inekf.rej_cnt)

    print('Finish the state estimation\n')

    inekf_X = array(inekf_X)
    inekf_P = array(inekf_P)

    ## compute the error
    # interpolate Vicon measurements
    f_x = interpolate.splrep(t_vicon, pos_vicon[:,0], s = 0.5)
    f_y = interpolate.splrep(t_vicon, pos_vicon[:,1], s = 0.5)
    f_z = interpolate.splrep(t_vicon, pos_vicon[:,2], s = 0.5)
    x_interp = interpolate.splev(t, f_x, der = 0)
    y_interp = interpolate.splev(t, f_y, der = 0)
    z_interp = interpolate.splev(t, f_z, der = 0)

    vel_vicon = zeros((len(t_vicon), 3))
    vel_vicon[:,0] = interpolate.splev(t_vicon, f_x, der = 1)
    vel_vicon[:,1] = interpolate.splev(t_vicon, f_y, der = 1)
    vel_vicon[:,2] = interpolate.splev(t_vicon, f_z, der = 1)
    vx_interp = interpolate.splev(t, f_x, der = 1)
    vy_interp = interpolate.splev(t, f_y, der = 1)
    vz_interp = interpolate.splev(t, f_z, der = 1)

    ori_v = []
    for q in ori_vicon:
        ori_v.append(SO3_log(from_quaternion(q, 'xyzw')))
    ori_v = array(ori_v)

    f_ori_x = interpolate.splrep(t_vicon, ori_v[:,0], s = 0.5)
    f_ori_y = interpolate.splrep(t_vicon, ori_v[:,1], s = 0.5)
    f_ori_z = interpolate.splrep(t_vicon, ori_v[:,2], s = 0.5)
    ori_x_interp = interpolate.splev(t, f_ori_x, der = 0)
    ori_y_interp = interpolate.splev(t, f_ori_y, der = 0)
    ori_z_interp = interpolate.splev(t, f_ori_z, der = 0)

    x_error = eskf.Xpo[:,0] - x_interp
    y_error = eskf.Xpo[:,1] - y_interp
    z_error = eskf.Xpo[:,2] - z_interp
    pos_error = concatenate((x_error.reshape(-1,1), y_error.reshape(-1,1), z_error.reshape(-1,1)), axis = 1)
    rms_x = math.sqrt(mean_squared_error(x_interp, eskf.Xpo[:,0]))
    rms_y = math.sqrt(mean_squared_error(y_interp, eskf.Xpo[:,1]))
    rms_z = math.sqrt(mean_squared_error(z_interp, eskf.Xpo[:,2]))
    print('ESKF: The RMS error for position x is %f [m]' % rms_x)
    print('ESKF: The RMS error for position y is %f [m]' % rms_y)
    print('ESKF: The RMS error for position z is %f [m]' % rms_z)
    RMS_all = math.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
    print('ESKF: The overall RMS error of position estimation is %f [m]\n' % RMS_all)

    vx_error = eskf.Xpo[:,3] - vx_interp
    vy_error = eskf.Xpo[:,4] - vy_interp
    vz_error = eskf.Xpo[:,5] - vz_interp
    vel_error = concatenate((vx_error.reshape(-1,1), vy_error.reshape(-1,1), vz_error.reshape(-1,1)), axis = 1)
    rms_vx = math.sqrt(mean_squared_error(vx_interp, eskf.Xpo[:,3]))
    rms_vy = math.sqrt(mean_squared_error(vy_interp, eskf.Xpo[:,4]))
    rms_vz = math.sqrt(mean_squared_error(vz_interp, eskf.Xpo[:,5]))
    print('ESKF: The RMS error for velocity x is %f [m/s]' % rms_vx)
    print('ESKF: The RMS error for velocity y is %f [m/s]' % rms_vy)
    print('ESKF: The RMS error for velocity z is %f [m/s]' % rms_vz)
    RMS_all = math.sqrt(rms_vx**2 + rms_vy**2 + rms_vz**2)
    print('ESKF: The overall RMS error of velocity estimation is %f [m/s]\n' % RMS_all)

    r_list = []
    for q in eskf.q_list:
        r_list.append(SO3_log(from_quaternion(q, 'wxyz')))
    r_list = array(r_list)

    ox_error = r_list[:,0] - ori_x_interp
    oy_error = r_list[:,1] - ori_y_interp
    oz_error = r_list[:,2] - ori_z_interp
    ori_error = concatenate((ox_error.reshape(-1,1), oy_error.reshape(-1,1), oz_error.reshape(-1,1)), axis = 1)
    rms_ox = math.sqrt(mean_squared_error(ori_x_interp, r_list[:,0]))
    rms_oy = math.sqrt(mean_squared_error(ori_y_interp, r_list[:,1]))
    rms_oz = math.sqrt(mean_squared_error(ori_z_interp, r_list[:,2]))
    print('ESKF: The RMS error for orientation x is %f [rad]' % rms_ox)
    print('ESKF: The RMS error for orientation y is %f [rad]' % rms_oy)
    print('ESKF: The RMS error for orientation z is %f [rad]' % rms_oz)
    RMS_all = math.sqrt(rms_ox**2 + rms_oy**2 + rms_oz**2)
    print('ESKF: The overall RMS error of orientation estimation is %f [rad]\n' % RMS_all)

    # visualization
    plot_pos_eskf(t, eskf.Xpo, t_vicon, pos_vicon)
    plot_vel_eskf(t, eskf.Xpo, t_vicon, vel_vicon)
    plot_ori_eskf(t, eskf.q_list, t_vicon, ori_vicon)
    plot_pos_err_eskf(t, pos_error, eskf.Ppo)
    plot_vel_err_eskf(t, vel_error, eskf.Ppo)
    plot_ori_err_eskf(t, ori_error, eskf.Ppo)
    plot_traj_eskf(pos_vicon, eskf.Xpo, anchor_position)

    # InEKF
    x_error = inekf_X[:,6] - x_interp
    y_error = inekf_X[:,7] - y_interp
    z_error = inekf_X[:,8] - z_interp
    pos_error = concatenate((x_error.reshape(-1,1), y_error.reshape(-1,1), z_error.reshape(-1,1)), axis = 1)
    rms_x = math.sqrt(mean_squared_error(x_interp, inekf_X[:,6]))
    rms_y = math.sqrt(mean_squared_error(y_interp, inekf_X[:,7]))
    rms_z = math.sqrt(mean_squared_error(z_interp, inekf_X[:,8]))
    print('InEKF: The RMS error for position x is %f [m]' % rms_x)
    print('InEKF: The RMS error for position y is %f [m]' % rms_y)
    print('InEKF: The RMS error for position z is %f [m]' % rms_z)
    RMS_all = math.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
    print('InEKF: The overall RMS error of position estimation is %f [m]\n' % RMS_all)

    vx_error = inekf_X[:,3] - vx_interp
    vy_error = inekf_X[:,4] - vy_interp
    vz_error = inekf_X[:,5] - vz_interp
    vel_error = concatenate((vx_error.reshape(-1,1), vy_error.reshape(-1,1), vz_error.reshape(-1,1)), axis = 1)
    rms_vx = math.sqrt(mean_squared_error(vx_interp, inekf_X[:,3]))
    rms_vy = math.sqrt(mean_squared_error(vy_interp, inekf_X[:,4]))
    rms_vz = math.sqrt(mean_squared_error(vz_interp, inekf_X[:,5]))
    print('InEKF: The RMS error for velocity x is %f [m/s]' % rms_vx)
    print('InEKF: The RMS error for velocity y is %f [m/s]' % rms_vy)
    print('InEKF: The RMS error for velocity z is %f [m/s]' % rms_vz)
    RMS_all = math.sqrt(rms_vx**2 + rms_vy**2 + rms_vz**2)
    print('InEKF: The overall RMS error of velocity estimation is %f [m/s]\n' % RMS_all)

    ox_error = inekf_X[:,0] - ori_x_interp
    oy_error = inekf_X[:,1] - ori_y_interp
    oz_error = inekf_X[:,2] - ori_z_interp
    ori_error = concatenate((ox_error.reshape(-1,1), oy_error.reshape(-1,1), oz_error.reshape(-1,1)), axis = 1)
    rms_ox = math.sqrt(mean_squared_error(ori_x_interp, inekf_X[:,0]))
    rms_oy = math.sqrt(mean_squared_error(ori_y_interp, inekf_X[:,1]))
    rms_oz = math.sqrt(mean_squared_error(ori_z_interp, inekf_X[:,2]))
    print('InEKF: The RMS error for orientation x is %f [rad]' % rms_ox)
    print('InEKF: The RMS error for orientation y is %f [rad]' % rms_oy)
    print('InEKF: The RMS error for orientation z is %f [rad]' % rms_oz)
    RMS_all = math.sqrt(rms_ox**2 + rms_oy**2 + rms_oz**2)
    print('InEKF: The overall RMS error of orientation estimation is %f [rad]\n' % RMS_all)

    # visualization
    plot_pos_inekf("InEKF", t, inekf_X, t_vicon, pos_vicon)
    plot_vel_inekf("InEKF", t, inekf_X, t_vicon, vel_vicon)
    plot_ori_inekf("InEKF", t, inekf_X, t_vicon, ori_vicon)
    plot_pos_err_inekf("InEKF", t, pos_error, inekf_P)
    plot_vel_err_inekf("InEKF", t, vel_error, inekf_P)
    plot_ori_err_inekf("InEKF", t, ori_error, inekf_P)
    plot_traj_inekf("InEKF", pos_vicon, inekf_X, anchor_position)

    plot_traj(
        anchor_position,
        pos_vicon,
        eskf.Xpo,
        inekf_X)

    plt.show()
