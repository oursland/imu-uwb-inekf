#!/usr/bin/env python3

import os

from tqdm import tqdm

import numpy as np
import pandas as pd

from pyquaternion import Quaternion
from scipy import interpolate

from scripts.utility.praser import extract_gt, extract_tdoa, extract_acc, extract_gyro, interp_meas, extract_tdoa_meas

from scripts.estimation.eskf_class import ESKF
from scripts.estimation.inekf import InEKF, SO3_exp, SO3_log, SE3_2_exp

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

def isin(t_np,t_k):
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
    mat = np.zeros((3, 3))

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

datasets = [
    # ("dataset/flight-dataset/csv-data/const3/const3-trial7-tdoa2-manual1.csv", "dataset/flight-dataset/survey-results/anchor_const3.npz"),
    # ("dataset/flight-dataset/csv-data/const3/const3-trial7-tdoa2-manual2.csv", "dataset/flight-dataset/survey-results/anchor_const3.npz"),
    # ("dataset/flight-dataset/csv-data/const3/const3-trial7-tdoa3-manual3.csv", "dataset/flight-dataset/survey-results/anchor_const3.npz"),
    # ("dataset/flight-dataset/csv-data/const3/const3-trial7-tdoa3-manual4.csv", "dataset/flight-dataset/survey-results/anchor_const3.npz"),
    ("dataset/flight-dataset/csv-data/const4/const4-trial7-tdoa2-manual1.csv", "dataset/flight-dataset/survey-results/anchor_const4.npz"),
    ("dataset/flight-dataset/csv-data/const4/const4-trial7-tdoa2-manual2.csv", "dataset/flight-dataset/survey-results/anchor_const4.npz"),
    ("dataset/flight-dataset/csv-data/const4/const4-trial7-tdoa2-manual3.csv", "dataset/flight-dataset/survey-results/anchor_const4.npz"),
]

def read_dataset(csv_file, anchor_npz):
    df = pd.read_csv(csv_file)
    anchor_survey = np.load(anchor_npz)
    # select anchor constellations
    anchor_position = anchor_survey['an_pos']

    # --------------- extract csv file --------------- #
    gt_pose = extract_gt(df)
    tdoa    = extract_tdoa(df)
    acc     = extract_acc(df)
    gyr     = extract_gyro(df)

    t_vicon = gt_pose[:,0]
    pos_vicon = gt_pose[:,1:4]
    ori_vicon = gt_pose[:,4:8]
    # t_tdoa = tdoa[:,0]
    # uwb_tdoa  = tdoa[:,1:]

    t_imu = acc[:,0]
    gyr_x_syn = interp_meas(gyr[:,0], gyr[:,1], t_imu).reshape(-1,1)
    gyr_y_syn = interp_meas(gyr[:,0], gyr[:,2], t_imu).reshape(-1,1)
    gyr_z_syn = interp_meas(gyr[:,0], gyr[:,3], t_imu).reshape(-1,1)

    imu = np.concatenate((acc[:,1:], gyr_x_syn, gyr_y_syn, gyr_z_syn), axis = 1)

    min_t = min(tdoa[0,0], t_imu[0], t_vicon[0])
    # get the vicon information from min_t
    t_vicon = np.array(t_vicon)
    idx = np.argwhere(t_vicon > min_t)
    t_vicon = np.squeeze(t_vicon[idx])
    pos_vicon = np.squeeze(np.array(pos_vicon)[idx,:])

    # reset time base
    t_vicon = (t_vicon - min_t)
    t_imu = (t_imu - min_t).reshape(-1,1)
    tdoa[:,0] = tdoa[:,0] - min_t

    return t_vicon, pos_vicon, ori_vicon, t_imu, imu, tdoa, anchor_position

if __name__ == "__main__":
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
    print("    std_tdoa = ", std_tdoa)
    print()

    for dataset in datasets:
        csv_file = dataset[0]
        # access the survey results
        anchor_npz = dataset[1]

        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]

        t_vicon, pos_vicon, ori_vicon, t_imu, imu, tdoa, anchor_position = read_dataset(csv_file, anchor_npz)

        # interpolate Vicon measurements
        f_x = interpolate.splrep(t_vicon, pos_vicon[:,0], s = 0.5)
        f_y = interpolate.splrep(t_vicon, pos_vicon[:,1], s = 0.5)
        f_z = interpolate.splrep(t_vicon, pos_vicon[:,2], s = 0.5)

        vel_vicon = np.zeros((len(t_vicon), 3))
        vel_vicon[:,0] = interpolate.splev(t_vicon, f_x, der = 1)
        vel_vicon[:,1] = interpolate.splev(t_vicon, f_y, der = 1)
        vel_vicon[:,2] = interpolate.splev(t_vicon, f_z, der = 1)

        # Error found in: [ const3-trial2-tdoa2.csv, const4-trial3-tdoa2-traj3.csv ]
        if t_vicon.shape[0] != ori_vicon.shape[0]:
            print('Extra Vicon orientation detected!!!')
            ori_vicon = ori_vicon[:t_vicon.shape[0]]

        ori_v = []
        for q in ori_vicon:
            ori_v.append(SO3_log(from_quaternion(q, 'xyzw')))
        ori_v = np.array(ori_v)

        f_ori_x = interpolate.splrep(t_vicon, ori_v[:,0], s = 0.5)
        f_ori_y = interpolate.splrep(t_vicon, ori_v[:,1], s = 0.5)
        f_ori_z = interpolate.splrep(t_vicon, ori_v[:,2], s = 0.5)

        # extract tdoa meas.
        tdoa_70, tdoa_01, tdoa_12, tdoa_23, tdoa_34, tdoa_45, tdoa_56, tdoa_67 = extract_tdoa_meas(tdoa[:,0], tdoa[:,1:4])

        # convert back to tdoa
        tdoa_c = np.concatenate((
            tdoa_70, tdoa_01, tdoa_12, tdoa_23,
            tdoa_34, tdoa_45, tdoa_56, tdoa_67,
            ), axis = 0)

        sort_id=np.argsort(tdoa_c[:,0])
        t_uwb = tdoa_c[sort_id, 0].reshape(-1,1)
        uwb   = tdoa_c[sort_id, 1:4]

        # Create a compound vector t with a sorted merge of all the sensor time bases
        time = np.sort(np.concatenate((t_imu, t_uwb)))
        t = np.unique(time)
        K = t.shape[0]

        for yaw_delta in range(16, -16, -1):
            # ----------------------- INITIALIZATION OF EKF -------------------------#
            # Initial estimate for the state vector
            pos0 = pos_vicon[0]
            vel0 = np.array([ 0.0, 0.0, 0.0 ])
            ori0 = SO3_log(from_quaternion(ori_vicon[0], 'xyzw'))
            ori0q = Quaternion(np.block([ori_vicon[0][3], ori_vicon[0][0:3]]))

            # create the object of ESKF
            eskf_X = np.zeros((6))
            eskf_X[0:3] = pos0
            eskf_X[3:6] = vel0
            eskf_q = ori0q*Quaternion(axis=(0.0, 0.0, 1.0), radians=yaw_delta*np.pi/16)
            eskf_P = np.diag([
                std_xy0**2,   std_xy0**2,   std_z0**2,
                std_vel0**2,  std_vel0**2,  std_vel0**2,
                std_rp0**2,   std_rp0**2,   std_yaw0**2,
            ])
            eskf = ESKF(eskf_X, eskf_q, eskf_P, K)

            # x = np.array([ Rx, Ry, Rz, Vx, Vy, Vz, Px, Py, Pz ])
            inekf_X = SE3_2_exp(np.zeros(9))
            inekf_X[:3,:3] = SO3_exp(ori0 + [ 0, 0, yaw_delta*np.pi/16 ])
            inekf_X[:3, 3] = vel0
            inekf_X[:3, 4] = pos0
            inekf_P = np.diag([
                std_rp0**2,  std_rp0**2,  std_yaw0**2,
                std_vel0**2, std_vel0**2, std_vel0**2,
                std_xy0**2,  std_xy0**2,  std_z0**2,
                std_b_acc**2, std_b_acc**2, std_b_acc**2,
                std_b_gyr**2, std_b_gyr**2, std_b_gyr**2,
            ])
            inekf_bias = np.zeros(6)
            inekf = InEKF(inekf_X, inekf_P, inekf_bias)

            eskf.std_uwb_tdoa = std_tdoa
            inekf.w_tdoa = np.array([std_tdoa])

            pos = np.zeros((len(t), 3))
            pos[:,0] = interpolate.splev(t, f_x, der = 0)
            pos[:,1] = interpolate.splev(t, f_y, der = 0)
            pos[:,2] = interpolate.splev(t, f_z, der = 0)

            vel = np.zeros((len(t), 3))
            vel[:,0] = interpolate.splev(t, f_x, der = 1)
            vel[:,1] = interpolate.splev(t, f_y, der = 1)
            vel[:,2] = interpolate.splev(t, f_z, der = 1)

            ori = np.zeros((len(t), 3))
            ori[:,0] = interpolate.splev(t, f_ori_x, der = 0)
            ori[:,1] = interpolate.splev(t, f_ori_y, der = 0)
            ori[:,2] = interpolate.splev(t, f_ori_z, der = 0)

            print(dataset_name + " Yaw Delta: " + str(yaw_delta) + "/16 𝜋")

            rows = []

            # log initial conditions
            eskf_ori = SO3_log(from_quaternion(eskf_q, 'wxyz'))
            inekf_ori = SO3_log(inekf_X[:3,:3])
            row = {
                't':  0.0,

                'x':  pos[0, 0],
                'y':  pos[0, 1],
                'z':  pos[0, 2],
                'vx': vel[0, 0],
                'vy': vel[0, 1],
                'vz': vel[0, 2],
                'ox': ori[0, 0],
                'oy': ori[0, 1],
                'oz': ori[0, 2],

                'eskf_x':  eskf_X[0],
                'eskf_y':  eskf_X[1],
                'eskf_z':  eskf_X[2],
                'eskf_vx': eskf_X[3],
                'eskf_vy': eskf_X[4],
                'eskf_vz': eskf_X[5],
                'eskf_ox': eskf_ori[0],
                'eskf_oy': eskf_ori[1],
                'eskf_oz': eskf_ori[2],

                'eskf_x_cov':  eskf_P[0,0],
                'eskf_y_cov':  eskf_P[1,1],
                'eskf_z_cov':  eskf_P[2,2],
                'eskf_vx_cov': eskf_P[3,3],
                'eskf_vy_cov': eskf_P[4,4],
                'eskf_vz_cov': eskf_P[5,5],
                'eskf_ox_cov': eskf_P[6,6],
                'eskf_oy_cov': eskf_P[7,7],
                'eskf_oz_cov': eskf_P[8,8],

                'eskf_rej': 0,

                'inekf_x':  inekf_X[0,4],
                'inekf_y':  inekf_X[1,4],
                'inekf_z':  inekf_X[2,4],
                'inekf_vx': inekf_X[0,3],
                'inekf_vy': inekf_X[1,3],
                'inekf_vz': inekf_X[2,3],
                'inekf_ox': inekf_ori[0],
                'inekf_oy': inekf_ori[1],
                'inekf_oz': inekf_ori[2],

                'inekf_x_cov':  inekf_P[6,6],
                'inekf_y_cov':  inekf_P[7,7],
                'inekf_z_cov':  inekf_P[8,8],
                'inekf_vx_cov': inekf_P[3,3],
                'inekf_vy_cov': inekf_P[4,4],
                'inekf_vz_cov': inekf_P[5,5],
                'inekf_ox_cov': inekf_P[0,0],
                'inekf_oy_cov': inekf_P[1,1],
                'inekf_oz_cov': inekf_P[2,2],

                'inekf_rej': 0,
            }
            rows.append(row)

            # InEKF's dt is based on IMU measurements, so track previous IMU timestamp
            t_imu_prev = 0

            for k in tqdm(range(1, K)):
                # Find what measurements are available at the current time (help function: isin() )
                imu_k, imu_check = isin(t_imu, t[k-1])
                uwb_k, uwb_check = isin(t_uwb, t[k-1])
                dt = t[k]-t[k-1]

                imu_dt = t[k] - t_imu_prev
                if imu_check:
                    t_imu_prev = t[k]

                inekf_rej = 0
                eskf_rej = 0

                # ESKF Prediction
                eskf_X, eskf_q, eskf_P = eskf.predict(imu[imu_k,:], dt, imu_check, k)

                # InEKF Prediction
                if imu_check:
                    inekf_X, inekf_P, inekf_bias = inekf.predict(imu[imu_k,:], imu_dt)

                if uwb_check:
                    eskf_X, eskf_q, eskf_P, eskf_rej = eskf.correct(uwb[uwb_k,:], anchor_position, k)
                    inekf_X, inekf_P, inekf_bias, inekf_rej = inekf.correct(uwb[uwb_k,:], anchor_position)

                ## add to dataframe
                eskf_ori = SO3_log(from_quaternion(eskf_q, 'wxyz'))
                inekf_ori = SO3_log(inekf_X[:3,:3])

                row = {
                    't':  t[k],

                    'x':  pos[k, 0],
                    'y':  pos[k, 1],
                    'z':  pos[k, 2],
                    'vx': vel[k, 0],
                    'vy': vel[k, 1],
                    'vz': vel[k, 2],
                    'ox': ori[k, 0],
                    'oy': ori[k, 1],
                    'oz': ori[k, 2],

                    'eskf_x':  eskf_X[0],
                    'eskf_y':  eskf_X[1],
                    'eskf_z':  eskf_X[2],
                    'eskf_vx': eskf_X[3],
                    'eskf_vy': eskf_X[4],
                    'eskf_vz': eskf_X[5],
                    'eskf_ox': eskf_ori[0],
                    'eskf_oy': eskf_ori[1],
                    'eskf_oz': eskf_ori[2],

                    'eskf_x_cov':  eskf_P[0,0],
                    'eskf_y_cov':  eskf_P[1,1],
                    'eskf_z_cov':  eskf_P[2,2],
                    'eskf_vx_cov': eskf_P[3,3],
                    'eskf_vy_cov': eskf_P[4,4],
                    'eskf_vz_cov': eskf_P[5,5],
                    'eskf_ox_cov': eskf_P[6,6],
                    'eskf_oy_cov': eskf_P[7,7],
                    'eskf_oz_cov': eskf_P[8,8],

                    'eskf_rej': eskf_rej,

                    'inekf_x':  inekf_X[0,4],
                    'inekf_y':  inekf_X[1,4],
                    'inekf_z':  inekf_X[2,4],
                    'inekf_vx': inekf_X[0,3],
                    'inekf_vy': inekf_X[1,3],
                    'inekf_vz': inekf_X[2,3],
                    'inekf_ox': inekf_ori[0],
                    'inekf_oy': inekf_ori[1],
                    'inekf_oz': inekf_ori[2],

                    'inekf_x_cov':  inekf_P[6,6],
                    'inekf_y_cov':  inekf_P[7,7],
                    'inekf_z_cov':  inekf_P[8,8],
                    'inekf_vx_cov': inekf_P[3,3],
                    'inekf_vy_cov': inekf_P[4,4],
                    'inekf_vz_cov': inekf_P[5,5],
                    'inekf_ox_cov': inekf_P[0,0],
                    'inekf_oy_cov': inekf_P[1,1],
                    'inekf_oz_cov': inekf_P[2,2],

                    'inekf_rej': inekf_rej,
                }
                rows.append(row)

            if yaw_delta >= 0:
                sign = "+"
            else:
                sign = ""

            results_dir = "results" + "-tdoa" + str(var_tdoa) + "-std_pos" + str(std_xy0) + "-std_vel" + str(std_vel0) + "-std_yaw" + str(std_yaw0) + "/initial-yaw/"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            output_name = results_dir + dataset_name + "-yaw" + sign + str(yaw_delta) + ".csv.zst"

            df = pd.DataFrame(rows)
            df.to_csv(output_name)
