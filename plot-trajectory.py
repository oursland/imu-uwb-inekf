#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})

from scripts.utility.praser import extract_gt, extract_tdoa, extract_acc, extract_gyro, interp_meas, extract_tdoa_meas

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

FONTSIZE = 16
TICK_SIZE = 16

def plot_trajectory(filename):
    df = pd.read_csv(filename)

    t = df['t']
    speed = np.linalg.norm(df[['vx','vy','vz']].values,axis=1)
    index_of_first_motion = np.where(speed > 0.1)[0][0]
    time_of_first_motion = t[index_of_first_motion]

    subsample_ratio = 128
    t = df['t'].values[::subsample_ratio]
    x = df['x'].values[::subsample_ratio]
    y = df['y'].values[::subsample_ratio]
    z = df['z'].values[::subsample_ratio]
    vx = df['vx'].values[::subsample_ratio]
    vy = df['vy'].values[::subsample_ratio]
    vz = df['vz'].values[::subsample_ratio]
    ox = df['ox'].values[::subsample_ratio]
    oy = df['oy'].values[::subsample_ratio]
    oz = df['oz'].values[::subsample_ratio]

    eskf_x = df['eskf_x'].values[::subsample_ratio]
    eskf_y = df['eskf_y'].values[::subsample_ratio]
    eskf_z = df['eskf_z'].values[::subsample_ratio]
    eskf_vx = df['eskf_vx'].values[::subsample_ratio]
    eskf_vy = df['eskf_vy'].values[::subsample_ratio]
    eskf_vz = df['eskf_vz'].values[::subsample_ratio]
    eskf_ox = df['eskf_ox'].values[::subsample_ratio]
    eskf_oy = df['eskf_oy'].values[::subsample_ratio]
    eskf_oz = df['eskf_oz'].values[::subsample_ratio]
    eskf_x_cov = df['eskf_x_cov'].values[::subsample_ratio]
    eskf_y_cov = df['eskf_y_cov'].values[::subsample_ratio]
    eskf_z_cov = df['eskf_z_cov'].values[::subsample_ratio]
    eskf_vx_cov = df['eskf_vx_cov'].values[::subsample_ratio]
    eskf_vy_cov = df['eskf_vy_cov'].values[::subsample_ratio]
    eskf_vz_cov = df['eskf_vz_cov'].values[::subsample_ratio]
    eskf_ox_cov = df['eskf_ox_cov'].values[::subsample_ratio]
    eskf_oy_cov = df['eskf_oy_cov'].values[::subsample_ratio]
    eskf_oz_cov = df['eskf_oz_cov'].values[::subsample_ratio]

    inekf_x = df['inekf_x'].values[::subsample_ratio]
    inekf_y = df['inekf_y'].values[::subsample_ratio]
    inekf_z = df['inekf_z'].values[::subsample_ratio]
    inekf_vx = df['inekf_vx'].values[::subsample_ratio]
    inekf_vy = df['inekf_vy'].values[::subsample_ratio]
    inekf_vz = df['inekf_vz'].values[::subsample_ratio]
    inekf_ox = df['inekf_ox'].values[::subsample_ratio]
    inekf_oy = df['inekf_oy'].values[::subsample_ratio]
    inekf_oz = df['inekf_oz'].values[::subsample_ratio]
    inekf_x_cov = df['inekf_x_cov'].values[::subsample_ratio]
    inekf_y_cov = df['inekf_y_cov'].values[::subsample_ratio]
    inekf_z_cov = df['inekf_z_cov'].values[::subsample_ratio]
    inekf_vx_cov = df['inekf_vx_cov'].values[::subsample_ratio]
    inekf_vy_cov = df['inekf_vy_cov'].values[::subsample_ratio]
    inekf_vz_cov = df['inekf_vz_cov'].values[::subsample_ratio]
    inekf_ox_cov = df['inekf_ox_cov'].values[::subsample_ratio]
    inekf_oy_cov = df['inekf_oy_cov'].values[::subsample_ratio]
    inekf_oz_cov = df['inekf_oz_cov'].values[::subsample_ratio]

    fig = plt.figure(facecolor = "white",figsize=(5.4, 4.8))
    ax_t = fig.add_subplot(projection='3d')
    # make the panes transparent
    ax_t.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_t.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_t.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # change the color of the grid lines
    ax_t.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax_t.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax_t.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)

    ax_t.plot(x, y, z, color='black', linewidth=2.0, alpha=1.0, label='Ground Truth')
    ax_t.plot(eskf_x, eskf_y, eskf_z, color='tab:blue', linewidth=1.0, alpha=1.0, label = 'ESKF')
    ax_t.plot(inekf_x, inekf_y, inekf_z, color='tab:red', linewidth=1.0, alpha=1.0, label = 'InEKF')
    ax_t.set_xlim([-3.5,3.5])
    ax_t.set_ylim([-3.9,3.9])
    ax_t.set_zlim([-0.0,3.0])
    # use LaTeX fonts in the plot
    ax_t.set_xlabel(r'X [m]') #,fontsize=FONTSIZE)
    ax_t.set_ylabel(r'Y [m]') #,fontsize=FONTSIZE)
    ax_t.set_zlabel(r'Z [m]') #,fontsize=FONTSIZE)
    ax_t.legend(loc='upper left', fontsize=12)
    ax_t.view_init(24, -58)
    ax_t.set_box_aspect((1, 1, 0.5))  # xy aspect ratio is 1:1, but change z axis

    fig.suptitle('Trajectory', fontsize=FONTSIZE)

    # POSITION PLOT
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, x, color='black')
    ax1.plot(t, eskf_x, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_x, color='tab:red', alpha=0.5)
    ax1.fill_between(t, eskf_x + 3*eskf_x_cov, eskf_x - 3*eskf_x_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, inekf_x + 3*inekf_x_cov, inekf_x - 3*inekf_x_cov, color='tab:red', alpha=0.2)
    ax1.set_title('Position (X-axis)')
    ax1.set_ylabel('Position (m)')
    ax1.set_xlabel('Time (s)')
    ax1.axvline(x=time_of_first_motion, color='tab:red')

    ax2.plot(t, y, color='black')
    ax2.plot(t, eskf_y, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_y, color='tab:red', alpha=0.5)
    ax2.fill_between(t, eskf_y + 3*eskf_y_cov, eskf_y - 3*eskf_y_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, inekf_y + 3*inekf_y_cov, inekf_y - 3*inekf_y_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Position (Y-axis)')
    ax2.set_ylabel('Position (m)')
    ax2.set_xlabel('Time (s)')
    ax2.axvline(x=time_of_first_motion, color='tab:red')

    ax3.plot(t, z, color='black')
    ax3.plot(t, eskf_z, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_z, color='tab:red', alpha=0.5)
    ax3.fill_between(t, eskf_z + 3*eskf_z_cov, eskf_z - 3*eskf_z_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, inekf_z + 3*inekf_z_cov, inekf_z - 3*inekf_z_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Position (Z-axis)')
    ax3.set_ylabel('Position (m)')
    ax3.set_xlabel('Time (s)')
    ax3.axvline(x=time_of_first_motion, color='tab:red')

    hK, = plt.plot([0,0], color='black', linestyle='-')
    hB, = plt.plot([0,0], color='tab:blue', linestyle='-')
    hR, = plt.plot([0,0], color='tab:red', linestyle='-')
    fig.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Position', fontsize=FONTSIZE)

    # POSITION ERROR PLOT
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, eskf_x - x, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_x - x, color='tab:red', alpha=0.5)
    ax1.fill_between(t, -3*eskf_x_cov,         3*eskf_x_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, -3*inekf_x_cov,        3*inekf_x_cov, color='tab:red', alpha=0.2)
    ax1.set_title('X-axis')
    ax1.set_ylabel('Position Error (m)')
    ax1.set_xlabel('Time (s)')
    ax1.axvline(x=time_of_first_motion, color='tab:red')
    ax1.axhline(y=0, color='black')

    ax2.plot(t, eskf_y - y, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_y - y, color='tab:red', alpha=0.5)
    ax2.fill_between(t, -3*eskf_y_cov,         3*eskf_y_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, -3*inekf_y_cov,        3*inekf_y_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Y-axis')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_xlabel('Time (s)')
    ax2.axvline(x=time_of_first_motion, color='tab:red')
    ax2.axhline(y=0, color='black')

    ax3.plot(t, eskf_z - z, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_z - z, color='tab:red', alpha=0.5)
    ax3.fill_between(t, -3*eskf_z_cov,         3*eskf_z_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, -3*inekf_z_cov,        3*inekf_z_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Z-axis')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_xlabel('Time (s)')
    ax3.axvline(x=time_of_first_motion, color='tab:red')
    ax3.axhline(y=0, color='black')

    hK, = plt.plot([0,0], color='black', linestyle='-')
    hB, = plt.plot([0,0], color='tab:blue', linestyle='-')
    hR, = plt.plot([0,0], color='tab:red', linestyle='-')
    fig.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Position Error', fontsize=FONTSIZE)

    # VELOCITY PLOT
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, vx, color='black')
    ax1.plot(t, eskf_vx, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_vx, color='tab:red', alpha=0.5)
    ax1.fill_between(t, eskf_vx + 3*eskf_vx_cov, eskf_vx - 3*eskf_vx_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, inekf_vx + 3*inekf_vx_cov, inekf_vx - 3*inekf_vx_cov, color='tab:red', alpha=0.2)
    ax1.set_title('Velocity (X-axis)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_xlabel('Time (s)')
    ax1.axvline(x=time_of_first_motion, color='tab:red')

    ax2.plot(t, vy, color='black')
    ax2.plot(t, eskf_vy, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_vy, color='tab:red', alpha=0.5)
    ax2.fill_between(t, eskf_vy + 3*eskf_vy_cov, eskf_vy - 3*eskf_vy_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, inekf_vy + 3*inekf_vy_cov, inekf_vy - 3*inekf_vy_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Velocity (Y-axis)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.axvline(x=time_of_first_motion, color='tab:red')

    ax3.plot(t, vz, color='black')
    ax3.plot(t, eskf_vz, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_vz, color='tab:red', alpha=0.5)
    ax3.fill_between(t, eskf_vz + 3*eskf_vz_cov, eskf_vz - 3*eskf_vz_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, inekf_vz + 3*inekf_vz_cov, inekf_vz - 3*inekf_vz_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Velocity (Z-axis)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_xlabel('Time (s)')
    ax3.axvline(x=time_of_first_motion, color='tab:red')

    hK, = plt.plot([0,0], color='black', linestyle='-')
    hB, = plt.plot([0,0], color='tab:blue', linestyle='-')
    hR, = plt.plot([0,0], color='tab:red', linestyle='-')
    fig.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Velocity', fontsize=FONTSIZE)

    # VELOCITY ERROR PLOT
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, eskf_vx - vx, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_vx - vx, color='tab:red', alpha=0.5)
    ax1.fill_between(t, -3*eskf_vx_cov,         3*eskf_vx_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, -3*inekf_vx_cov,        3*inekf_vx_cov, color='tab:red', alpha=0.2)
    ax1.set_title('X-axis')
    ax1.set_ylabel('Velocity Error (m)')
    ax1.set_xlabel('Time (s)')
    ax1.axvline(x=time_of_first_motion, color='tab:red')
    ax1.axhline(y=0, color='black')

    ax2.plot(t, eskf_vy - vy, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_vy - vy, color='tab:red', alpha=0.5)
    ax2.fill_between(t, -3*eskf_vy_cov,         3*eskf_vy_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, -3*inekf_vy_cov,        3*inekf_vy_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Y-axis')
    ax2.set_ylabel('Velocity Error (m)')
    ax2.set_xlabel('Time (s)')
    ax2.axvline(x=time_of_first_motion, color='tab:red')
    ax2.axhline(y=0, color='black')

    ax3.plot(t, eskf_vz - vz, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_vz - vz, color='tab:red', alpha=0.5)
    ax3.fill_between(t, -3*eskf_vz_cov,         3*eskf_vz_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, -3*inekf_vz_cov,        3*inekf_vz_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Z-axis')
    ax3.set_ylabel('Velocity Error (m)')
    ax3.set_xlabel('Time (s)')
    ax3.axvline(x=time_of_first_motion, color='tab:red')
    ax3.axhline(y=0, color='black')

    hK, = plt.plot([0,0], color='black', linestyle='-')
    hB, = plt.plot([0,0], color='tab:blue', linestyle='-')
    hR, = plt.plot([0,0], color='tab:red', linestyle='-')
    fig.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Velocity Error', fontsize=FONTSIZE)

    # ORIENTATION PLOT
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, ox, color='black')
    ax1.plot(t, eskf_ox, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_ox, color='tab:red', alpha=0.5)
    ax1.fill_between(t, eskf_ox + 3*eskf_ox_cov, eskf_ox - 3*eskf_ox_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, inekf_ox + 3*inekf_ox_cov, inekf_ox - 3*inekf_ox_cov, color='tab:red', alpha=0.2)
    ax1.set_title('Orientation (X-axis)')
    ax1.set_ylabel('Angle (rad)')
    ax1.set_xlabel('Time (s)')
    ax1.axvline(x=time_of_first_motion, color='tab:red')

    ax2.plot(t, oy, color='black')
    ax2.plot(t, eskf_oy, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_oy, color='tab:red', alpha=0.5)
    ax2.fill_between(t, eskf_oy + 3*eskf_oy_cov, eskf_oy - 3*eskf_oy_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, inekf_oy + 3*inekf_oy_cov, inekf_oy - 3*inekf_oy_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Orientation (Y-axis)')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_xlabel('Time (s)')
    ax2.axvline(x=time_of_first_motion, color='tab:red')

    ax3.plot(t, oz, color='black')
    ax3.plot(t, eskf_oz, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_oz, color='tab:red', alpha=0.5)
    ax3.fill_between(t, eskf_oz + 3*eskf_oz_cov, eskf_oz - 3*eskf_oz_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, inekf_oz + 3*inekf_oz_cov, inekf_oz - 3*inekf_oz_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Orientation (Z-axis)')
    ax3.set_ylabel('Angle (rad)')
    ax3.set_xlabel('Time (s)')
    ax3.axvline(x=time_of_first_motion, color='tab:red')

    hK, = plt.plot([0,0], color='black', linestyle='-')
    hB, = plt.plot([0,0], color='tab:blue', linestyle='-')
    hR, = plt.plot([0,0], color='tab:red', linestyle='-')
    fig.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Orientation', fontsize=FONTSIZE)

    # ORIENTATION ERROR PLOT
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, eskf_ox - ox, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_ox - ox, color='tab:red', alpha=0.5)
    ax1.fill_between(t, -3*eskf_ox_cov,         3*eskf_ox_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, -3*inekf_ox_cov,        3*inekf_ox_cov, color='tab:red', alpha=0.2)
    ax1.set_title('X-axis')
    ax1.set_ylabel('Orientation Error (m)')
    ax1.set_xlabel('Time (s)')
    ax1.axvline(x=time_of_first_motion, color='tab:red')
    ax1.axhline(y=0, color='black')

    ax2.plot(t, eskf_oy - oy, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_oy - oy, color='tab:red', alpha=0.5)
    ax2.fill_between(t, -3*eskf_oy_cov,         3*eskf_oy_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, -3*inekf_oy_cov,        3*inekf_oy_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Y-axis')
    ax2.set_ylabel('Orientation Error (m)')
    ax2.set_xlabel('Time (s)')
    ax2.axvline(x=time_of_first_motion, color='tab:red')
    ax2.axhline(y=0, color='black')

    ax3.plot(t, eskf_oz - oz, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_oz - oz, color='tab:red', alpha=0.5)
    ax3.fill_between(t, -3*eskf_oz_cov,         3*eskf_oz_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, -3*inekf_oz_cov,        3*inekf_oz_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Z-axis')
    ax3.set_ylabel('Orientation Error (m)')
    ax3.set_xlabel('Time (s)')
    ax3.axvline(x=time_of_first_motion, color='tab:red')
    ax3.axhline(y=0, color='black')

    hK, = plt.plot([0,0], color='black', linestyle='-')
    hB, = plt.plot([0,0], color='tab:blue', linestyle='-')
    hR, = plt.plot([0,0], color='tab:red', linestyle='-')
    fig.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Orientation Error', fontsize=FONTSIZE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    # set window background to white
    plt.rcParams['figure.facecolor'] = 'w'

    mpl.rc('xtick', labelsize=TICK_SIZE)
    mpl.rc('ytick', labelsize=TICK_SIZE)

    plot_trajectory(args.filename)

    plt.show()
