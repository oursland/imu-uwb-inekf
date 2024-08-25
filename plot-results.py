#!/usr/bin/env python3

import glob
import re

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

FONTSIZE = 12
TICK_SIZE = 12

def read_summary(directory):
    df = pd.read_csv(directory + "summary.csv")
    return df

def set_box_2colors(bp):
    plt.setp(bp['boxes'][0], color='tab:blue')
    plt.setp(bp['caps'][0], color='tab:blue')
    plt.setp(bp['caps'][1], color='tab:blue')
    plt.setp(bp['whiskers'][0], color='tab:blue')
    plt.setp(bp['whiskers'][1], color='tab:blue')
    # plt.setp(bp['fliers'][0], color='tab:blue')
    # plt.setp(bp['fliers'][1], color='tab:blue')
    plt.setp(bp['medians'][0], color='tab:blue')

    plt.setp(bp['boxes'][1], color='tab:red')
    plt.setp(bp['caps'][2], color='tab:red')
    plt.setp(bp['caps'][3], color='tab:red')
    plt.setp(bp['whiskers'][2], color='tab:red')
    plt.setp(bp['whiskers'][3], color='tab:red')
    # plt.setp(bp['fliers'][2], color='tab:red')
    # plt.setp(bp['fliers'][3], color='tab:red')
    plt.setp(bp['medians'][1], color='tab:red')

def error_boxplot():
    df = read_summary("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/main/")

    const_pos_list = []
    const_vel_list = []
    const_ori_list = []

    for const in range(1, 5):
        const_pos_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_pos'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_pos'] ])
        const_vel_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_vel'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_vel'] ])
        const_ori_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_ori'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_ori'] ])

    ticks = [ '1', '2', '3', '4' ]
    tick_pos = [ 1.5, 3.5, 5.5, 7.5 ]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(5.4, 4.8))

    bp = ax1.boxplot(const_pos_list[0], positions = [ 1, 2 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax1.boxplot(const_pos_list[1], positions = [ 3, 4 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax1.boxplot(const_pos_list[2], positions = [ 5, 6 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax1.boxplot(const_pos_list[3], positions = [ 7, 8 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    ax1.set_xticks(tick_pos, ticks)
    ax1.axvline(x=2.5, color='k', linestyle='dashed')
    ax1.axvline(x=4.5, color='k', linestyle='dashed')
    ax1.axvline(x=6.5, color='k', linestyle='dashed')
    ax1.set_title('Position Error', fontsize=FONTSIZE)
    ax1.set_ylabel('RMS Error (m)', fontsize=FONTSIZE)

    bp = ax2.boxplot(const_vel_list[0], positions = [ 1, 2 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax2.boxplot(const_vel_list[1], positions = [ 3, 4 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax2.boxplot(const_vel_list[2], positions = [ 5, 6 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax2.boxplot(const_vel_list[3], positions = [ 7, 8 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    ax2.set_xticks(tick_pos, ticks)
    ax2.axvline(x=2.5, color='k', linestyle='dashed')
    ax2.axvline(x=4.5, color='k', linestyle='dashed')
    ax2.axvline(x=6.5, color='k', linestyle='dashed')
    ax2.set_title('Velocity Error', fontsize=FONTSIZE)
    ax2.set_ylabel('RMS Error (m/s)', fontsize=FONTSIZE)

    bp = ax3.boxplot(const_ori_list[0], positions = [ 1, 2 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax3.boxplot(const_ori_list[1], positions = [ 3, 4 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax3.boxplot(const_ori_list[2], positions = [ 5, 6 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    bp = ax3.boxplot(const_ori_list[3], positions = [ 7, 8 ], showfliers=False, widths = 0.6, showmeans=False)
    set_box_2colors(bp)
    ax3.set_xticks(tick_pos, ticks)
    ax3.axvline(x=2.5, color='k', linestyle='dashed')
    ax3.axvline(x=4.5, color='k', linestyle='dashed')
    ax3.axvline(x=6.5, color='k', linestyle='dashed')
    ax3.set_title('Orientation Error', fontsize=FONTSIZE)
    ax3.set_ylabel('RMS Error (rad)', fontsize=FONTSIZE)
    deg = ax3.twinx()
    deg.set_ylim(ax3.get_ylim()[0]*(180/np.pi), ax3.get_ylim()[1]*(180/np.pi))
    deg.set_ylabel('RMS Error (degrees)', fontsize=FONTSIZE)

    hB, = plt.plot([0,0],'b-')
    hR, = plt.plot([0,0],'r-')
    fig.legend((hB, hR), ('ESKF', 'InEKF'), loc='lower left', fontsize=FONTSIZE*0.5)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('RMS Error by Constellation', fontsize=FONTSIZE*1.5)

    plt.tight_layout()
    plt.savefig('paper/figures/error-boxplot.pdf')

def orientation_sensitivity_boxplot():
    df = read_summary("results-tdoa0.13-std_pos0.1-std_vel0.01-std_yaw1.0/initial-yaw/")

    const_pos_list = []
    const_vel_list = []
    const_ori_list = []

    for const in range(4, 5):
        const_pos_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_pos'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_pos'] ])
        const_vel_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_vel'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_vel'] ])
        const_ori_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_ori'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_ori'] ])

    # ticks = [ '3', '4' ]
    ticks = [ '4' ]
    # tick_pos = [ 1.5, 3.5 ]
    tick_pos = [ 1.5 ]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(4, 4))

    bp1 = ax1.boxplot(const_pos_list[0], positions = [1, 2], widths = 0.6, showfliers=False)
    set_box_2colors(bp1)
    # bp2 = ax1.boxplot(const_pos_list[1], positions = [3, 4], widths = 0.6, showfliers=False)
    # set_box_2colors(bp2)
    ax1.set_xticks(tick_pos, ticks)
    ax1.set_title('Position Error', fontsize=FONTSIZE)
    ax1.set_ylabel('RMS Error (m)', fontsize=FONTSIZE)
    # ax1.axvline(x=2.5, color='k', linestyle='dashed')

    bp1 = ax2.boxplot(const_vel_list[0], positions = [1, 2], widths = 0.6, showfliers=False)
    set_box_2colors(bp1)
    # bp2 = ax2.boxplot(const_vel_list[1], positions = [3, 4], widths = 0.6, showfliers=False)
    # set_box_2colors(bp2)
    ax2.set_xticks(tick_pos, ticks)
    ax2.set_title('Velocity Error', fontsize=FONTSIZE)
    ax2.set_ylabel('RMS Error (m/s)', fontsize=FONTSIZE)
    # ax2.axvline(x=2.5, color='k', linestyle='dashed')

    bp1 = ax3.boxplot(const_ori_list[0], positions = [1, 2], widths = 0.6, showfliers=False)
    set_box_2colors(bp1)
    # bp2 = ax3.boxplot(const_ori_list[1], positions = [3, 4], widths = 0.6, showfliers=False)
    # set_box_2colors(bp2)
    ax3.set_xticks(tick_pos, ticks)
    ax3.set_title('Orientation Error', fontsize=FONTSIZE)
    ax3.set_ylabel('RMS Error (rad)', fontsize=FONTSIZE)
    # ax3.axvline(x=2.5, color='k', linestyle='dashed')
    deg = ax3.twinx()
    deg.set_ylim(ax3.get_ylim()[0]*(180/np.pi), ax3.get_ylim()[1]*(180/np.pi))
    deg.set_ylabel('RMS Error (degrees)', fontsize=FONTSIZE)

    hB, = plt.plot([0,0],'b-')
    hR, = plt.plot([0,0],'r-')
    fig.legend((hB, hR),('ESKF', 'InEKF'), loc='lower left', fontsize=FONTSIZE*0.5)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Initial Yaw Error RMS Error', fontsize=FONTSIZE*1.5)

    print("Initial Yaw Error RMS Error mean: ", np.mean(const_ori_list[0], axis=1))

    plt.tight_layout()
    plt.savefig('paper/figures/orientation-sensitivity-boxplot.pdf')

def position_sensitivity_boxplot():
    df = read_summary("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/initial-position/")

    const_pos_list = []
    const_vel_list = []
    const_ori_list = []

    for const in range(1, 5):
        const_pos_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_pos'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_pos'] ])
        const_vel_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_vel'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_vel'] ])
        const_ori_list.append([ df[(df['const']==const) & (df['filter']=='eskf')]['rms_ori'], df[(df['const']==const) & (df['filter']=='inekf')]['rms_ori'] ])

    ticks = [ '1', '2', '3', '4']
    tick_pos = [ 1.5, 3.5, 5.5, 7.5 ]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(5.4, 4.8))

    bp1 = ax1.boxplot(const_pos_list[0], positions = [1, 2], widths = 0.6, showfliers=False)
    set_box_2colors(bp1)
    bp2 = ax1.boxplot(const_pos_list[1], positions = [3, 4], widths = 0.6, showfliers=False)
    set_box_2colors(bp2)
    bp3 = ax1.boxplot(const_pos_list[2], positions = [5, 6], widths = 0.6, showfliers=False)
    set_box_2colors(bp3)
    bp4 = ax1.boxplot(const_pos_list[0], positions = [7, 8], widths = 0.6, showfliers=False)
    set_box_2colors(bp4)
    ax1.set_xticks(tick_pos, ticks)
    ax1.axvline(x=2.5, color='k', linestyle='dashed')
    ax1.axvline(x=4.5, color='k', linestyle='dashed')
    ax1.axvline(x=6.5, color='k', linestyle='dashed')
    ax1.set_title('Position Error', fontsize=FONTSIZE)
    ax1.set_ylabel('RMS Error (m)', fontsize=FONTSIZE)

    bp1 = ax2.boxplot(const_vel_list[0], positions = [1, 2], widths = 0.6, showfliers=False)
    set_box_2colors(bp1)
    bp2 = ax2.boxplot(const_vel_list[1], positions = [3, 4], widths = 0.6, showfliers=False)
    set_box_2colors(bp2)
    bp3 = ax2.boxplot(const_vel_list[2], positions = [5, 6], widths = 0.6, showfliers=False)
    set_box_2colors(bp3)
    bp4 = ax2.boxplot(const_vel_list[0], positions = [7, 8], widths = 0.6, showfliers=False)
    set_box_2colors(bp4)
    ax2.set_xticks(tick_pos, ticks)
    ax2.axvline(x=2.5, color='k', linestyle='dashed')
    ax2.axvline(x=4.5, color='k', linestyle='dashed')
    ax2.axvline(x=6.5, color='k', linestyle='dashed')
    ax2.set_title('Velocity Error', fontsize=FONTSIZE)
    ax2.set_ylabel('RMS Error (m/s)', fontsize=FONTSIZE)

    bp1 = ax3.boxplot(const_ori_list[0], positions = [1, 2], widths = 0.6, showfliers=False)
    set_box_2colors(bp1)
    bp2 = ax3.boxplot(const_ori_list[1], positions = [3, 4], widths = 0.6, showfliers=False)
    set_box_2colors(bp2)
    bp3 = ax3.boxplot(const_ori_list[2], positions = [5, 6], widths = 0.6, showfliers=False)
    set_box_2colors(bp3)
    bp4 = ax3.boxplot(const_ori_list[0], positions = [7, 8], widths = 0.6, showfliers=False)
    set_box_2colors(bp4)
    ax3.set_xticks(tick_pos, ticks)
    ax3.axvline(x=2.5, color='k', linestyle='dashed')
    ax3.axvline(x=4.5, color='k', linestyle='dashed')
    ax3.axvline(x=6.5, color='k', linestyle='dashed')
    ax3.set_title('Orientation Error', fontsize=FONTSIZE)
    ax3.set_ylabel('RMS Error (rad)', fontsize=FONTSIZE)
    deg = ax3.twinx()
    deg.set_ylim(ax3.get_ylim()[0]*(180/np.pi), ax3.get_ylim()[1]*(180/np.pi))
    deg.set_ylabel('RMS Error (degrees)', fontsize=FONTSIZE)

    hB, = plt.plot([0,0],'b-')
    hR, = plt.plot([0,0],'r-')
    fig.legend((hB, hR),('ESKF', 'InEKF'), loc='lower left', fontsize=FONTSIZE*0.5)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Initial Position Error RMS Error', fontsize=FONTSIZE*1.5)

    plt.tight_layout()
    plt.savefig('paper/figures/position-sensitivity-boxplot.pdf')

def exemplar_pos_plot():
    df = pd.read_csv("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/main/const4-trial7-tdoa2-manual1.csv.zst")
    # df = pd.read_csv("results-tdoa0.13-std_pos0.1-std_vel0.01-std_yaw1.0/main/const4-trial7-tdoa2-manual1.csv.zst")

    subsample_ratio = 128
    t = df['t'].values[::subsample_ratio]
    x = df['x'].values[::subsample_ratio]
    y = df['y'].values[::subsample_ratio]
    z = df['z'].values[::subsample_ratio]
    eskf_x = df['eskf_x'].values[::subsample_ratio]
    eskf_y = df['eskf_y'].values[::subsample_ratio]
    eskf_z = df['eskf_z'].values[::subsample_ratio]
    eskf_x_cov = df['eskf_x_cov'].values[::subsample_ratio]
    eskf_y_cov = df['eskf_y_cov'].values[::subsample_ratio]
    eskf_z_cov = df['eskf_z_cov'].values[::subsample_ratio]
    inekf_x = df['inekf_x'].values[::subsample_ratio]
    inekf_y = df['inekf_y'].values[::subsample_ratio]
    inekf_z = df['inekf_z'].values[::subsample_ratio]
    inekf_x_cov = df['inekf_x_cov'].values[::subsample_ratio]
    inekf_y_cov = df['inekf_y_cov'].values[::subsample_ratio]
    inekf_z_cov = df['inekf_z_cov'].values[::subsample_ratio]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, x, color='black')
    ax1.plot(t, eskf_x, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_x, color='tab:red', alpha=0.5)
    ax1.fill_between(t, eskf_x + 3*eskf_x_cov, eskf_x - 3*eskf_x_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, inekf_x + 3*inekf_x_cov, inekf_x - 3*inekf_x_cov, color='tab:red', alpha=0.2)
    ax1.set_title('Position (X-axis)')
    ax1.set_ylabel('Position (m)')
    ax1.set_xlabel('Time (s)')

    ax2.plot(t, y, color='black')
    ax2.plot(t, eskf_y, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_y, color='tab:red', alpha=0.5)
    ax2.fill_between(t, eskf_y + 3*eskf_y_cov, eskf_y - 3*eskf_y_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, inekf_y + 3*inekf_y_cov, inekf_y - 3*inekf_y_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Position (Y-axis)')
    ax2.set_ylabel('Position (m)')
    ax2.set_xlabel('Time (s)')

    ax3.plot(t, z, color='black')
    ax3.plot(t, eskf_z, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_z, color='tab:red', alpha=0.5)
    ax3.fill_between(t, eskf_z + 3*eskf_z_cov, eskf_z - 3*eskf_z_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, inekf_z + 3*inekf_z_cov, inekf_z - 3*inekf_z_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Position (Z-axis)')
    ax3.set_ylabel('Position (m)')
    ax3.set_xlabel('Time (s)')

    hK, = plt.plot([0,0],'k-')
    hB, = plt.plot([0,0],'b-')
    hR, = plt.plot([0,0],'r-')
    ax1.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Example Trial (Position)', fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig('paper/figures/example-run-pos.pdf')

def exemplar_vel_plot():
    df = pd.read_csv("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/main/const4-trial7-tdoa2-manual1.csv.zst")
    # df = pd.read_csv("results-tdoa0.13-std_pos0.1-std_vel0.01-std_yaw1.0/main/const4-trial7-tdoa2-manual1.csv.zst")

    subsample_ratio = 128
    t = df['t'].values[::subsample_ratio]
    vx = df['vx'].values[::subsample_ratio]
    vy = df['vy'].values[::subsample_ratio]
    vz = df['vz'].values[::subsample_ratio]
    eskf_vx = df['eskf_vx'].values[::subsample_ratio]
    eskf_vy = df['eskf_vy'].values[::subsample_ratio]
    eskf_vz = df['eskf_vz'].values[::subsample_ratio]
    eskf_vx_cov = df['eskf_vx_cov'].values[::subsample_ratio]
    eskf_vy_cov = df['eskf_vy_cov'].values[::subsample_ratio]
    eskf_vz_cov = df['eskf_vz_cov'].values[::subsample_ratio]
    inekf_vx = df['inekf_vx'].values[::subsample_ratio]
    inekf_vy = df['inekf_vy'].values[::subsample_ratio]
    inekf_vz = df['inekf_vz'].values[::subsample_ratio]
    inekf_vx_cov = df['inekf_vx_cov'].values[::subsample_ratio]
    inekf_vy_cov = df['inekf_vy_cov'].values[::subsample_ratio]
    inekf_vz_cov = df['inekf_vz_cov'].values[::subsample_ratio]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, vx, color='black')
    ax1.plot(t, eskf_vx, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_vx, color='tab:red', alpha=0.5)
    ax1.fill_between(t, eskf_vx + 3*eskf_vx_cov, eskf_vx - 3*eskf_vx_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, inekf_vx + 3*inekf_vx_cov, inekf_vx - 3*inekf_vx_cov, color='tab:red', alpha=0.2)
    ax1.set_title('Velocity (X-axis)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_xlabel('Time (s)')

    ax2.plot(t, vy, color='black')
    ax2.plot(t, eskf_vy, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_vy, color='tab:red', alpha=0.5)
    ax2.fill_between(t, eskf_vy + 3*eskf_vy_cov, eskf_vy - 3*eskf_vy_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, inekf_vy + 3*inekf_vy_cov, inekf_vy - 3*inekf_vy_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Velocity (Y-axis)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_xlabel('Time (s)')

    ax3.plot(t, vz, color='black')
    ax3.plot(t, eskf_vz, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_vz, color='tab:red', alpha=0.5)
    ax3.fill_between(t, eskf_vz + 3*eskf_vz_cov, eskf_vz - 3*eskf_vz_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, inekf_vz + 3*inekf_vz_cov, inekf_vz - 3*inekf_vz_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Velocity (Z-axis)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_xlabel('Time (s)')

    hK, = plt.plot([0,0],'k-')
    hB, = plt.plot([0,0],'b-')
    hR, = plt.plot([0,0],'r-')
    ax1.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Example Trial (Velocity)', fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig('paper/figures/example-run-vel.pdf')

def exemplar_ori_plot():
    df = pd.read_csv("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/main/const4-trial7-tdoa2-manual1.csv.zst")
    # df = pd.read_csv("results-tdoa0.13-std_pos0.1-std_vel0.01-std_yaw1.0/main/const4-trial7-tdoa2-manual1.csv.zst")

    subsample_ratio = 128
    t = df['t'].values[::subsample_ratio]
    ox = df['ox'].values[::subsample_ratio]
    oy = df['oy'].values[::subsample_ratio]
    oz = df['oz'].values[::subsample_ratio]
    eskf_ox = df['eskf_ox'].values[::subsample_ratio]
    eskf_oy = df['eskf_oy'].values[::subsample_ratio]
    eskf_oz = df['eskf_oz'].values[::subsample_ratio]
    eskf_ox_cov = df['eskf_ox_cov'].values[::subsample_ratio]
    eskf_oy_cov = df['eskf_oy_cov'].values[::subsample_ratio]
    eskf_oz_cov = df['eskf_oz_cov'].values[::subsample_ratio]
    inekf_ox = df['inekf_ox'].values[::subsample_ratio]
    inekf_oy = df['inekf_oy'].values[::subsample_ratio]
    inekf_oz = df['inekf_oz'].values[::subsample_ratio]
    inekf_ox_cov = df['inekf_ox_cov'].values[::subsample_ratio]
    inekf_oy_cov = df['inekf_oy_cov'].values[::subsample_ratio]
    inekf_oz_cov = df['inekf_oz_cov'].values[::subsample_ratio]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(9, 5))

    ax1.plot(t, ox, color='black')
    ax1.plot(t, eskf_ox, color='tab:blue', alpha=0.5)
    ax1.plot(t, inekf_ox, color='tab:red', alpha=0.5)
    ax1.fill_between(t, eskf_ox + 3*eskf_ox_cov, eskf_ox - 3*eskf_ox_cov, color='tab:blue', alpha=0.2)
    ax1.fill_between(t, inekf_ox + 3*inekf_ox_cov, inekf_ox - 3*inekf_ox_cov, color='tab:red', alpha=0.2)
    ax1.set_title('Orientation (X-axis)')
    ax1.set_ylabel('Angle (rad)')
    ax1.set_xlabel('Time (s)')

    ax2.plot(t, oy, color='black')
    ax2.plot(t, eskf_oy, color='tab:blue', alpha=0.5)
    ax2.plot(t, inekf_oy, color='tab:red', alpha=0.5)
    ax2.fill_between(t, eskf_oy + 3*eskf_oy_cov, eskf_oy - 3*eskf_oy_cov, color='tab:blue', alpha=0.2)
    ax2.fill_between(t, inekf_oy + 3*inekf_oy_cov, inekf_oy - 3*inekf_oy_cov, color='tab:red', alpha=0.2)
    ax2.set_title('Orientation (Y-axis)')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_xlabel('Time (s)')

    ax3.plot(t, oz, color='black')
    ax3.plot(t, eskf_oz, color='tab:blue', alpha=0.5)
    ax3.plot(t, inekf_oz, color='tab:red', alpha=0.5)
    ax3.fill_between(t, eskf_oz + 3*eskf_oz_cov, eskf_oz - 3*eskf_oz_cov, color='tab:blue', alpha=0.2)
    ax3.fill_between(t, inekf_oz + 3*inekf_oz_cov, inekf_oz - 3*inekf_oz_cov, color='tab:red', alpha=0.2)
    ax3.set_title('Orientation (Z-axis)')
    ax3.set_ylabel('Angle (rad)')
    ax3.set_xlabel('Time (s)')

    hK, = plt.plot([0,0],'k-')
    hB, = plt.plot([0,0],'b-')
    hR, = plt.plot([0,0],'r-')
    ax1.legend((hK, hB, hR), ('Ground Truth', 'ESKF', 'InEKF'), loc='upper right')
    hK.set_visible(False)
    hB.set_visible(False)
    hR.set_visible(False)

    fig.suptitle('Example Trial (Orientation)', fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig('paper/figures/example-run-ori.pdf')

def exemplar_traj_plot():
    df = pd.read_csv("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/main/const4-trial7-tdoa2-manual1.csv.zst")

    anchor_survey = np.load("dataset/flight-dataset/survey-results/anchor_const4.npz")
    # select anchor constellations
    anchor_pos = anchor_survey['an_pos']

    subsample_ratio = 128
    t = df['t'].values[::subsample_ratio]
    x = df['x'].values[::subsample_ratio]
    y = df['y'].values[::subsample_ratio]
    z = df['z'].values[::subsample_ratio]
    eskf_x = df['eskf_x'].values[::subsample_ratio]
    eskf_y = df['eskf_y'].values[::subsample_ratio]
    eskf_z = df['eskf_z'].values[::subsample_ratio]
    eskf_x_cov = df['eskf_x_cov'].values[::subsample_ratio]
    eskf_y_cov = df['eskf_y_cov'].values[::subsample_ratio]
    eskf_z_cov = df['eskf_z_cov'].values[::subsample_ratio]
    inekf_x = df['inekf_x'].values[::subsample_ratio]
    inekf_y = df['inekf_y'].values[::subsample_ratio]
    inekf_z = df['inekf_z'].values[::subsample_ratio]
    inekf_x_cov = df['inekf_x_cov'].values[::subsample_ratio]
    inekf_y_cov = df['inekf_y_cov'].values[::subsample_ratio]
    inekf_z_cov = df['inekf_z_cov'].values[::subsample_ratio]

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

    ax_t.scatter(anchor_pos[:,0], anchor_pos[:,1], anchor_pos[:,2],color='Teal', s = 100, alpha = 0.9, label = 'Anchors')
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

    fig.suptitle('Example Trial Trajectory', fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig('paper/figures/example-run-traj.pdf')

def orientation_sensitivity_multiplot():
    files = sorted(glob.glob("results-tdoa0.13-std_pos0.1-std_vel0.01-std_yaw1.0/initial-yaw/const4-trial7-tdoa2-manual3-yaw*.csv.zst"))

    # remove extreme ESKF cases that make reading the plot hard
    # reg = re.compile('.*-yaw-15.*')
    # files = list(filter(lambda x: not reg.search(x), files))
    # reg = re.compile('.*yaw\+9.*')
    # files = list(filter(lambda x: not reg.search(x), files))

    t = []

    eskf_error = []
    inekf_error = []

    time_of_first_motion = 0

    for file in files:
        print("Reading: ", file)
        df = pd.read_csv(file)

        # the RMS values are for analysis and debugging purposes
        eskf_rms_x = np.sqrt(np.mean((df['eskf_x'] - df['x'])**2))
        eskf_rms_y = np.sqrt(np.mean((df['eskf_y'] - df['y'])**2))
        eskf_rms_z = np.sqrt(np.mean((df['eskf_z'] - df['z'])**2))
        eskf_rms_pos = np.sqrt(eskf_rms_x**2 + eskf_rms_y**2 + eskf_rms_z**2)
        print("ESKF RMS error x: ", eskf_rms_x)
        print("ESKF RMS error y: ", eskf_rms_y)
        print("ESKF RMS error z: ", eskf_rms_z)
        print("ESKF RMS error pos: ", eskf_rms_pos)
        eskf_rms_vx = np.sqrt(np.mean((df['eskf_vx'] - df['vx'])**2))
        eskf_rms_vy = np.sqrt(np.mean((df['eskf_vy'] - df['vy'])**2))
        eskf_rms_vz = np.sqrt(np.mean((df['eskf_vz'] - df['vz'])**2))
        eskf_rms_vel = np.sqrt(eskf_rms_vx**2 + eskf_rms_vy**2 + eskf_rms_vz**2)
        print("ESKF RMS error vx: ", eskf_rms_vx)
        print("ESKF RMS error vy: ", eskf_rms_vy)
        print("ESKF RMS error vz: ", eskf_rms_vz)
        print("ESKF RMS error vel: ", eskf_rms_vel)
        eskf_rms_ox = np.sqrt(np.mean((df['eskf_ox'] - df['ox'])**2))
        eskf_rms_oy = np.sqrt(np.mean((df['eskf_oy'] - df['oy'])**2))
        eskf_rms_oz = np.sqrt(np.mean((df['eskf_oz'] - df['oz'])**2))
        eskf_rms_ori = np.sqrt(eskf_rms_ox**2 + eskf_rms_oy**2 + eskf_rms_oz**2)
        print("ESKF RMS error ox: ", eskf_rms_ox)
        print("ESKF RMS error oy: ", eskf_rms_oy)
        print("ESKF RMS error oz: ", eskf_rms_oz)
        print("ESKF RMS error ori: ", eskf_rms_ori)
        inekf_rms_x = np.sqrt(np.mean((df['inekf_x'] - df['x'])**2))
        inekf_rms_y = np.sqrt(np.mean((df['inekf_y'] - df['y'])**2))
        inekf_rms_z = np.sqrt(np.mean((df['inekf_z'] - df['z'])**2))
        inekf_rms_pos = np.sqrt(inekf_rms_x**2 + inekf_rms_y**2 + inekf_rms_z**2)
        print("IEKF RMS error x: ", inekf_rms_x)
        print("IEKF RMS error y: ", inekf_rms_y)
        print("IEKF RMS error z: ", inekf_rms_z)
        print("IEKF RMS error pos: ", inekf_rms_pos)
        inekf_rms_vx = np.sqrt(np.mean((df['inekf_vx'] - df['vx'])**2))
        inekf_rms_vy = np.sqrt(np.mean((df['inekf_vy'] - df['vy'])**2))
        inekf_rms_vz = np.sqrt(np.mean((df['inekf_vz'] - df['vz'])**2))
        inekf_rms_vel = np.sqrt(inekf_rms_vx**2 + inekf_rms_vy**2 + inekf_rms_vz**2)
        print("IEKF RMS error vx: ", inekf_rms_vx)
        print("IEKF RMS error vy: ", inekf_rms_vy)
        print("IEKF RMS error vz: ", inekf_rms_vz)
        print("IEKF RMS error vel: ", inekf_rms_vel)
        inekf_rms_ox = np.sqrt(np.mean((df['inekf_ox'] - df['ox'])**2))
        inekf_rms_oy = np.sqrt(np.mean((df['inekf_oy'] - df['oy'])**2))
        inekf_rms_oz = np.sqrt(np.mean((df['inekf_oz'] - df['oz'])**2))
        inekf_rms_ori = np.sqrt(inekf_rms_ox**2 + inekf_rms_oy**2 + inekf_rms_oz**2)
        print("IEKF RMS error ox: ", inekf_rms_ox)
        print("IEKF RMS error oy: ", inekf_rms_oy)
        print("IEKF RMS error oz: ", inekf_rms_oz)
        print("IEKF RMS error ori: ", inekf_rms_ori)
        print()

        eskf_rms_ox = np.sqrt(np.mean((df['eskf_ox'] - df['ox'])**2))
        eskf_rms_oy = np.sqrt(np.mean((df['eskf_oy'] - df['oy'])**2))
        eskf_rms_oz = np.sqrt(np.mean((df['eskf_oz'] - df['oz'])**2))
        eskf_rms_ori = np.sqrt(eskf_rms_ox**2 + eskf_rms_oy**2 + eskf_rms_oz**2)
        # print("ESKF RMS error ori: ", eskf_rms_ori)

        inekf_rms_ox = np.sqrt(np.mean((df['inekf_ox'] - df['ox'])**2))
        inekf_rms_oy = np.sqrt(np.mean((df['inekf_oy'] - df['oy'])**2))
        inekf_rms_oz = np.sqrt(np.mean((df['inekf_oz'] - df['oz'])**2))
        inekf_rms_ori = np.sqrt(inekf_rms_ox**2 + inekf_rms_oy**2 + inekf_rms_oz**2)
        # print("IEKF RMS error ori: ", inekf_rms_ori)
        # print()

        # determine when the vehicle begins moving
        t = df['t']
        speed = np.linalg.norm(df[['vx','vy','vz']].values,axis=1)
        index_of_first_motion = np.where(speed > 0.1)[0][0]
        time_of_first_motion = t[index_of_first_motion]

        subsample_ratio = 128
        t = df['t'].values[::subsample_ratio]
        oz = df['oz'].values[::subsample_ratio]
        eskf_oz = df['eskf_oz'].values[::subsample_ratio]
        inekf_oz = df['inekf_oz'].values[::subsample_ratio]

        eskf_error.append(eskf_oz - oz)
        inekf_error.append(inekf_oz - oz)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 5))
    for i in range(len(files)):
        ax1.plot(t, eskf_error[i], linewidth=1)
        ax2.plot(t, inekf_error[i], linewidth=1)

    ax1.set_title('ESKF Yaw Error')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Error (rad)')
    ax1.set_ylim(-np.pi,np.pi)
    # ax1.axvline(x=time_of_first_motion, color='tab:red')
    ax1.axhline(y=0, color='black')

    ax2.set_title('InEKF Yaw Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (rad)')
    ax2.set_ylim(-np.pi,np.pi)
    # ax2.axvline(x=time_of_first_motion, color='tab:red')
    ax2.axhline(y=0, color='black')

    fig.suptitle('Initial Yaw Error Trajectories', fontsize=FONTSIZE*1.5)

    plt.tight_layout()
    plt.savefig('paper/figures/orientation-sensitivity-multiplot.pdf')

def position_sensitivity_multiplot():
    files = sorted(glob.glob("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/initial-position/const4-trial7-tdoa2-manual1-pos*.csv.zst"))
    # files = sorted(glob.glob("results-tdoa0.13-std_pos0.1-std_vel0.01-std_yaw1.0/initial-position/const4-trial7-tdoa2-manual1-pos*.csv.zst"))

    t = []

    eskf_error = []
    inekf_error = []

    time_of_first_motion = 0

    for file in files:
        print("Reading: ", file)
        df = pd.read_csv(file)

        t = df['t']

        speed = np.linalg.norm(df[['vx','vy','vz']].values,axis=1)
        index_of_first_motion = np.where(speed > 0.1)[0][0]
        time_of_first_motion = t[index_of_first_motion]

        eskf_rms_x = np.sqrt(np.mean((df['eskf_x'] - df['x'])**2))
        eskf_rms_y = np.sqrt(np.mean((df['eskf_y'] - df['y'])**2))
        eskf_rms_z = np.sqrt(np.mean((df['eskf_z'] - df['z'])**2))
        eskf_rms_pos = np.sqrt(eskf_rms_x**2 + eskf_rms_y**2 + eskf_rms_z**2)
        print("ESKF RMS error x: ", eskf_rms_x)
        print("ESKF RMS error y: ", eskf_rms_y)
        print("ESKF RMS error z: ", eskf_rms_z)
        print("ESKF RMS error pos: ", eskf_rms_pos)
        eskf_rms_vx = np.sqrt(np.mean((df['eskf_vx'] - df['vx'])**2))
        eskf_rms_vy = np.sqrt(np.mean((df['eskf_vy'] - df['vy'])**2))
        eskf_rms_vz = np.sqrt(np.mean((df['eskf_vz'] - df['vz'])**2))
        eskf_rms_vel = np.sqrt(eskf_rms_vx**2 + eskf_rms_vy**2 + eskf_rms_vz**2)
        print("ESKF RMS error vx: ", eskf_rms_vx)
        print("ESKF RMS error vy: ", eskf_rms_vy)
        print("ESKF RMS error vz: ", eskf_rms_vz)
        print("ESKF RMS error vel: ", eskf_rms_vel)
        eskf_rms_ox = np.sqrt(np.mean((df['eskf_ox'] - df['ox'])**2))
        eskf_rms_oy = np.sqrt(np.mean((df['eskf_oy'] - df['oy'])**2))
        eskf_rms_oz = np.sqrt(np.mean((df['eskf_oz'] - df['oz'])**2))
        eskf_rms_ori = np.sqrt(eskf_rms_ox**2 + eskf_rms_oy**2 + eskf_rms_oz**2)
        print("ESKF RMS error ox: ", eskf_rms_ox)
        print("ESKF RMS error oy: ", eskf_rms_oy)
        print("ESKF RMS error oz: ", eskf_rms_oz)
        print("ESKF RMS error ori: ", eskf_rms_ori)
        inekf_rms_x = np.sqrt(np.mean((df['inekf_x'] - df['x'])**2))
        inekf_rms_y = np.sqrt(np.mean((df['inekf_y'] - df['y'])**2))
        inekf_rms_z = np.sqrt(np.mean((df['inekf_z'] - df['z'])**2))
        inekf_rms_pos = np.sqrt(inekf_rms_x**2 + inekf_rms_y**2 + inekf_rms_z**2)
        print("IEKF RMS error x: ", inekf_rms_x)
        print("IEKF RMS error y: ", inekf_rms_y)
        print("IEKF RMS error z: ", inekf_rms_z)
        print("IEKF RMS error pos: ", inekf_rms_pos)
        inekf_rms_vx = np.sqrt(np.mean((df['inekf_vx'] - df['vx'])**2))
        inekf_rms_vy = np.sqrt(np.mean((df['inekf_vy'] - df['vy'])**2))
        inekf_rms_vz = np.sqrt(np.mean((df['inekf_vz'] - df['vz'])**2))
        inekf_rms_vel = np.sqrt(inekf_rms_vx**2 + inekf_rms_vy**2 + inekf_rms_vz**2)
        print("IEKF RMS error vx: ", inekf_rms_vx)
        print("IEKF RMS error vy: ", inekf_rms_vy)
        print("IEKF RMS error vz: ", inekf_rms_vz)
        print("IEKF RMS error vel: ", inekf_rms_vel)
        inekf_rms_ox = np.sqrt(np.mean((df['inekf_ox'] - df['ox'])**2))
        inekf_rms_oy = np.sqrt(np.mean((df['inekf_oy'] - df['oy'])**2))
        inekf_rms_oz = np.sqrt(np.mean((df['inekf_oz'] - df['oz'])**2))
        inekf_rms_ori = np.sqrt(inekf_rms_ox**2 + inekf_rms_oy**2 + inekf_rms_oz**2)
        print("IEKF RMS error ox: ", inekf_rms_ox)
        print("IEKF RMS error oy: ", inekf_rms_oy)
        print("IEKF RMS error oz: ", inekf_rms_oz)
        print("IEKF RMS error ori: ", inekf_rms_ori)
        print()

        eskf_rms_ox = np.sqrt(np.mean((df['eskf_ox'] - df['ox'])**2))
        eskf_rms_oy = np.sqrt(np.mean((df['eskf_oy'] - df['oy'])**2))
        eskf_rms_oz = np.sqrt(np.mean((df['eskf_oz'] - df['oz'])**2))
        eskf_rms_ori = np.sqrt(eskf_rms_ox**2 + eskf_rms_oy**2 + eskf_rms_oz**2)
        # print("ESKF RMS error ori: ", eskf_rms_ori)

        inekf_rms_ox = np.sqrt(np.mean((df['inekf_ox'] - df['ox'])**2))
        inekf_rms_oy = np.sqrt(np.mean((df['inekf_oy'] - df['oy'])**2))
        inekf_rms_oz = np.sqrt(np.mean((df['inekf_oz'] - df['oz'])**2))
        inekf_rms_ori = np.sqrt(inekf_rms_ox**2 + inekf_rms_oy**2 + inekf_rms_oz**2)
        # print("IEKF RMS error ori: ", inekf_rms_ori)
        # print()

        eskf_error.append(np.linalg.norm(df[['eskf_x','eskf_y','eskf_z']].values - df[['x','y','z']].values,axis=1))
        inekf_error.append(np.linalg.norm(df[['inekf_x','inekf_y','inekf_z']].values - df[['x','y','z']].values,axis=1))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5.4, 4.8))
    for i in range(len(files)):
        ax1.plot(t, eskf_error[i], linewidth=1)
        ax2.plot(t, inekf_error[i], linewidth=1)

    ax1.set_title('ESKF Position Error')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Error (m)')
    # ax1.axvline(x=time_of_first_motion, color='tab:red')
    ax1.axhline(y=0, color='black')
    ax1.axhline(y=0.5, color='black', linestyle='dotted')
    ax1.axhline(y=1, color='black', linestyle='dotted')
    ax1.axhline(y=1.5, color='black', linestyle='dotted')
    ax1.axhline(y=2, color='black', linestyle='dotted')
    ax1.set_ylim([0,2.5])

    ax2.set_title('InEKF Position Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (m)')
    # ax2.axvline(x=time_of_first_motion, color='tab:red')
    ax2.axhline(y=0, color='black')
    ax2.axhline(y=0.5, color='black', linestyle='dotted')
    ax2.axhline(y=1, color='black', linestyle='dotted')
    ax2.axhline(y=1.5, color='black', linestyle='dotted')
    ax2.axhline(y=2, color='black', linestyle='dotted')
    ax2.set_ylim([0,2.5])

    fig.suptitle('Initial Position Error Trajectories', fontsize=FONTSIZE*1.5)

    plt.tight_layout()
    plt.savefig('paper/figures/position-sensitivity-multiplot.pdf')

def constellation_plot():
    for i in range(1, 5):
        anchor_survey = np.load("dataset/flight-dataset/survey-results/anchor_const" + str(i) + ".npz")
        # select anchor constellations
        anchor_pos = anchor_survey['an_pos']

        # load an example trajectory
        if i == 1:
            csv_file = "dataset/flight-dataset/csv-data/const" + str(i) + "/const" + str(i) + "-trial2-tdoa2.csv"
        elif i == 2:
            csv_file = "dataset/flight-dataset/csv-data/const" + str(i) + "/const" + str(i) + "-trial2-tdoa3.csv"
        elif i == 3:
            csv_file = "dataset/flight-dataset/csv-data/const" + str(i) + "/const" + str(i) + "-trial7-tdoa2-manual1.csv"
        elif i == 4:
            csv_file = "dataset/flight-dataset/csv-data/const" + str(i) + "/const" + str(i) + "-trial7-tdoa2-manual2.csv"
        df = pd.read_csv(csv_file)

        gt_pose = extract_gt(df)
        pos_vicon = gt_pose[:,1:4]

        fig_traj = plt.figure(facecolor = "white",figsize=(6/2, 5/2))
        ax_t = fig_traj.add_subplot(projection='3d')

        # make the panes transparent
        ax_t.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_t.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_t.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # change the color of the grid lines
        ax_t.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
        ax_t.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
        ax_t.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)

        ax_t.plot(pos_vicon[:,0],pos_vicon[:,1],pos_vicon[:,2],color='b',linewidth=0.5, alpha=0.7, label='ground truth')
        ax_t.scatter(anchor_pos[:,0], anchor_pos[:,1], anchor_pos[:,2],color='Teal', s = 20, alpha = 0.9, label = 'anchors')
        ax_t.set_xlim([-3.5,3.5])
        ax_t.set_ylim([-3.9,3.9])
        ax_t.set_zlim([-0.0,3.0])

        # use LaTeX fonts in the plot
        ax_t.set_xlabel(r'X [m]') #,fontsize=FONTSIZE)
        ax_t.set_ylabel(r'Y [m]') #,fontsize=FONTSIZE)
        ax_t.set_zlabel(r'Z [m]') #,fontsize=FONTSIZE)
        # ax_t.legend(loc='upper left', fontsize=FONTSIZE)
        ax_t.view_init(24, -58)
        ax_t.set_box_aspect((1, 1, 0.5))  # xy aspect ratio is 1:1, but change z axis

        ax_t.set_title('Constellation ' + str(i), fontsize=FONTSIZE*1.5)

        plt.tight_layout()
        plt.savefig('paper/figures/const' + str(i) + '-traj.pdf')

if __name__ == "__main__":
    # set window background to white
    plt.rcParams['figure.facecolor'] = 'w'

    mpl.rc('xtick', labelsize=TICK_SIZE)
    mpl.rc('ytick', labelsize=TICK_SIZE)

    print("Performing Error Boxplot")
    error_boxplot()

    print("Plotting Example Trajectory")
    exemplar_pos_plot()
    exemplar_vel_plot()
    exemplar_ori_plot()
    exemplar_traj_plot()

    print("Performing Position Sensitivity Boxplot")
    position_sensitivity_boxplot()

    print("Performing Position Sensitivity Multiplot")
    position_sensitivity_multiplot()

    print("Performing Orientation Sensitivity Boxplot")
    orientation_sensitivity_boxplot()

    print("Performing Orientation Sensitivity Multiplot")
    orientation_sensitivity_multiplot()

    print("Plotting Constellation")
    constellation_plot()

    plt.show()
