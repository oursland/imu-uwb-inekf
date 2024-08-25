#!/usr/bin/env python3

import glob
import os

import numpy as np
import pandas as pd

def generate_table(directory):
    rows = []

    for const in range(1, 5):
        files = sorted(glob.glob(directory + "const" + str(const) + "*.csv.zst"))

        for file in files:
            filename = os.path.splitext(os.path.basename(file))[0]

            print("Reading: ", filename)
            df = pd.read_csv(file)

            for filter in ['eskf', 'inekf']:
                rms_x = np.sqrt(np.mean((df[filter + '_x'] - df['x'])**2))
                rms_y = np.sqrt(np.mean((df[filter + '_y'] - df['y'])**2))
                rms_z = np.sqrt(np.mean((df[filter + '_z'] - df['z'])**2))
                rms_pos = np.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
                rms_vx = np.sqrt(np.mean((df[filter + '_vx'] - df['vx'])**2))
                rms_vy = np.sqrt(np.mean((df[filter + '_vy'] - df['vy'])**2))
                rms_vz = np.sqrt(np.mean((df[filter + '_vz'] - df['vz'])**2))
                rms_vel = np.sqrt(rms_vx**2 + rms_vy**2 + rms_vz**2)
                # ESKF sometimes returns really large errors in orientation for a single step that messes up the RMS values
                e_ox = (df[filter + '_ox'] - df['ox'])**2
                e_oy = (df[filter + '_oy'] - df['oy'])**2
                e_oz = (df[filter + '_oz'] - df['oz'])**2
                rms_ox = np.sqrt(np.mean(np.where(e_ox < np.pi**2, e_ox, 0)))
                rms_oy = np.sqrt(np.mean(np.where(e_oy < np.pi**2, e_oy, 0)))
                rms_oz = np.sqrt(np.mean(np.where(e_oz < np.pi**2, e_oz, 0)))
                rms_ori = np.sqrt(rms_ox**2 + rms_oy**2 + rms_oz**2)

                row = {
                    "file": filename,
                    "const": const,
                    "filter": filter,
                    "rms_x": rms_x,
                    "rms_y": rms_y,
                    "rms_z": rms_z,
                    "rms_pos": rms_pos,
                    "rms_vx": rms_vx,
                    "rms_vy": rms_vy,
                    "rms_vz": rms_vz,
                    "rms_vel": rms_vel,
                    "rms_ox": rms_ox,
                    "rms_oy": rms_oy,
                    "rms_oz": rms_oz,
                    "rms_ori": rms_ori,
                }
                rows.append(row)

    output_name = directory + "summary.csv"

    df = pd.DataFrame(rows)
    df.to_csv(output_name)

if __name__ == "__main__":
    generate_table("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/main/")
    generate_table("results-tdoa0.05-std_pos0.1-std_vel0.01-std_yaw0.01/initial-position/")
    generate_table("results-tdoa0.13-std_pos0.1-std_vel0.01-std_yaw1.0/initial-yaw/")
