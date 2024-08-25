import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

FONTSIZE = 12
TICK_SIZE = 0

def plot_collsion():
    anchor_survey = np.load("dataset/flight-dataset/survey-results/anchor_const1.npz")
    # select anchor constellations
    anchor_pos = anchor_survey['an_pos']

    fig_traj = plt.figure(facecolor = "white", figsize=(5, 5))
    ax = fig_traj.add_subplot(projection='3d', computed_zorder=False)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # change the color of the grid lines
    ax.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)


    # Draw a circle on the x=0 'wall'
    p = Rectangle((-4, 0), 8, 3, alpha=0.9, label='Wall')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=3.5, zdir="y")

    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, 2, 100)
    r = 2
    x = r * np.sin(theta)
    y = r * np.cos(theta) - 0.5

    ax.plot(x, y, z, color='k',linewidth=2.0, alpha=1, label='Desired Trajectory')

    theta = np.linspace(np.pi, 1.5 * np.pi, 100)
    z = np.linspace(0, 0.5, 100)
    r = 2
    x = r * np.sin(theta)
    y = r * np.cos(theta) + 3.5

    ax.plot(x, y, z, color='r',linewidth=2.0, alpha=1, label='Actual Trajectory', zorder=3)
    ax.plot(-2, 3.5, 0.5, '*', markerfacecolor='y', markeredgecolor='r', markersize=20, alpha=1.0, label="Collision")

    ax.scatter(anchor_pos[:,0], anchor_pos[:,1], anchor_pos[:,2], color='Teal', s = 100, alpha = 0.9, label = 'UWB Anchors')
    ax.set_xlim([-3.5,3.5])
    ax.set_ylim([-3.9,3.9])
    ax.set_zlim([-0.0,3.0])

    # use LaTeX fonts in the plot
    # ax.set_xlabel(r'X [m]') #,fontsize=FONTSIZE)
    # ax.set_ylabel(r'Y [m]') #,fontsize=FONTSIZE)
    # ax.set_zlabel(r'Z [m]') #,fontsize=FONTSIZE)
    ax.legend(loc='upper left', fontsize=FONTSIZE)
    ax.view_init(30, -10)
    ax.set_box_aspect((1, 1, 0.5))  # xy aspect ratio is 1:1, but change z axis

    ax.set_title('Collision from Poor Heading Estimation', fontsize=FONTSIZE*1.5)

    plt.tight_layout()
    plt.savefig('paper/figures/collision.pdf')


if __name__ == "__main__":
    # set window background to white
    plt.rcParams['figure.facecolor'] = 'w'

    mpl.rc('xtick', labelsize=TICK_SIZE)
    mpl.rc('ytick', labelsize=TICK_SIZE)

    plot_collsion()

    plt.show()
