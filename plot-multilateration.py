from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import math
import matplotlib as mpl
import pandas as pd

from matplotlib.patches import Patch, Circle, Rectangle, Arrow

FONTSIZE = 16
TICK_SIZE = 0

mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
})

anchors = np.array([
    [ 1.0, 5.0 ],
    [ 1.0, 1.0 ],
    [ 5.0, 1.0 ],
])

tags = np.array([
    [ 3.0, 4.0 ],
])

legend_elements = [
    Circle((0, 0), 1, color='r', label='Anchor'),
    Rectangle((0, 0), 1, 1, color='b', label='Tag'),
    # Arrow(0, 0, 0.1, 0.1, color='k', label='TWR Ranging'),
]

def plot_multilateration():
    fig, ax = plt.subplots(figsize=(4, 4))

    for i, tag in enumerate(tags):
        for j, anchor in enumerate(anchors):
            d = np.linalg.norm(anchor - (tag + [ 0.1, 0.1 ]))
            circle = plt.Circle(anchor, d, color='k', fill=False)
            ax.add_patch(circle)

    for i, anchor in enumerate(anchors):
        circle = plt.Circle(anchor, 0.2, color='r', label='Anchor')
        ax.add_patch(circle)
        # text = ax.text(anchor[0]-0.5, anchor[1]-0.5, 'Anchor ' + str(i+1), bbox=dict(facecolor='white', edgecolor='black'))

    for i, tag in enumerate(tags):
        rectangle = plt.Rectangle(tag, 0.2, 0.2, color='b', label='Tag')
        ax.add_patch(rectangle)
        # text = ax.text(tag[0]-0.1, tag[1]-0.3, 'Tag', bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_aspect('equal')
    ax.set_xlim((0, 6))
    ax.set_ylim((0, 6))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig('paper/figures/multilateration.pdf')
    plt.show()

def plot_twr():
    fig, ax = plt.subplots(figsize=(4, 4))

    for i, tag in enumerate(tags):
        for j, anchor in enumerate(anchors):
            arrow = plt.Arrow(*anchor, *(tag + [0.1,0.1] - anchor), color='k', width=0.2)
            ax.add_patch(arrow)
            arrow = plt.Arrow(*(tag + [0.1,0.1]), *(anchor - tag - [0.1, 0.1]), color='k', width=0.2)
            ax.add_patch(arrow)

    for i, anchor in enumerate(anchors):
        circle = plt.Circle(anchor, 0.2, color='r', label='Anchor')
        ax.add_patch(circle)
        # text = ax.text(anchor[0]-0.5, anchor[1]-0.5, 'Anchor ' + str(i+1), bbox=dict(facecolor='white', edgecolor='black'))

    for i, tag in enumerate(tags):
        rectangle = plt.Rectangle(tag, 0.2, 0.2, color='b', label='Tag')
        ax.add_patch(rectangle)
        # text = ax.text(tag[0]-0.1, tag[1]-0.3, 'Tag', bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_aspect('equal')
    ax.set_xlim((0, 6))
    ax.set_ylim((0, 6))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig('paper/figures/uwb-twr.pdf')
    plt.show()

def plot_tdoa():
    fig, ax = plt.subplots(figsize=(4, 4))

    for i, tag in enumerate(tags):
        for j, anchor in enumerate(anchors):
            d = np.linalg.norm(anchor - (tag + [ 0.1, 0.1 ]))
            circle = plt.Circle(anchor, d, color='k', fill=False)
            ax.add_patch(circle)

    for i, anchor1 in enumerate(anchors):
        for j, anchor2 in enumerate(anchors):
            arrow = plt.Arrow(*anchor1, *(anchor2 - anchor1), color='k', width=0.2)
            ax.add_patch(arrow)

    for i, anchor in enumerate(anchors):
        circle = plt.Circle(anchor, 0.2, color='r')
        ax.add_patch(circle)
        # text = ax.text(anchor[0]-0.5, anchor[1]-0.5, 'Anchor ' + str(i+1), bbox=dict(facecolor='white', edgecolor='black'))

    for i, tag in enumerate(tags):
        rectangle = plt.Rectangle(tag, 0.2, 0.2, color='b', label='Tag')
        ax.add_patch(rectangle)
        # text = ax.text(tag[0]-0.1, tag[1]-0.3, 'Tag', bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_aspect('equal')
    ax.set_xlim((0, 6))
    ax.set_ylim((0, 6))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig('paper/figures/uwb-tdoa.pdf')
    plt.show()

if __name__ == "__main__":
    # set window background to white
    plt.rcParams['figure.facecolor'] = 'w'

    mpl.rc('xtick', labelsize=TICK_SIZE)
    mpl.rc('ytick', labelsize=TICK_SIZE)

    plot_multilateration()
    plot_twr()
    plot_tdoa()
