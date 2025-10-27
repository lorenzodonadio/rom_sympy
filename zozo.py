import numpy as np

import plotly.graph_objects as go

from sympy import Symbol, symbols
from sympy import Matrix, Function, diff

# from sympy.abc import *
from sympy import cos, sin, pi

import sympy.physics.mechanics as me

# import Body as SympyBody
# from sympy.physics.mechanics import RigidBody as SympyRigidBody
# from sympy.physics.mechanics import Point, ReferenceFrame, inertia, dynamicsymbols
# from sympy.physics.mechanics.functions import msubs


class Rot:
    """Rotations, methods x, y, z for rotations along those angles (inverse cosine matrix)"""  # i think

    def __init__(self):
        pass

    @staticmethod
    def x(a):
        return Matrix([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])  # type: ignore

    @staticmethod
    def y(a):
        return Matrix([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])  # type: ignore

    @staticmethod
    def z(a):
        return Matrix([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])  # type: ignore


R = Rot()


def plot_frame(ax, dcm_mat: np.ndarray, origin: np.ndarray, color, label):
    # Get basis vectors
    x = dcm_mat[:, 0]
    y = dcm_mat[:, 1]
    z = dcm_mat[:, 2]
    # Plot arrows
    ax.quiver(*origin, *x, color=color, length=1, normalize=True)
    ax.quiver(*origin, *y, color=color, length=1, normalize=True)
    ax.quiver(*origin, *z, color=color, length=1, normalize=True)
    # Label
    ax.text(*(origin + x), f"{label}_x", color=color)
    ax.text(*(origin + y), f"{label}_y", color=color)
    ax.text(*(origin + z), f"{label}_z", color=color)


def plot_referenceframely(
    fig: go.Figure,
    dcm_mat: np.ndarray,
    origin: np.ndarray | list[float],
    label_prefix=None,
    color="#1f77b4",
):
    assert dcm_mat.shape == (3, 3)
    O = np.array(origin).flatten()
    assert len(O) == 3

    labels = (
        [f"{label_prefix}_x", f"{label_prefix}_y", f"{label_prefix}_z"]
        if label_prefix
        else ["", "", ""]
    )

    # Plot lines and cones for each axis
    for i in range(3):
        # Line from origin to axis tip
        tip = O + 0.6 * dcm_mat[:, i]
        fig.add_trace(
            go.Scatter3d(
                x=[O[0], tip[0]],
                y=[O[1], tip[1]],
                z=[O[2], tip[2]],
                mode="lines+text",
                text=["", labels[i]],
                textposition="top center",
                textfont=dict(size=16, color=color),
                line=dict(color=color, width=8),
                showlegend=False,
            )
        )
        # Cone at the tip, pointing in the axis direction
        fig.add_trace(
            go.Cone(
                x=[tip[0]],
                y=[tip[1]],
                z=[tip[2]],
                u=[dcm_mat[0, i]],
                v=[dcm_mat[1, i]],
                w=[dcm_mat[2, i]],
                sizemode="absolute",
                sizeref=0.4,
                anchor="tail",
                showscale=False,
                colorscale=[[0, color], [1, color]],
                cmin=0,
                cmax=1,
                name=labels[i],
            )
        )


import numpy as np
import pythreejs as p3js


def plot_ref_p3js(
    scene: p3js.Scene,
    dcm_mat: np.ndarray,
    origin: np.ndarray | list[float],
    label_prefix=None,
    size=1,
):
    O = np.array(origin).flatten()
    assert dcm_mat.shape == (3, 3)

    # Create AxesHelper
    helper = p3js.AxesHelper(size=size)

    # Set position
    helper.position = O.tolist()

    # Convert DCM to quaternion
    # PyThreeJS expects a flat list in column-major order for Matrix4
    dcm4 = np.eye(4)
    dcm4[:3, :3] = dcm_mat
    m = p3js.Matrix4(*dcm4.T.flatten().tolist())
    helper.matrix = m
    helper.matrixAutoUpdate = (
        False  # Otherwise Three.js will overwrite your manual matrix
    )

    scene.add(helper)
