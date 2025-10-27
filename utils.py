import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from sympy.algebras.quaternion import Quaternion
from typing import List, Iterable
from typing import TypedDict


def smaa(expression, small_angle):
    """Small angle approximation cos and sin"""
    return expression.replace(
        lambda e: e.func == sm.sin and e.args[0] == small_angle, lambda e: e.args[0]
    ).replace(lambda e: e.func == sm.cos and e.args[0] == small_angle, lambda e: 1)


class FrameDict(TypedDict):
    longname: str
    name: str
    frame: me.ReferenceFrame
    position: me.Point


def frame_rotation_to_list(
    out_frame: me.ReferenceFrame, ref_frame: me.ReferenceFrame, subs_dict={}
):
    """output is eurler xyz"""
    q = Quaternion.from_rotation_matrix(out_frame.dcm(ref_frame).subs(subs_dict).T)
    ex, ey, ez = q.to_euler("xyz")
    return np.array([ex.evalf(), ey.evalf(), ez.evalf()]).astype(float).tolist()


def point_position_to_list(point: me.Point, origin, ref_frame: me.ReferenceFrame, subs_dict={}):
    return (
        np.array(point.pos_from(origin).to_matrix(ref_frame).subs(subs_dict))
        .astype(float)
        .flatten()
        .tolist()
    )


def frames_to_dict(
    frames: Iterable[FrameDict], origin: me.Point, ref_frame: me.ReferenceFrame, subs={}
):
    out = []
    for f in frames:
        rot = frame_rotation_to_list(f["frame"], ref_frame, subs)
        pos = point_position_to_list(f["position"], origin, ref_frame, subs)
        out.append({"longname": f["longname"], "name": f["name"], "rot": rot, "pos": pos})
    return out
