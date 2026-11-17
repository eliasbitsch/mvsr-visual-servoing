"""Motion guard: blocks joint velocities that would lead into self-collision or out of the
allowed workspace within the look-ahead time. Motion that moves away from a violation stays
allowed, so the arm can always back out.
"""
import os
import re

import numpy as np
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory


def _package_dirs(urdf):
    pkgs = set(re.findall(r'package://([^/]+)/', urdf))
    return [os.path.dirname(get_package_share_directory(p)) for p in pkgs]


class SafetyMonitor:

    def __init__(self, urdf, arm_joints, margin=0.01, lookahead=0.2, min_height=0.03,
                 height_frames=(), workspace_frame='tool0', workspace=None):
        full = pin.buildModelFromXML(urdf)
        geom = pin.buildGeomFromUrdfString(full, urdf, pin.GeometryType.COLLISION,
                                           package_dirs=_package_dirs(urdf))
        lock = [full.getJointId(n) for n in full.names[1:] if n not in set(arm_joints)]
        self.model, (self.geom,) = pin.buildReducedModel(full, [geom], lock, pin.neutral(full))
        self.data = self.model.createData()

        # Parts on the same joint are rigidly connected, neighbours touch by design (no SRDF here)
        objs = self.geom.geometryObjects
        self.geom.addAllCollisionPairs()
        for cp in list(self.geom.collisionPairs):
            a, b = int(objs[cp.first].parentJoint), int(objs[cp.second].parentJoint)
            if abs(a - b) <= 1:
                self.geom.removeCollisionPair(cp)
        self.gdata = pin.GeometryData(self.geom)
        for req in self.gdata.collisionRequests:
            req.security_margin = margin

        self.lookahead = lookahead
        self.min_height = min_height
        self.height_ids = [self.model.getFrameId(f) for f in height_frames
                           if self.model.existFrame(f)]
        self.ws_id = self.model.getFrameId(workspace_frame)
        self.ws_lo, self.ws_hi = (np.array(workspace[:3]), np.array(workspace[3:])) \
            if workspace else (None, None)

    def _collisions(self, q):
        if not pin.computeCollisions(self.model, self.data, self.geom, self.gdata, q, False):
            return {}
        out = {}
        for i, res in enumerate(self.gdata.collisionResults):
            if res.isCollision():
                pin.computeDistance(self.geom, self.gdata, i)
                out[i] = self.gdata.distanceResults[i].min_distance
        return out

    def _names(self, i):
        cp = self.geom.collisionPairs[i]
        objs = self.geom.geometryObjects
        return f'{objs[cp.first].name.rsplit("_", 1)[0]}/{objs[cp.second].name.rsplit("_", 1)[0]}'

    def _positions(self, q):
        pin.framesForwardKinematics(self.model, self.data, q)
        return {i: self.data.oMf[i].translation.copy() for i in self.height_ids + [self.ws_id]}

    def check(self, q, qd):
        """Return None if qd is safe for the look-ahead time, else the reason."""
        q_next = pin.integrate(self.model, q, qd * self.lookahead)

        now, nxt = self._collisions(q), self._collisions(q_next)
        for i, dist in nxt.items():
            if i not in now or dist < now[i]:
                return f'self-collision ahead: {self._names(i)}'

        p_now, p_next = self._positions(q), self._positions(q_next)
        for i in self.height_ids:
            if p_next[i][2] < self.min_height and p_next[i][2] < p_now[i][2]:
                return f'{self.model.frames[i].name} below {self.min_height} m'
        if self.ws_lo is not None:
            lo, hi = self.ws_lo, self.ws_hi
            outside_next = np.maximum(lo - p_next[self.ws_id], 0) + np.maximum(p_next[self.ws_id] - hi, 0)
            outside_now = np.maximum(lo - p_now[self.ws_id], 0) + np.maximum(p_now[self.ws_id] - hi, 0)
            if np.any(outside_next > outside_now + 1e-6):
                return f'{self.model.frames[self.ws_id].name} leaving the workspace'
        return None

    def in_collision(self, q):
        return [self._names(i) for i in self._collisions(q)]
