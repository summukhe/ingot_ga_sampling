import scipy
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from .ingot import Ingot


class IngotStatePlotter:
    def __init__(self,
                 point_cloud: np.ndarray,
                 ingot: Ingot = None,
                 sampling_rate: int = 50,
                 pc_color: str = '#ffa64d',
                 cyl_color: str = '#751aff',
                 ):
        if not isinstance(point_cloud, np.ndarray) or \
                (len(point_cloud.shape) != 2) or \
                point_cloud.shape[-1] != 3:
            raise ValueError(f"Error: expects 3d point cloud data!")
        self._xyz = point_cloud
        self._ingot = ingot
        self._sampling_rate = sampling_rate
        self._pc_color = pc_color
        self._cyl_color = cyl_color

    def set_ingot_state(self, ingot: Ingot):
        self._ingot = ingot
        return self

    def is_set(self) -> bool:
        return (self._ingot is not None) and isinstance(self._ingot, Ingot)

    def cylinder_xyz(self):
        mag = scipy.linalg.norm(self._ingot.z_axis)
        if mag < 1e-5:
            raise ValueError("Error: too small magnitude of axis vector!")
        v = self._ingot.z_axis.copy()
        v = v / mag
        not_v = v + np.random.random(v.shape)
        n1 = np.cross(v, not_v)
        n1 /= scipy.linalg.norm(n1)
        n2 = np.cross(v, n1)

        b = self._ingot.base_center
        r = self._ingot.top_diameter / 2
        s = self._ingot.scale
        h = self._ingot.cylinder_height

        t  = np.linspace(0,
                         h,
                         self._sampling_rate) * s
        theta = np.linspace(0, 2 * np.pi, self._sampling_rate)
        r_sample = [0, r]
        t, theta2 = np.meshgrid(t, theta)
        _, theta = np.meshgrid(r_sample, theta)
        x, y, z = [b[i] + v[i]*t +
                   r * s * np.sin(theta2) * n1[i] + r * s * np.cos(theta2) * n2[i]
                   for i in range(3)]
        return x, y, z

    def compute_point_clouds(self):
        if not self.is_set():
            raise RuntimeError(f"Error: the system is not ready for plotting!")
        data = dict()
        data['pc'] = (self._xyz[:, 0], self._xyz[:, 1], self._xyz[:, 2])
        data['ingot'] = self.cylinder_xyz()
        return data

    def plot(self, fig):
        data = self.compute_point_clouds()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = data['pc']
        ax.scatter(x, y, z, s=0.2, color=self._pc_color)
        x, y, z = data['ingot']
        ax.scatter(x, y, z, s=0.1, color=self._cyl_color)



