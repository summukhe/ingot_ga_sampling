import numpy as np
from typing import Tuple, List


def unit_vector(x: float,
                y: float,
                z: float):
    n = max(np.sqrt(x**2 + y**2 + z**2), 1e-3)
    return np.array([x, y, z]).astype('float32') / n


class Ingot:
    def __init__(self,
                 top_diameter: float = 100,
                 base_diameter: float = 250,
                 cone_height: float = 165,
                 cylinder_height: float = 110,
                 base_center: Tuple[float, float, float] = (0, 0, 0),
                 z_axis: Tuple[float, float, float] = (0, 0, 1),
                 scale: float = 1.0,
                 allowed_error: float = 1.0,
                 ):
        self._cylinder_diameter = top_diameter
        self._base_diameter = base_diameter
        self._cone_height = cone_height
        self._cylinder_height = cylinder_height
        self._base_center = np.array(base_center, dtype='float32')
        self._z_axis = unit_vector(*z_axis)
        self._scale = scale
        self._allowed_error = allowed_error

    @property
    def top_diameter(self) -> float:
        return self._cylinder_diameter

    @property
    def base_diameter(self) -> float:
        return self._base_diameter

    @property
    def cone_height(self) -> float:
        return self._cone_height

    @property
    def base_center(self) -> Tuple[float, float, float]:
        return tuple(self._base_center)

    @property
    def z_axis(self) -> np.ndarray:
        return self._z_axis.copy()

    def _likelihood(self,
                    x: Tuple[float, float, float]) -> float:
        x = np.array(x).astype('float32') - self._base_center

        s = self._scale
        h = self._cylinder_height + self._cone_height
        err = s * self._allowed_error
        p = np.dot(x, self._z_axis)
        if (p < -err) or (p > s * h + err):
            return 0.0

        px = p * self._z_axis
        d = np.sqrt(np.sum((x - px)**2))

        if (np.abs(p) < err) and (d < s * self._base_diameter  + err):
            return (0.5/err) * np.exp(-(np.abs(p)/err))
        elif (np.abs(p - s * h) < err) and (d < s * self._cylinder_diameter  + err):
            return (0.5/err) * np.exp(-(np.abs(p)/err))
        else:
            if p > self._cone_height:
                dh = s * self._cylinder_diameter
            else:
                dh = s * (self._base_diameter +
                          (p / self._cone_height) * (self._cylinder_diameter -
                                                     self._base_diameter))
            return (0.5/err) * np.exp(-(np.abs(d-dh)/err)) if np.abs(d - dh) < err else 0

    def likelihood(self,
                   points: List[Tuple[float, float, float]],
                   thresh: float = 0.01):
        p = np.array([self._likelihood(x) for x in points])
        return len(np.where(p > thresh)[0])/len(p)


if __name__ == "__main__":
    import os
    import open3d as o3d

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, '../../data')
    data_file = os.path.join(data_dir, 'Cropped_point_cloud.pcd')
    pc = o3d.io.read_point_cloud(data_file)
    data = np.array(pc.points, dtype='float32')

    x_mx, y_mx, z_mx = np.max(data[:, 0]), np.max(data[:, 1]), np.max(data[:, 2])
    x_mn, y_mn, z_mn = np.min(data[:, 0]), np.min(data[:, 1]), np.min(data[:, 2])

    center = np.mean(data, axis=0)
    scale = min(x_mx - x_mn, y_mx, y_mn)/250
    ingot = Ingot(base_center=tuple(center),
                  scale=scale)

    print(ingot.likelihood(data))