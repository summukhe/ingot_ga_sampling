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
                 allowed_inaccuracies: float = 0.05,
                 ):
        self._cylinder_diameter = top_diameter
        self._base_diameter = base_diameter
        self._cone_height = cone_height
        self._cylinder_height = cylinder_height
        self._base_center = np.array(base_center, dtype='float32')
        self._z_axis = unit_vector(*z_axis)
        self._scale = scale
        self._allowed_error = allowed_error
        self._allowed_inaccuracy = allowed_inaccuracies

    @property
    def top_diameter(self) -> float:
        return self._cylinder_diameter

    @property
    def cylinder_height(self) -> float:
        return self._cylinder_height

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

    @property
    def scale(self) -> float:
        return self._scale

    def _likelihood(self,
                    x: List[Tuple[float, float, float]]) -> float:
        x = np.array(x).astype('float32') - self._base_center

        s = self._scale
        h = self._cylinder_height + self._cone_height
        err = s * self._allowed_error
        p = np.dot(x, self._z_axis)
        idx = np.where((p > -err) & (p < s * h + err))[0]
        if len(idx) == 0:
            return 0.
        p = p[idx]
        px = p[..., np.newaxis] * self._z_axis[np.newaxis, ...]
        d = np.sqrt(np.sum((x[idx] - px)**2, axis=1))
        base_idx = np.where((np.abs(p) < err) &
                            (d < s * self._base_diameter / 2 + err))[0]

        top_idx = np.where((np.abs(p - s * h) < err) &
                           (d < s * self._cylinder_diameter / 2 + err))[0]
        cyl_surf = np.where((p > s * self._cone_height) &
                            (p < s * h) &
                            (np.abs(d - s * self._cylinder_diameter / 2) < err))[0]
        cone_idx = np.where(p < s * self._cone_height)[0]

        cyl_intersection = np.where((p > s * self._cone_height) & (p < s * h) &
                                    (d < (s * self._cylinder_diameter / 2) - err))[0]

        dh = s * (self._base_diameter + (p[cone_idx] / self._cone_height) * (self._cylinder_diameter -
                                                                             self._base_diameter))
        cone_surf = np.where(np.abs(d[cone_idx] - dh) < err)[0]
        if len(cone_surf) > 0:
            cone_surf = cone_idx[cone_surf]
        surf_points = set(base_idx).union(top_idx).union(cyl_surf).union(cone_surf)
        remaining = set(cyl_intersection).difference(surf_points)
        if len(surf_points) == 0:
            return 0
        if len(remaining)/len(surf_points) > self._allowed_inaccuracy:
            return 0
        return (len(surf_points) - len(remaining))/x.shape[0]

    def likelihood(self,
                   points: List[Tuple[float, float, float]],
                   thresh: float = 0.1):
        return self._likelihood(np.array(points))


