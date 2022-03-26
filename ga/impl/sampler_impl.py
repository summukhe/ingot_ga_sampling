import numpy as np
from typing import Tuple, List
from ga.base.sampler import ChromosomeSampler
from scipy.stats import qmc
from ga.base.chromosome import Chromosome
from ga.impl.chromosome_impl import IngotState


class IngotStateSampler(ChromosomeSampler):
    def __init__(self,
                 scale_range: Tuple[float, float],
                 min_axis_angle: Tuple[float, float],
                 max_axis_angle: Tuple[float, float],
                 min_base_plane: Tuple[float, float, float],
                 max_base_plane: Tuple[float, float, float]):

        assert (len(scale_range) == 2) and (scale_range[0] < scale_range[1])
        assert (len(min_base_plane) == len(max_base_plane)) and (len(min_base_plane) == 3)
        assert all([min_base_plane[i] < max_base_plane[i] for i in range(len(min_base_plane))])

        self._scale_range = scale_range
        self._min_base_plane = min_base_plane
        self._max_base_plane = max_base_plane
        self._min_axis_angle = min_axis_angle
        self._max_axis_angle = max_axis_angle

    def sample(self,
               n: int) -> List[Chromosome]:
        ub, lb = [], []
        lb.append(self._scale_range[0])
        ub.append(self._scale_range[1])

        for i in range(3):
            lb.append(self._min_base_plane[i])
            ub.append(self._max_base_plane[i])

        for i in range(2):
            lb.append(self._min_axis_angle[i])
            ub.append(self._max_axis_angle[i])

        d = len(ub)
        s = qmc.scale(qmc.Halton(d=d).random(n), lb, ub)
        pop = []
        for i in range(n):
            theta = s[i, 4]
            phi = s[i, 5]
            z = np.cos(theta)
            y = np.sin(theta) * np.sin(phi)
            x = np.sin(theta) * np.cos(phi)
            pop.append(IngotState(scale=s[i, 0],
                                  base=tuple(s[i, 1:4]),
                                  z_axis=(x, y, z)))
        return pop


if __name__ == "__main__":
    s = IngotStateSampler(scale_range=(0.2, 1.2),
                          min_base_plane=(0, 0, 0),
                          max_base_plane=(2, 2, 2),
                          min_axis_angle=(-np.pi/2, -np.pi/2),
                          max_axis_angle=(np.pi/2, np.pi/2))
    print(s.sample(4))

