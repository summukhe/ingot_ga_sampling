import numpy as np
from typing import Tuple
from copy import deepcopy
from ga.base.operations import GAOps
from ga.base.chromosome import Chromosome
from ga.impl.chromosome_impl import IngotState


class IngotStateUpdate(GAOps):
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

    def xover(self,
              c1: Chromosome,
              c2: Chromosome,
              **kwargs) -> Tuple[Chromosome, Chromosome]:
        assert isinstance(c1, IngotState) and isinstance(c2, IngotState)
        sel = int(np.random.random() / 0.3)
        child1, child2 = deepcopy(c1), deepcopy(c2)
        if sel == 0:
            child1._scale = c2.scale
            child2._scale = c1.scale
        elif sel == 1:
            child1._base = tuple(c2.base)
            child2._base = tuple(c1.base)
        else:
            child1._z_axis = tuple(c2.z_axis)
            child2._z_axis = tuple(c1.z_axis)
        return child1, child2

    def mutate(self,
               c1: Chromosome,
               **kwargs) -> Chromosome:
        sel = int(np.random.random() / 0.3)
        c2 = deepcopy(c1)
        if sel == 0:
            c2._scale = np.random.random() * (self._scale_range[1] - self._scale_range[0]) + self._scale_range[0]
        elif sel == 1:
            dx = [min(c1.base[i] - self._min_base_plane[i],
                      self._max_base_plane[i] - c1.base[i]) for i in range(3)]
            ds = [np.random.normal(0, dx[i]) for i in range(3)]
            du = [np.clip(x, a_min=self._min_base_plane[i], a_max=self._max_base_plane[i])
                  for i, x in enumerate(tuple(np.array(c2.base) + np.array(ds)))]
            c2._base = tuple(du)
        else:
            theta = np.random.random() * (self._max_axis_angle[0] -
                                          self._min_axis_angle[0]) + \
                    self._min_axis_angle[0]
            phi = np.random.random() * (self._max_axis_angle[1] -
                                        self._min_axis_angle[1]) + \
                  self._min_axis_angle[1]

            z = np.cos(theta)
            y = np.sin(theta) * np.sin(phi)
            x = np.sin(theta) * np.cos(phi)
            c2._z_axis = tuple([x, y, z])
        return c2
