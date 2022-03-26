import numpy as np
from ga.impl.ingot import Ingot
from typing import Any, Tuple, Union, List
from ga.base.chromosome import Chromosome


class IngotState(Chromosome):
    def __init__(self,
                 scale: float,
                 base: Tuple[float, float, float],
                 z_axis: Tuple[float, float, float],
                 **kwargs,
                 ):
        super(IngotState, self).__init__()
        self._scale = scale
        self._base = base
        self._z_axis = z_axis
        self._kwargs = kwargs

    def __len__(self) -> int:
        return 3

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def base(self) -> Tuple[float, float, float]:
        return self._base

    @property
    def z_axis(self) -> Tuple[float, float, float]:
        return self._z_axis

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            if key == 0:
                return self._scale
            if key == 1:
                return self._base
            if key == 2:
                return self._z_axis
        else:
            if key == 'scale':
                return self._scale
            if key == 'base':
                return self._base
            if key == 'z_axis':
                return self._z_axis
        return ValueError(f"Error: invalid key [{key}]")

    def phenotype(self) -> Any:
        return Ingot(scale=self._scale,
                     z_axis=self._z_axis,
                     base_center=self._base,
                     **self._kwargs)

    def tolist(self) -> List[float]:
        return [self._scale] + list(self._base) + list(self._z_axis)

    def __eq__(self, other) -> bool:
        if isinstance(other, IngotState):
            return np.mean(np.square(np.array(self.tolist()) - np.array(other.tolist()))) < 1e-3
        return False

    def __lt__(self, other) -> bool:
        if not isinstance(other, IngotState):
            raise ValueError("Error: instance does not match!")
        return self._scale < other._scale

    def __str__(self):
        return f'Ingot(scale={self.scale}, ' \
               f'base=({self.base[0]:.1f}, {self.base[1]:.1f}, {self.base[2]:.1f}), ' \
               f'axis=({self.z_axis[0]:.1f}, {self.z_axis[1]:.1f}, {self.z_axis[2]:.1f}))'

    def __repr__(self):
        return self.__str__()

