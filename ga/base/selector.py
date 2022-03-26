import numpy as np
from typing import Tuple, List
from abc import ABC, abstractmethod


class Selector(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def set_fitness(self,
                    values: List[float]):
        raise NotImplementedError()

    @abstractmethod
    def select_pair(self,
                    n: int) -> List[Tuple[int, int]]:
        raise NotImplementedError()

    @abstractmethod
    def select_random(self,
                      n: int) -> List[int]:
        raise NotImplementedError()


class RouletteSelector(Selector):
    def __init__(self,
                 max_partitions: int = 10,
                 ):
        self._max_partitions = max_partitions
        self._fitness_values = []

    @property
    def length(self) -> int:
        return len(self._fitness_values)

    def __len__(self) -> int:
        return self.length

    def set_fitness(self,
                    values: List[float]):
        self._fitness_values = np.array(values)
        self._fitness_values = self._fitness_values - np.min(self._fitness_values)
        self._fitness_values = self._fitness_values / np.sum(self._fitness_values)

    def is_fitted(self) -> bool:
        return self.length > 0

    @staticmethod
    def recurse_select(data: List[float],
                       n_select: int,
                       max_partition: int,
                       ) -> List[int]:
        data = np.array(data)
        n = len(data)
        groups = {}
        if n > max_partition:
            avg_repeat = n // max_partition
            assignment = np.repeat(list(range(max_partition)), avg_repeat)
            np.random.shuffle(assignment)
            groups = {i: [ ] for i in range(max_partition)}
            for i, c in enumerate(assignment):
                groups[c].append(i)
            n_assigned = len(assignment)
            assignment = np.random.choice(max_partition, n - n_assigned)
            for i, c in enumerate(assignment):
                groups[c].append(i + n_assigned)
        else:
            groups = {i: [i] for i, x in enumerate(data)}
        values = {g: np.sum(data[groups[g]]) for g in groups}
        group_ids = sorted(list(groups.keys()))
        p = np.array([0.0] + [values[g] for g in group_ids])
        p = np.cumsum(p / np.sum(p))

        p_select = np.random.random(size=n_select)
        selected_index = []
        for pv in p_select:
            s = np.where(p < pv)[0][-1]
            if len(groups[s]) > 1:
                v = RouletteSelector.recurse_select(data[groups[s]], 1, max_partition)
                s = [groups[s][i] for i in v]
            else:
                s = groups[s]
            selected_index = selected_index + s
        return selected_index

    def select_pair(self,
                    n: int = 1) -> List[Tuple[int, int]]:
        if not self.is_fitted():
            raise RuntimeError(f"Error: no fitness value set for selection!")

        result = []
        while len(result) < n:
            selected = RouletteSelector.recurse_select(self._fitness_values, 2*n, self._max_partitions)
            np.random.shuffle(selected)
            selected = [(selected[2*i], selected[2*i+1]) for i in range(n) if selected[2*i] != selected[2*i+1]]
            result = result + selected
        return result

    def select_random(self,
                      n: int) -> List[int]:
        fv = np.max(self._fitness_values) - self._fitness_values
        return RouletteSelector.recurse_select(fv, n, self._max_partitions)


if __name__ == "__main__":
    x = np.random.randint(0, 50, 32)
    rs = RouletteSelector()
    print({i: xx for i, xx in enumerate(x)})
    rs.set_fitness(x)
    print(rs.select_pair(5))
