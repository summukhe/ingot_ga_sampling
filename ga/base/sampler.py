from ga.base.chromosome import Chromosome
from abc import ABC, abstractmethod
from typing import List


class ChromosomeSampler(ABC):
    @abstractmethod
    def sample(self,
               n: int) -> List[Chromosome]:
        raise NotImplementedError()

