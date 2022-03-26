from abc import ABC, abstractmethod
from typing import Tuple

from ga.base.chromosome import Chromosome


class GAOps(ABC):

    @abstractmethod
    def xover(self,
              c1: Chromosome,
              c2: Chromosome,
              **kwargs,
              ) -> Tuple[Chromosome, Chromosome]:
        raise NotImplementedError("Error: not implemented in base class!")

    @abstractmethod
    def mutate(self,
               c1: Chromosome,
               **kwargs) -> Chromosome:
        raise NotImplementedError("Error: not implemented in base class!")



