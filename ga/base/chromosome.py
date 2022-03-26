from typing import Any, Union, Tuple
from abc import ABC, abstractmethod


class Chromosome(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self,
                    key: Union[str, int]) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def phenotype(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __lt__(self, other) -> bool:
        raise NotImplementedError()
