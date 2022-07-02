import time
import numpy as np
from typing import Callable, List
from ga.base.chromosome import Chromosome
from ga.base.selector import Selector
from ga.base.sampler import ChromosomeSampler
from ga.base.operations import GAOps
from typing import Callable


class GASearch:
    def __init__(self,
                 fitness: Callable,
                 n_population: int,
                 n_iteration: int,
                 generator: ChromosomeSampler,
                 operator: GAOps,
                 selector: Selector,
                 p_mutation: float = 0.2,
                 n_elites: int = 2,
                 visitor: Callable = None,
                 ):
        self._fn = fitness
        self._n_pop = n_population
        self._n_iter = n_iteration
        self._initializer = generator
        self._operator = operator
        self._selector = selector
        self._p_mutation = p_mutation
        self._n_elites = n_elites
        self._visitor = visitor

    def search(self, **kwargs) -> Chromosome:
        p_mutation = kwargs.get('p_mutation', self._p_mutation)
        n_elites = kwargs.get('n_elites', self._n_elites)
        verbose = kwargs.get('verbose', False)

        pop = self._initializer.sample(self._n_pop)
        for ii in range(self._n_iter):
            start = time.time_ns()
            fv = np.array([self._fn(p) for p in pop])
            best_i = np.argmax(fv)
            if self._visitor is not None:
                self._visitor(pop[best_i])
            end = time.time_ns()
            if verbose:
                print("Function evaluation time : {:.3f}ms".format((end - start)/1e6))
                print("Max value: {:.5f}".format(np.max(fv)))
            self._selector.set_fitness(fv)
            p = np.random.random(size=self._n_pop - n_elites)
            nm = len(np.where(p < p_mutation)[0])
            nx = len(p) - nm
            new_pop = [pop[i] for i in np.argsort(fv)[-n_elites:]]
            if nm > 0:
                sm = self._selector.select_random(nm)
                new_pop = new_pop + [self._operator.mutate(pop[i]) for i in sm]
            if nx > 0:
                xm = self._selector.select_pair(nx)
                new_pop = new_pop + [self._operator.xover(pop[i],
                                                          pop[j])[np.random.choice([0, 1])]
                                     for i, j in xm]
            pop = new_pop
        fv = np.array([self._fn(p) for p in pop])
        return pop[np.argmax(fv)]

