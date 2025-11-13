from backend.schedulers.scheduler import OfflineScheduler
from backend.factory_state import FactoryState
from typing import List


class GeneticScheduler(OfflineScheduler):
    def __init__(self, population_size: int = 100, generations: int = 50, mutation_rate: float = 0.01):
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def schedule(self, factory_state: FactoryState) -> List[Action]:
        """
        Schedules the jobs for the factory.
        """
        return []

    def _evaluate_fitness(self, factory_state: FactoryState) -> float:
        """
        Evaluates the fitness of the factory state.
        """
        return 0

    def _select_parents(self, factory_state: FactoryState) -> List[FactoryState]:
        """
        Selects the parents for the next generation.
        """
        return []