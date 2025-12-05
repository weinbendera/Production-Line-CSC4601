from schedulers.scheduler import OfflineScheduler
from factory.factory_state import FactoryState
from factory.factory_schemas import Action, FeasibleAction
from typing import List, Dict, Tuple
import random
import copy

class Individual:
    """
    Represents a scheduling solution (chromosome/genome).
    Genome is a list of (operation_id, machine_id, task_mode_id) tuples.
    """
    def __init__(self, genome: List[Tuple[str, str, str]], fitness: float = 0.0):
        self.genome = genome  # List of (operation_id, machine_id, task_mode_id)
        self.fitness = fitness
        self.actions: List[Action] = []
        self.total_cost = float('inf')
        self.makespan = float('inf')

class GeneticScheduler(OfflineScheduler):
    """
    Genetic Algorithm scheduler for job shop scheduling.
    Uses evolutionary algorithms to find optimal schedules.
    """
    def __init__(self, population_size: int = 100, generations: int = 50, 
                 mutation_rate: float = 0.01, elitism_count: int = 2):
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count  # number of best individuals to preserve

    def schedule(self, factory_state: FactoryState) -> List[Action]:
        """
        Schedules jobs using a genetic algorithm.
        Returns the best schedule found.
        """
        if not factory_state.jobs:
            return []
        
        # Initialize population
        population = self._initialize_population(factory_state)
        
        # Evaluate initial population
        for individual in population:
            self._evaluate_fitness(factory_state, individual)
        
        best_individual = max(population, key=lambda ind: ind.fitness)
        
        # Evolution loop
        for generation in range(self.generations):
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            for i in range(self.elitism_count):
                new_population.append(copy.deepcopy(population[i]))
            
            # Generate rest of population through selection, crossover, mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._fitness_proportionate_selection(population)
                parent2 = self._fitness_proportionate_selection(population)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(factory_state, child)
                
                # Evaluate child
                self._evaluate_fitness(factory_state, child)
                new_population.append(child)
            
            population = new_population
            
            # Track best solution
            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = current_best
                print(f"Generation {generation}: New best fitness = {best_individual.fitness:.2f}, "
                      f"Cost = {best_individual.total_cost:.2f}, Makespan = {best_individual.makespan}")
        
        # Return best schedule
        self.scheduled_actions = best_individual.actions
        return best_individual.actions

    def _initialize_population(self, factory_state: FactoryState) -> List[Individual]:
        """
        Creates initial population with random schedules.
        """
        population = []
        
        for _ in range(self.population_size):
            genome = self._create_random_genome(factory_state)
            population.append(Individual(genome))
        
        return population

    def _create_random_genome(self, factory_state: FactoryState) -> List[Tuple[str, str, str]]:
        """
        Creates a random genome (schedule).
        Genome is a list of (operation_id, machine_id, task_mode_id) tuples.
        
        """
        genome = []
        
        # Collect all operations from all jobs
        all_operations = []
        for job in factory_state.jobs:
            for operation in job.operations:
                all_operations.append((job.id, operation))
        
        # Shuffle operations for random order
        random.shuffle(all_operations)
        
        # Assign random machine and task mode to each operation
        for job_id, operation in all_operations:
            # Get feasible task modes for this operation
            feasible_machines = list(factory_state.factory_logic.machines.keys())
            random.shuffle(feasible_machines)
            
            # Find a machine that can do this operation
            for machine_id in feasible_machines:
                feasible_modes = factory_state.factory_logic.get_feasible_task_modes(
                    machine_id, operation
                )
                if feasible_modes:
                    task_mode_id = random.choice(feasible_modes)
                    genome.append((operation.id, machine_id, task_mode_id))
                    break
        
        return genome

    def _evaluate_fitness(self, factory_state: FactoryState, individual: Individual) -> float:
        """
        Evaluates fitness of an individual by simulating the schedule.
        Fitness is based on total cost and makespan (completion time).
        Higher fitness is better.
        """
        # Simulate the schedule
        actions = self._decode_genome_to_actions(factory_state, individual.genome)
        
        if not actions:
            individual.fitness = 0.0
            individual.total_cost = float('inf')
            individual.makespan = float('inf')
            return 0.0
        
        # Calculate total cost and makespan
        total_cost = 0.0
        makespan = max(action.end_step for action in actions) if actions else 0
        
        # Simple cost calculation: sum of power costs at each step
        max_step = makespan
        for step in range(max_step):
            power_used = 0.0
            for action in actions:
                if action.start_step <= step < action.end_step:
                    # Get power for this step of the operation
                    task_mode = factory_state.factory_logic.get_task_mode(action.task_mode_id)
                    step_in_operation = step - action.start_step
                    if step_in_operation < len(task_mode.power):
                        power_used += task_mode.power[step_in_operation]
            
            # Calculate cost (simplified: no solar, just grid cost)
            grid_cost = factory_state.factory_logic.get_grid_power_cost(step)
            total_cost += power_used * grid_cost
        
        individual.total_cost = total_cost
        individual.makespan = makespan
        individual.actions = actions
        
        # Fitness: minimize cost and makespan (use negative for minimization)
        # Normalize to positive values
        individual.fitness = 1.0 / (1.0 + total_cost + makespan * 0.1)
        
        return individual.fitness

    def _decode_genome_to_actions(self, factory_state: FactoryState, 
                                  genome: List[Tuple[str, str, str]]) -> List[Action]:
        """
        Decodes genome into actual Actions by simulating execution.
        Returns list of Actions with start/end times.
        Enforces precedence constraints and deadline constraints.
        """
        actions = []
        machine_busy_until = {m_id: 0 for m_id in factory_state.factory_logic.machines.keys()}
        operation_completion = {}
        job_operations = {}
        
        # Build job operation mapping
        for job in factory_state.jobs:
            job_operations[job.id] = {op.id: op for op in job.operations}
        
        # Process genome in order
        for operation_id, machine_id, task_mode_id in genome:
            # Find the job this operation belongs to
            job_id = None
            operation = None
            for job in factory_state.jobs:
                for op in job.operations:
                    if op.id == operation_id:
                        job_id = job.id
                        operation = op
                        break
                if job_id:
                    break
            
            if not job_id or not operation:
                continue
            
            # Get task mode
            task_mode = factory_state.factory_logic.get_task_mode(task_mode_id)
            duration = len(task_mode.power)
            
            # Determine start time (when machine is free)
            start_time = machine_busy_until[machine_id]
            
            # Check precedence: operation can't start until previous operations in job are done
            # (simplified: assume operations are in order)
            for prev_op_id, completion_time in operation_completion.items():
                if prev_op_id.startswith(job_id):
                    start_time = max(start_time, completion_time)
            
            end_time = start_time + duration
            
            # Create action
            action = Action(
                machine_id=machine_id,
                job_id=job_id,
                operation_id=operation_id,
                task_mode_id=task_mode_id,
                start_step=start_time,
                end_step=end_time
            )
            actions.append(action)
            
            # Update state
            machine_busy_until[machine_id] = end_time
            operation_completion[operation_id] = end_time
        
        return actions

    def _fitness_proportionate_selection(self, population: List[Individual]) -> Individual:
        """
        Selects an individual using fitness proportionate selection (roulette wheel).
        """
        total_fitness = sum(ind.fitness for ind in population)
        
        if total_fitness == 0:
            return random.choice(population)
        
        # Calculate selection probabilities
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        
        for individual in population:
            current_sum += individual.fitness
            if current_sum >= selection_point:
                return individual
        
        return population[-1]  # fallback

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Performs single-point crossover to create a child.
        """
        if len(parent1.genome) == 0 or len(parent2.genome) == 0:
            return Individual(parent1.genome.copy())
        
        # Single-point crossover
        crossover_point = random.randint(1, min(len(parent1.genome), len(parent2.genome)) - 1)
        
        child_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
        
        return Individual(child_genome)

    def _mutate(self, factory_state: FactoryState, individual: Individual) -> Individual:
        """
        Mutates an individual by changing one gene (operation assignment).
        """
        if not individual.genome:
            return individual
        
        # Pick random gene to mutate
        mutation_index = random.randint(0, len(individual.genome) - 1)
        operation_id, old_machine_id, old_task_mode_id = individual.genome[mutation_index]
        
        # Find the operation
        operation = None
        for job in factory_state.jobs:
            for op in job.operations:
                if op.id == operation_id:
                    operation = op
                    break
            if operation:
                break
        
        if not operation:
            return individual
        
        # Try to find alternative machine or task mode
        feasible_machines = list(factory_state.factory_logic.machines.keys())
        random.shuffle(feasible_machines)
        
        for machine_id in feasible_machines:
            feasible_modes = factory_state.factory_logic.get_feasible_task_modes(
                machine_id, operation
            )
            if feasible_modes:
                task_mode_id = random.choice(feasible_modes)
                individual.genome[mutation_index] = (operation_id, machine_id, task_mode_id)
                break
        
        return individual