from typing import Dict
import json
from typing import List
from factory.factory_schemas import TaskMode, Task, Product, Machine, EnergySource, Operation, Job

class FactoryLogic:
    """
    Container for all factory logic loaded from configuration.
    Provides convenient access to factory data structures.

    FactoryLogic: "What's physically/logically possible?"
    Factory: "What's possible given the current situation?"
    This is a stateless class that should not be modified by the factory.
    """
    
    def __init__(self, 
                 task_modes: List[TaskMode],
                 tasks: List[Task],
                 products: List[Product],
                 machines: List[Machine],
                 energy_sources: List[EnergySource]):
        
        # Store as dictionaries for fast lookup
        self.task_modes: Dict[str, TaskMode] = {tm.id: tm for tm in task_modes} 
        self.tasks: Dict[str, Task] = {t.id: t for t in tasks}
        self.products: Dict[str, Product] = {p.id: p for p in products}
        self.machines: Dict[str, Machine] = {m.id: m for m in machines}
        self.energy_sources: Dict[str, EnergySource] = {es.id: es for es in energy_sources}
    
    def get_task_mode(self, task_mode_id: str) -> TaskMode:
        """Get task mode by ID, raises KeyError if not found"""
        return self.task_modes[task_mode_id]
    
    def get_task(self, task_id: str) -> Task:
        """Get task by ID, raises KeyError if not found"""
        return self.tasks[task_id]
    
    def get_product(self, product_id: str) -> Product:
        """Get product by ID, raises KeyError if not found"""
        return self.products[product_id]
    
    def get_machine(self, machine_id: str) -> Machine:
        """Get machine by ID, raises KeyError if not found"""
        return self.machines[machine_id]
    
    def get_energy_source(self, energy_source_id: str) -> EnergySource:
        """Get energy source by ID, raises KeyError if not found"""
        return self.energy_sources[energy_source_id]
    
    def get_solar_power_available(self, step: int) -> float:
        """Get solar power available at given time step"""
        solar = self.energy_sources.get("Solar")
        if solar and solar.availability:
            return solar.availability[step] if step < len(solar.availability) else 0.0
        return 0.0
    
    def get_grid_power_cost(self, step: int) -> float:
        """Get grid power cost at given time step"""
        grid = self.energy_sources.get("Socket Energy")
        if grid and grid.price:
            return grid.price[step] if step < len(grid.price) else 0.0
        return 0.0
    
    def validate_task_mode_for_operation(self, machine_id: str, operation: Operation, task_mode_id: str) -> bool:
        """
        Validate that a task mode can be used for an operation on a machine.
        Returns True if valid, False otherwise.
        """
        # Check if task mode exists
        if task_mode_id not in self.task_modes:
            return False
        
        # Check if task exists
        if operation.task_id not in self.tasks:
            return False
        
        # Check if machine exists
        if machine_id not in self.machines:
            return False
        
        # Check if task mode is valid for the task
        task = self.tasks[operation.task_id]
        if task_mode_id not in task.task_modes:
            return False
        
        # Check if task mode is valid for the machine
        machine = self.machines[machine_id]
        if task_mode_id not in machine.task_modes:
            return False
        
        return True
    
    def get_feasible_task_modes(self, machine_id: str, operation: Operation) -> List[str]:
        """
        Get all feasible task modes for an operation on a machine.
        Returns empty list if no feasible modes.
        """
        if machine_id not in self.machines or operation.task_id not in self.tasks:
            return []
        
        machine_modes = set(self.machines[machine_id].task_modes)
        task_modes = set(self.tasks[operation.task_id].task_modes)
        
        return list(machine_modes.intersection(task_modes))


# ============================================
# FACTORY LOGIC LOADER
# ============================================

class FactoryLogicLoader:
    """
    Loads factory logic from JSON configuration file.
    Validates and converts to FactoryLogic object.
    """
    
    @staticmethod
    def load_from_file(filepath: str) -> FactoryLogic:
        """Load factory logic from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return FactoryLogicLoader.load_from_dict(data)
    
    @staticmethod
    def load_from_dict(data: Dict) -> FactoryLogic:
        """Load factory logic from dictionary"""
        # Validate required keys
        required_keys = ["task_modes", "tasks", "products", "machines", "energy_sources"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in factory logic")
        
        # Convert to Pydantic models
        task_modes = [TaskMode(**tm) for tm in data["task_modes"]]
        tasks = [Task(**t) for t in data["tasks"]]
        products = [Product(**p) for p in data["products"]]
        machines = [Machine(**m) for m in data["machines"]]
        energy_sources = [EnergySource(**es) for es in data["energy_sources"]]
        
        return FactoryLogic(
            task_modes=task_modes,
            tasks=tasks,
            products=products,
            machines=machines,
            energy_sources=energy_sources
        )