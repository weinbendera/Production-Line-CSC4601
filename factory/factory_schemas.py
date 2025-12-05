from pydantic import BaseModel
from typing import List, Optional

"""
PYDANTIC MODELS OF THE DATA (FACTORY LOGIC)
"""

class TaskMode(BaseModel):
    id: str
    power: List[float] # list of power values for the task mode for each step
 
class Task(BaseModel):
    id: str
    task_modes: List[str]

class ProductTask(BaseModel):
    """
    Each Product has multiple tasks that need to be performed.
    They may also have mutliple runs of the same task.
    This is a schema for the tasks of a product.
    """
    task: str
    runs: int

class Product(BaseModel):
    """
    A Product has a list of tasks that need to be done to complete the product.
    """
    id: str
    tasks: List[ProductTask]

class EnergySource(BaseModel):
    """
    Used to map each energy source id to its price and availability.
    Solar has the availability for the factory at that time step.
    Socket Energy has the price per kwh of the grid power for the factory at that time step.
    """
    id: str
    price: Optional[List[float]] = None
    availability: Optional[List[float]] = None

class Machine(BaseModel):
    """
    Used to map each machine id to its task modes.
    """
    id: str
    task_modes: List[str]

"""
MODELS FOR THE USER'S INPUT
"""

class ProductRequest(BaseModel):
    product: str
    amount: int
    deadline: Optional[int] = None

"""
MODELS FOR THE JOB BUILDER
"""

class Operation(BaseModel):
    """
    Pydantic model for an operation.
    This class represents an operation in the production line.
    An operation is a task's run that needs to be done for the product to be completed.
    """
    id: str 
    task_id: str # id of the task that the operation is a run of
    run_index: int # index of the run of the task
    deadline: Optional[int] = None # deadline for the job to be completed
    started: bool = False # whether the operation has been started on a machine
    done: bool = False # whether the operation has been completed

    @staticmethod
    def make_id(job_id: str, task_id: str, run_index: int) -> str:
        # Example: J_TSHIRT#0001#CUT#0001
        return f"{job_id}#{task_id.upper()}#{run_index:04d}"


class Job(BaseModel):
    """
    Pydantic model for a job.
    This class represents a job in the production line.
    A job is a request for a product to be produced.
    A job has a list of operations which is the Tasks that need to be done for the product to be completed.
    """
    id: str
    product_id: str
    operations: List[Operation] # list of tasks needed to complete the product
    being_processed: bool = False
    deadline: Optional[int] = None
    done: bool = False

    @staticmethod
    def make_id(product_id: str, deadline: Optional[int], index: int) -> str:
        # Example: TSHIRT#1000#0001
        return f"{product_id}#S={deadline}#I={index:04d}"

    def check_completion(self) -> bool:
        """
        Check if all operations in this job are complete.
        If so, mark the job as done and return True.
        Returns True if job is/was complete, False otherwise.
        """
        if self.done:
            return True  # Already marked as complete
        
        if all(operation.done for operation in self.operations):
            self.done = True
            return True
        
        return False

"""
MODELS FOR THE ACTION SPACE
"""

class FeasibleAction(BaseModel):
    """
    A feasible operation is an operation that is executable by a machine.
    It includes the job_id, operation_id, and the task_mode_id that is feasible for the operation.
    """
    machine_id: str
    job_id: str
    operation_id: str
    task_mode_id: str

class Action(BaseModel):
    """
    The chosen action to be taken by the scheduler's machine/agent.
    """
    machine_id: str
    job_id: str
    operation_id: str
    task_mode_id: str

    # used for Gantt charts
    start_step: int
    end_step: int