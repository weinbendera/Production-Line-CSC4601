from typing import List, Optional
from factory.factory_logic_loader import FactoryLogic
from factory.factory_schemas import ProductRequest, Job, Operation

class JobBuilder:
    """
    Takes in ProductRequests and builds a list of Jobs.
    """
    def __init__(self, factory_logic: FactoryLogic):
        self.factory_logic = factory_logic

    def build_jobs(self, product_requests: List[ProductRequest]) -> List[Job]:
        """
        Builds a list of jobs from the user's product requests.
        """
        jobs = []
        for product_request in product_requests: # for each type of product requested
            if product_request.product not in self.factory_logic.products:
                available_products = list(self.factory_logic.products.keys())
                raise ValueError(
                    f"Unknown product '{product_request.product}' in ProductRequest. "
                    f"Available products: {available_products}"
                )

            for index in range(1, product_request.amount + 1): # for each instance of the product requested
                job_id = Job.make_id(product_request.product, deadline=product_request.deadline, index=index)
                job = Job(
                    id=job_id,
                    product_id=product_request.product, # type of product requested
                    operations=self.build_job_operations(product_request.product, job_id, product_request.deadline),
                    deadline=product_request.deadline,
                )
                jobs.append(job)

        return jobs

    def build_job_operations(self, product: str, job_id: str, deadline: Optional[int]) -> List[Operation]:
        """
        Builds a dict that declares which tasks need to be done for the product to be completed.
        """
        product = self.factory_logic.get_product(product)
        operations = []

        for product_task in product.tasks: # for each task needed for the product to be completed
            if product_task.task not in self.factory_logic.tasks:
                raise ValueError(f"Task '{product_task.task}' not found for product '{product}' in the factory logic")
            
            for run_index in range(1, product_task.runs + 1): # for each run of the task
                operations.append(Operation(
                    id=Operation.make_id(job_id, product_task.task, run_index),
                    task_id=product_task.task,
                    run_index=run_index,
                    deadline=deadline,
                    started=False,
                    done=False
                ))

        return operations