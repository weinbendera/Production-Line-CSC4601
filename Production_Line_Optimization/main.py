from backend.models import ProductRequest
from backend.job_builder import JobBuilder
from backend.factory_logic_loader import FactoryLogicLoader
from backend.factory import Factory
from backend.schedulers.greedy_scheduler import GreedyScheduler
import json


def main():

    steps = 1152
    end_of_day_steps = [192, 384, 576, 768, 960]

    with open("data/Input_JSON_Schedule_Optimization.json") as f:
        data = json.load(f) # frontend will change this, product_requests and steps specifically

    product_requests_data = data["product_requests"] # frontend will change this, product_requests and steps specifically
    product_requests = [ProductRequest(**pr) for pr in product_requests_data]

    factory_logic = FactoryLogicLoader.load_from_file(filepath="data/Input_JSON_Schedule_Optimization.json")
    job_builder = JobBuilder(factory_logic=factory_logic)
    jobs = job_builder.build_jobs(product_requests=product_requests) # job objects
    factory = Factory(factory_logic=factory_logic)
    factory.add_jobs(jobs=jobs)

    # choose the scheduler
    scheduler = GreedyScheduler(greedy_type="min_power")

    state = factory.get_factory_state()

    total_power_cost = 0
    while not factory.done() and factory.current_step < steps:
        chosen_actions = scheduler.choose(state)
        factory.apply_actions(chosen_actions)
        total_power_cost += factory.step()
        state = factory.get_factory_state()

    scheduled_actions = scheduler.scheduled_actions
    print(f"Scheduled actions: {scheduled_actions}")

    # count the number of jobs in the factory that are not done
    print(f"Number of jobs not done: {len([job for job in factory.jobs if not job.done])}")
    print(f"Number of jobs done: {len([job for job in factory.jobs if job.done])}")

if __name__ == "__main__":
    main()