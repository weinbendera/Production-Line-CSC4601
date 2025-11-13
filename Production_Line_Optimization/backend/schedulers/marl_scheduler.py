from backend.schedulers.Scheduler import Scheduler
from backend.schemas.Job import Job
from typing import List


class MARLScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def schedule(self, jobs: List[Job]) -> List[Job]:
        pass
