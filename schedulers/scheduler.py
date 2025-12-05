from abc import ABC, abstractmethod 
from typing import List, Dict, Optional
from factory.factory_schemas import Action, Job
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io


class Scheduler(ABC):
    """
    Abstract base class (interface) for all schedulers.

    Schedulers are used to choose actions for the factory's machines.
    Scheduler chooses the action -> factory runs the action -> factory updates its state -> scheduler chooses the next action.
    """
    def __init__(self):
        self.scheduled_actions = [] # list of actions that have been scheduled

    def plot_gantt_by_product(self, factory, actions: List[Action]):
        """
        Draws a Gantt chart with:
        - y-axis: machines
        - x-axis: time steps
        - color: product_id (i.e., product_request type)
        - label on each bar: product_id (short)
        
        Returns a PIL Image of the Gantt chart.
        """
        
        records = []
        for a in actions:
            job = factory.get_job_by_id(a.job_id)
            records.append({
                "machine": a.machine_id,
                "job_id": a.job_id,
                "product_id": job.product_id,
                "operation_id": a.operation_id,
                "start": a.start_step,
                "finish": a.end_step,
                "duration": a.end_step - a.start_step,
            })

        if not records:
            print("No actions to plot.")
            return None

        df = pd.DataFrame(records)

        df = df.sort_values(by=["machine", "start"])

        product_ids = sorted(df["product_id"].unique())
        cmap = plt.get_cmap("tab20")
        color_map = {pid: cmap(i % 20) for i, pid in enumerate(product_ids)}

        machines = sorted(df["machine"].unique())
        y_positions = {m: i for i, m in enumerate(machines)}

        fig, ax = plt.subplots(figsize=(16, 8))

        for _, row in df.iterrows():
            y = y_positions[row["machine"]]
            color = color_map[row["product_id"]]

            ax.barh(
                y=y,
                width=row["duration"],
                left=row["start"],
                color=color,
                edgecolor="black",
                alpha=0.9,
            )

        ax.set_yticks([y_positions[m] for m in machines])
        ax.set_yticklabels(machines)
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Machine")
        ax.set_title("Gantt Chart – Colored by Product Request Type")

        ax.set_xlim(df["start"].min() - 1, df["finish"].max() + 1)

        legend_patches = [
            mpatches.Patch(color=color_map[pid], label=str(pid))
            for pid in product_ids
        ]
        ax.legend(
            handles=legend_patches,
            title="Product Request Types (product_id)",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        
        plt.close(fig)
        
        return img

class OnlineScheduler(Scheduler):
    """
    Abstract base class (interface) for all online schedulers.
    All concrete online scheduler implementations must implement the choose method.
    Online schedulers are used to choose the next actions for each ofthe factory's machines, one step at a time.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose(self, factory_state) -> Dict[str, Action | None]:
        """
        Chooses the next actions for each of the factory's machines.
        Returns the actions to be taken by the scheduler for each machine, or None if no actions were taken.
        """
        raise NotImplementedError("Subclasses must implement this method")

class OfflineScheduler(Scheduler):
    """
    Abstract base class (interface) for all offline schedulers.
    All concrete offline scheduler implementations must implement the schedule method.
    Offline schedulers are used to schedule all of the given jobs at once.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def schedule(self, factory_state) -> List[Action]:
        """
        Schedules the given jobs for all machines.
        Returns the scheduled actions taken by the scheduler.
        """
        raise NotImplementedError("Subclasses must implement this method")