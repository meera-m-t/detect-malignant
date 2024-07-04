from typing import  Optional
from pydantic import BaseModel, Field





class SchedulerConfig(BaseModel):
    start: Optional[float] = Field(default=0.1, description="Starting learning rate.")
    end: Optional[float] = Field(default=0.000001, description="Ending learning rate.")



class EarlyStoppingConfig(BaseModel):
    patience: Optional[int] = Field(
        default=50, description="The number of epochs to wait for improvement before stopping training."
    )
    min_delta: Optional[float] = Field(
        default=100,
        description="The minimum improvement in the watched metric necessary to reset the patience countdown.",
    )
    ### The below are not implemented, just an idea. If we want to monitor another metric for early stopping besides val loss.
    watched_metric: Optional[str] = Field(
        default="val_loss", description="NOT IMPLEMENTED:which metric to monitor for early stopping."
    )
    watched_metric_polarity: Optional[str] = Field(
        default="Negative",
        description="NOT IMPLEMENTED:Whether the goal is a positive or negative change in the watched metric.",
    )




