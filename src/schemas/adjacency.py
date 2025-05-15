from typing import List

from pydantic import BaseModel, Field
import torch


class Adjacency(BaseModel):
    from_landmark: int = Field(
        ..., description="UUID of the landmark from which the connection originates"
    )
    to_landmark: int = Field(
        ..., description="UUID of the landmark to which the connection leads"
    )
    time: float = Field(
        ..., description="Time taken to travel between the two landmarks in hours"
    )
    distance: float = Field(
        ..., description="Distance between the two landmarks in kilometers"
    )

    def get_adjacency(
        self
    ) -> tuple[int, int]:
        return [self.from_landmark, self.to_landmark]

    def weights_to_tensor(
        self
    ) -> List[float]:
        return [1/self.time, 1/self.distance]
