from pydantic import BaseModel, Field


class Adjacency(BaseModel):
    from_landmark: str = Field(
        ..., description="UUID of the landmark from which the connection originates"
    )
    to_landmark: str = Field(
        ..., description="UUID of the landmark to which the connection leads"
    )
    time: float = Field(
        ..., description="Time taken to travel between the two landmarks in hours"
    )
    distance: float = Field(
        ..., description="Distance between the two landmarks in kilometers"
    )
