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
    energy_transfer: float = Field(
        0, description="Energy transfer between the two landmarks in kWh", min_value=0
    )
    water_transfer: float = Field(
        0, description="Water transfer between the two landmarks in liters", min_value=0
    )
    food_transfer: float = Field(
        0, description="Food transfer between the two landmarks in kg", min_value=0
    )

    # Capacity constraints
