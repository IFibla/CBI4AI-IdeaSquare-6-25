from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class Landmark(BaseModel):
    uuid: str = Field(..., description="UUID of the landmark")
    type: Literal[
        "healthcare",
        "agriculture",
        "animal_husbandry",
        "storage",
        "infrastructure",
        "water_production",
        "water_treatment",
        "data",
        "energy"
    ] = Field(..., description="Essential/generalized type of the landmark")
    citizens: int = Field(0, description="Number of citizens served by the landmark")
    energy_consumption: float = Field(
        0, description="Energy consumption of the landmark in kWh"
    )
    energy_production: float = Field(
        0, description="Energy production of the landmark in kWh"
    )
    water_consumption: float = Field(
        0, description="Water consumption of the landmark in liters"
    )
    water_production: float = Field(
        0, description="Water production of the landmark in liters"
    )
    food_consumption: float = Field(
        0, description="Food consumption of the landmark in kg"
    )
    food_production: float = Field(
        0, description="Food production of the landmark in kg"
    )
