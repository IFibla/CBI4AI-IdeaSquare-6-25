from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class LandmarkType(str, Enum):
    AGRICULTURE = "agriculture"
    ANIMAL_HUSBANDRY = "animal_husbandry"
    NEIGHBOURHOOD_SHOP = "neighbourhood_shop"
    POWER_PLANT = "power_plant"
    STORAGE = "storage"
    WATER_PRODUCTION = "water_production"
    WATER_TREATMENT = "water_treatment"


LandmarkProfile = {
    LandmarkType.AGRICULTURE: (1, 1, 0, 0, 0, 1),
    LandmarkType.ANIMAL_HUSBANDRY: (1, 1, 0, 0, 0, 1),
    LandmarkType.NEIGHBOURHOOD_SHOP: (1, 1, 1, 0, 0, 0),
    LandmarkType.POWER_PLANT: (1, 1, 0, 1, 0, 0),
    LandmarkType.STORAGE: (1, 1, 0, 0, 0, 0),
    LandmarkType.WATER_PRODUCTION: (1, 0, 0, 0, 1, 0),
}


class Landmark(BaseModel):
    uuid: str = Field(..., description="UUID of the landmark")
    type: LandmarkType = Field(
        ..., description="Essential/generalized type of the landmark"
    )
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
