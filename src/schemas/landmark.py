from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field
import torch


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
    uuid: int = Field(..., description="ID of the landmark")
    longitude: float = Field(
        0, description="Longitude of the landmark in degrees"
    )
    latitude: float = Field(
        0, description="Latitude of the landmark in degrees"
    )
    name: str = Field("", description="Name of the landmark")
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

    @property
    def needs(self) -> str:
        return [
            self.energy_consumption > 0,
            self.water_consumption > 0,
            self.food_consumption > 0,
        ]

    @property
    def numerical_type(self) -> List[str]:
        return [
            LandmarkType.AGRICULTURE == self.type,
            LandmarkType.ANIMAL_HUSBANDRY == self.type,
            LandmarkType.NEIGHBOURHOOD_SHOP == self.type,
            LandmarkType.POWER_PLANT == self.type,
            LandmarkType.STORAGE == self.type,
            LandmarkType.WATER_PRODUCTION == self.type,
        ]

    def to_tensor(self, dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
        return torch.tensor(
            [
                self.citizens,
                self.energy_consumption,
                self.energy_production,
                self.water_consumption,
                self.water_production,
                self.food_consumption,
                self.food_production,
            ]
            + self.needs
            + self.numerical_type,
            dtype=dtype,
        )
