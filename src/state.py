from typing import List, Union

from src.schemas import Adjacency, Landmark

class State:
    def __init__(
        self, citizens: int = 50000, energy_production_kwh: float = 1500000000,
        number_of_landmarks: int = 10, edge_probability: float = 0.5,
    ) -> None:
        self.citizens = citizens
        self.energy_production_kwh = energy_production_kwh
        self.number_of_landmarks = number_of_landmarks
        self.edge_probability = edge_probability

    def generate_state(self) -> Union[List[Adjacency], List[Landmark]]:
        landmarks = []
        for i in range(self.number_of_landmarks):
            landmark = Landmark(
                uuid=f"landmark-{i}",
                type="healthcare",
                citizens=self.citizens,
                energy_consumption=0,
                energy_production=self.energy_production_kwh / self.number_of_landmarks,
                water_consumption=0,
                water_production=0,
                food_consumption=0,
                food_production=0,
            )
            landmarks.append(landmark)