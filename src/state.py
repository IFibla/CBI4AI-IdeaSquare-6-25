from matplotlib import pyplot as plt
from itertools import combinations
from typing import List, Union
from tqdm import trange
import random
import uuid

from src.schemas import Adjacency, Landmark, LandmarkType, LandmarkProfile


class State:
    def __init__(
        self,
        citizens: int = 50000,
        energy_production_kwh: float = 1500000000,
        water_production_liters: float = 1500000000,
        food_production_kgs: float = 1500000000,
        n_landmarks: int = 30,
        edge_probability: float = 0.5,
    ) -> None:
        self.citizens = citizens
        self.energy_production_kwh = energy_production_kwh
        self.water_production_liters = water_production_liters
        self.food_production_kgs = food_production_kgs
        self.n_landmarks = n_landmarks
        self.edge_probability = edge_probability
        self.landmarks = []
        self.adjacencies = []

    @staticmethod
    def distribute_integer_randomly(total_value: int, n_targets: int) -> List[int]:
        if n_targets == 0 or total_value == 0:
            return [0] * n_targets
        if n_targets == 1:
            return [total_value]
        cuts = sorted(random.sample(range(1, total_value), n_targets - 1))
        return [a - b for a, b in zip(cuts + [total_value], [0] + cuts)]

    @staticmethod
    def distribute_float_randomly(total_value: float, n_targets: int) -> List[float]:
        if n_targets == 0 or total_value == 0:
            return [0.0] * n_targets
        weights = [random.random() for _ in range(n_targets)]
        total_weight = sum(weights)
        return [(w / total_weight) * total_value for w in weights]

    @staticmethod
    def map_distribution(
        indices: List[int], values: List[float], total_length: int
    ) -> List[float]:
        mapped = [0.0] * total_length
        for idx, val in zip(indices, values):
            mapped[idx] = val
        return mapped

    def generate_landmarks(self) -> List[Landmark]:
        types = list(LandmarkProfile.keys())

        selected_types = [random.choice(types) for _ in range(self.n_landmarks)]

        total_units = [0] * 6
        for lt in selected_types:
            profile = LandmarkProfile[lt]
            total_units = [x + y for x, y in zip(total_units, profile)]

        energy_producers = [
            i for i, lt in enumerate(selected_types) if LandmarkProfile[lt][3] > 0
        ]
        water_producers = [
            i for i, lt in enumerate(selected_types) if LandmarkProfile[lt][4] > 0
        ]
        food_producers = [
            i for i, lt in enumerate(selected_types) if LandmarkProfile[lt][5] > 0
        ]

        energy_consumers = [
            i for i, lt in enumerate(selected_types) if LandmarkProfile[lt][0] > 0
        ]
        water_consumers = [
            i for i, lt in enumerate(selected_types) if LandmarkProfile[lt][1] > 0
        ]
        food_consumers = [
            i for i, lt in enumerate(selected_types) if LandmarkProfile[lt][2] > 0
        ]

        energy_prod_values = self.distribute_float_randomly(
            self.energy_production_kwh, len(energy_producers)
        )
        water_prod_values = self.distribute_float_randomly(
            self.water_production_liters, len(water_producers)
        )
        food_prod_values = self.distribute_float_randomly(
            self.food_production_kgs, len(food_producers)
        )

        energy_cons_values = self.distribute_float_randomly(
            self.energy_production_kwh, len(energy_consumers)
        )
        water_cons_values = self.distribute_float_randomly(
            self.water_production_liters, len(water_consumers)
        )
        food_cons_values = self.distribute_float_randomly(
            self.food_production_kgs, len(food_consumers)
        )

        energy_production = self.map_distribution(
            energy_producers, energy_prod_values, self.n_landmarks
        )
        water_production = self.map_distribution(
            water_producers, water_prod_values, self.n_landmarks
        )
        food_production = self.map_distribution(
            food_producers, food_prod_values, self.n_landmarks
        )

        energy_consumption = self.map_distribution(
            energy_consumers, energy_cons_values, self.n_landmarks
        )
        water_consumption = self.map_distribution(
            water_consumers, water_cons_values, self.n_landmarks
        )
        food_consumption = self.map_distribution(
            food_consumers, food_cons_values, self.n_landmarks
        )

        shop_indices = [
            i
            for i, lt in enumerate(selected_types)
            if lt == LandmarkType.NEIGHBOURHOOD_SHOP
        ]
        citizen_distribution = self.distribute_integer_randomly(
            self.citizens, len(shop_indices)
        )
        citizens = [0] * self.n_landmarks
        for idx, val in zip(shop_indices, citizen_distribution):
            citizens[idx] = val

        landmarks = []
        for i, lt in enumerate(selected_types):
            landmark = Landmark(
                uuid=str(uuid.uuid4()),
                type=lt,
                citizens=citizens[i],
                energy_production=energy_production[i],
                water_production=water_production[i],
                food_production=food_production[i],
                energy_consumption=energy_consumption[i],
                water_consumption=water_consumption[i],
                food_consumption=food_consumption[i],
            )
            landmarks.append(landmark)

        return landmarks

    def generate_adjacency(self, landmarks: List[Landmark]) -> List[Adjacency]:
        if not landmarks:
            raise ValueError("Landmarks list cannot be empty")

        n_landmarks = len(landmarks)
        edge_prob = self.edge_probability

        adjacencies = []
        pairs = list(combinations(range(n_landmarks), 2))

        for _ in trange(len(pairs), desc="Generating adjacency matrix", unit="edge"):
            i, j = pairs[_]
            if random.random() < edge_prob:
                adjacencies.append(
                    Adjacency(
                        from_landmark=str(landmarks[i].uuid),
                        to_landmark=str(landmarks[j].uuid),
                        time=random.uniform(0.5, 2.0),
                        distance=random.uniform(1.0, 100.0),
                    )
                )

        return adjacencies

    def generate_state(self) -> Union[List[Landmark], List[Adjacency]]:
        self.landmarks = self.generate_landmarks()
        self.adjacencies = self.generate_adjacency(self.landmarks)
        return self.landmarks, self.adjacencies

    def __repr__(self) -> str:
        lines = ["```mermaid", "graph TD"]
        uuid_to_short = {}
        class_assignments = []

        for i, landmark in enumerate(self.landmarks):
            short_id = landmark.uuid[:6]
            uuid_to_short[landmark.uuid] = short_id
            node_label = f"{short_id}[{landmark.type.value}]"
            lines.append(f"    {node_label}")
            class_assignments.append(f"    class {short_id} {landmark.type.value};")

        for adjacency in self.adjacencies:
            from_id = uuid_to_short.get(adjacency.from_landmark)
            to_id = uuid_to_short.get(adjacency.to_landmark)
            if from_id and to_id:
                lines.append(f"    {from_id} --- {to_id}")

        # Define class colors
        type_colors = {
            "agriculture": "#a1d99b",
            "animal_husbandry": "#fdae6b",
            "neighbourhood_shop": "#9ecae1",
            "power_plant": "#fc9272",
            "storage": "#dadaeb",
            "water_production": "#9e9ac8",
            "water_treatment": "#c7e9c0",
        }

        for t, color in type_colors.items():
            lines.append(f"    classDef {t} fill:{color},stroke:#333,stroke-width:1px;")

        lines.extend(class_assignments)
        lines.append("```")
        return "\n".join(lines)

