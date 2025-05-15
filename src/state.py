from torch_geometric.data import Data
from itertools import combinations
import matplotlib.pyplot as plt
from typing import List, Union
from tqdm import trange
import networkx as nx
import random
import torch
import matplotlib.image as mpimg  # For reading background image
import os

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
                uuid=i,
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

    def generate_random_state(self) -> Union[List[Landmark], List[Adjacency]]:
        self.landmarks = self.generate_landmarks()
        self.adjacencies = self.generate_adjacency(self.landmarks)
        return self.landmarks, self.adjacencies

    def plot(self, figsize=(12, 8), background_image_path: str = None) -> plt.Figure:
        if not self.landmarks or not self.adjacencies:
            raise ValueError("Must generate state before plotting.")

        if background_image_path and os.path.exists(background_image_path):
            img = mpimg.imread(background_image_path)


        G = nx.Graph()
        id_map = {}

        # === 1. Add Nodes ===
        for lm in self.landmarks:
            short_id = str(lm.name)
            id_map[str(lm.uuid)] = short_id
            G.add_node(short_id, type=lm.type)

        # === 2. Add Edges ===
        for adj in self.adjacencies:
            G.add_edge(
                id_map[str(adj.from_landmark)],
                id_map[str(adj.to_landmark)],
                distance=adj.distance,
            )

        # === 3. Use latitude/longitude for positioning ===
        pos = {
            id_map[str(lm.uuid)]: (lm.longitude, lm.latitude)  # Flip y-axis for typical image coordinates
            for lm in self.landmarks
        }

        type_colors = {
            "agriculture": "#a1d99b",
            "animal_husbandry": "#fdae6b",
            "neighbourhood_shop": "#9ecae1",
            "power_plant": "#fc9272",
            "storage": "#dadaeb",
            "water_production": "#9e9ac8",
            "water_treatment": "#c7e9c0",
        }

        node_colors = [
            type_colors.get(G.nodes[n]["type"].value, "#cccccc") for n in G.nodes
        ]

        # === 4. Create Plot ===
        fig, ax = plt.subplots(figsize=figsize)

        # === 5. Add Background Image ===
        if background_image_path and os.path.exists(background_image_path):
            ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])  # Match coordinate system to image size

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#ff0000", width=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        # === 6. Legend ===
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                color="w",
                label=label.replace("_", " ").capitalize(),
                markerfacecolor=color,
                markersize=10,
            )
            for label, color in type_colors.items()
        ]

        ax.legend(
            handles=handles,
            title="Landmark Types",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        ax.set_title("Landmarks and Adjacencies (Using Latitude/Longitude)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()

        return fig

    @property
    def gnn_data(self, dtype: torch.dtype = torch.float32) -> Data:
        node_features = torch.stack(
            [lm.to_tensor(dtype=dtype) for lm in self.landmarks]
        )

        edge_tuples = [adj.get_adjacency() for adj in self.adjacencies]
        edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
        rev_index = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, rev_index], dim=1)

        edge_attr = torch.tensor(
            [adj.weights_to_tensor() for adj in self.adjacencies] + [adj.weights_to_tensor() for adj in self.adjacencies],
            dtype=dtype,
        )

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


