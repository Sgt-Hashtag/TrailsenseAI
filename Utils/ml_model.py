import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class SuitabilityModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    def train_mock_model(self):
        """Simulates training on Strava Heatmap data."""
        data = {
            "incline": [0.01, 0.05, 0.10, 0.02, 0.08, 0.15],
            "is_trail": [1, 0, 1, 0, 1, 1],
            "greenery": [0.9, 0.2, 0.8, 0.1, 0.9, 1.0],
            "popularity": [0.95, 0.4, 0.85, 0.3, 0.9, 0.7],  # Target
        }
        df = pd.DataFrame(data)
        self.model.fit(df[["incline", "is_trail", "greenery"]], df["popularity"])

    def predict_weight(self, edge_data):
        """Used by the routing engine to weight edges."""
        incline = abs(edge_data.get("grade_abs", 0))
        is_trail = 1 if edge_data.get("highway") in ["path", "track", "footway"] else 0
        greenery = 0.8 if is_trail else 0.2

        feat = pd.DataFrame(
            [[incline, is_trail, greenery]],
            columns=["incline", "is_trail", "greenery"],
        )

        # Predict popularity (0.0 to 1.0)
        pop_score = self.model.predict(feat)[0]

        # Lower cost for better routes:
        return edge_data.get("length", 1) * (1.0 / (pop_score + 0.1))
