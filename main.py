import os
import json
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import osmnx as ox
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from google import genai
from dotenv import load_dotenv
import numpy as np
import pickle
import math
from google.api_core import exceptions as gapi_exceptions
import time

load_dotenv()

# --- ML Model ---
class SuitabilityModel:
    def __init__(self, model_path: str = "trailsense_model.pkl"):
        self.model_path = model_path
        self.model = None
        
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print("No saved model found. Training and saving now...")
            self.train_and_save()

    def train_and_save(self):
        """Expanded training data simulating athlete preferences."""
        data = {
            'incline': [0.01, 0.05, 0.10, 0.02, 0.08, 0.15, 0.25, 0.00, 0.03, 0.12],
            'is_trail': [1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
            'greenery': [0.9, 0.2, 0.8, 0.1, 0.9, 1.0, 0.7, 0.0, 0.3, 0.8],
            'popularity': [0.95, 0.4, 0.85, 0.3, 0.9, 0.7, 0.4, 0.1, 0.5, 0.8]
        }
        df = pd.DataFrame(data)
        X = df[['incline', 'is_trail', 'greenery']]
        y = df['popularity']
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved at {self.model_path}")
    
    def load_model(self):
        """Load pre-trained model."""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {self.model_path}")
    
    def predict_weights_batch(self, edges_gdf):
        """VECTORIZED batch prediction for OSMnx edges"""
        df = edges_gdf.copy()
        
        df['incline'] = df.get('grade_abs', 0)
        df['is_trail'] = df['highway'].isin(['path', 'track', 'footway']).astype(int).fillna(0)
        df['greenery'] = np.where(df['is_trail'] == 1, 0.8, 0.2)
        
        features = ['incline', 'is_trail', 'greenery']
        X = df[features].fillna(0)
        pop_scores = self.model.predict(X)

        costs = df['length'].fillna(1) * (1.0 / (pop_scores + 0.1))
        
        scores_dict = dict(zip(df.index, costs))
        return scores_dict

# --- Gemini Parser ---
class ParsedIntent(BaseModel):
    distance_km: float = Field(description="The requested distance in kilometers")
    activity: str = Field(description="The type of activity (run, cycle, etc)")
    terrain: str = Field(description="The preferred terrain style")
    location: str = Field(description="The specific city or area")

GEMINI_API_KEY = "AIzaSyCvfB_C2qA2QD4D78Mc6YnYBth8b8gGxPI"
client = genai.Client(api_key=GEMINI_API_KEY)

def parse_athlete_input(user_input: str) -> ParsedIntent:
    prompt = f"""
Return ONLY valid JSON with these exact keys: distance_km, activity, terrain, location

User request: {user_input}
JSON:
"""

    for attempt in range(2):  # simple retry: 1 retry max
        try:
            resp = client.models.generate_content(
                model="models/gemini-2.5-flash-lite",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            raw = json.loads(resp.text)
            return ParsedIntent(**raw)

        # 🔹 AI service overloaded / rate-limited / unavailable
        except (
            gapi_exceptions.ResourceExhausted,   # 429 / quota / overload
            gapi_exceptions.ServiceUnavailable, # 503
            gapi_exceptions.DeadlineExceeded,   # timeout
        ):
            if attempt == 0:
                time.sleep(0.5)  # brief backoff
                continue
            break

        # 🔹 Model returned garbage / partial JSON
        except (json.JSONDecodeError, ValidationError):
            break

        # 🔹 Anything unexpected from the SDK
        except Exception:
            break

    # Safe fallback
    return ParsedIntent(
        distance_km=5.0,
        activity="run",
        terrain="balanced",
        location="Berlin",
    )
# --- FastAPI App ---
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing SuitabilityModel...")
    ml_engine = SuitabilityModel("trailsense_model.pkl")
    
    print("📍 Loading Berlin + connected component...")
    
    ox.settings.geocode_cache = True
    ox.settings.nominatim_timeout = 10
    ox.settings.log_console = False
    ox.settings.timeout = 300
    ox.settings.use_cache = True
    
    G_full = ox.graph_from_place("Berlin, Germany", network_type="walk")
    
    # Largest connected component
    largest_cc = max(nx.weakly_connected_components(G_full), key=len)
    G = G_full.subgraph(largest_cc).copy()  # IMPORTANT: .copy() to make it mutable
    
    print(f"✂️ Trimmed: {len(G_full.nodes)} → {len(G.nodes)} nodes")
    
    # DIAGNOSTIC: Check if edges have length attribute
    sample_edges = list(G.edges(data=True, keys=True))[:3]
    print("\n🔍 Sample edge data:")
    for u, v, key, data in sample_edges:
        print(f"  {u}->{v} [key={key}]: length={data.get('length', 'MISSING')}")
    
    # If no length attributes, add them using geometry
    edges_with_length = sum(1 for _, _, _, d in G.edges(data=True, keys=True) if 'length' in d)
    if edges_with_length == 0:
        print("⚠️ WARNING: No length attributes found! Adding them now...")
        G = ox.distance.add_edge_lengths(G)
        print("✅ Edge lengths added")
    else:
        print(f"✅ Found {edges_with_length}/{len(G.edges)} edges with length attribute")
    
    # Vectorized scoring
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    scores = ml_engine.predict_weights_batch(edges_gdf)
    nx.set_edge_attributes(G, scores, 'athlete_score')
    
    print(f"✅ Ready: {len(G.nodes)} nodes, {len(G.edges)} edges")
    resources["graph"] = G
    yield
    resources.clear()

app = FastAPI(title="TrailSense API", lifespan=lifespan)

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    preference: str = "balanced"

@app.get("/debug/edge/{u}/{v}")
def debug_edge(u: int, v: int):
    """Debug endpoint to inspect edge data structure."""
    G = resources["graph"]
    if u not in G.nodes or v not in G.nodes:
        return {"error": "Node not found"}
    
    if v in G[u]:
        edge_data = G[u][v]
        
        # Convert to serializable format
        if isinstance(edge_data, dict):
            first_key = next(iter(edge_data.keys())) if edge_data else None
            
            if isinstance(first_key, int):
                # MultiDiGraph
                serialized = {
                    "type": "MultiDiGraph",
                    "edges": {k: dict(d) for k, d in edge_data.items()}
                }
            else:
                # Simple DiGraph
                serialized = {
                    "type": "DiGraph",
                    "edge_data": dict(edge_data)
                }
        else:
            serialized = {
                "type": "Unknown",
                "raw": str(edge_data)
            }
        
        return {
            "u": u,
            "v": v,
            "structure": serialized
        }
    else:
        return {"error": f"No edge from {u} to {v}"}

class SearchRequest(BaseModel):
    prompt: str

class RouteResponse(BaseModel):
    distance_meters: float
    safety_score: float
    geometry: List[dict]
    route_type: str = "point_to_point"  # or "round_trip"
    intent: dict = {}  # For search endpoint to return parsed intent

def calculate_route_distance(G, route):
    """
    ACCURATE distance calculation matching OSM data.
    Works with both DiGraph and MultiDiGraph structures.
    """
    from geopy.distance import geodesic
    
    total_distance = 0.0
    
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        
        if v not in G[u]:
            # No edge - calculate geodesic distance
            u_coords = (G.nodes[u]['y'], G.nodes[u]['x'])
            v_coords = (G.nodes[v]['y'], G.nodes[v]['x'])
            edge_length = geodesic(u_coords, v_coords).meters
            total_distance += edge_length
            print(f"❌ No edge {u}->{v}, using geodesic: {edge_length:.1f}m")
            continue
        
        edge_data = G[u][v]
        edge_length = None
        
        # Try to extract length from the edge data structure
        try:
            # Check if it's a dict-like object (could be MultiDiGraph or DiGraph)
            if hasattr(edge_data, 'items'):
                # Get all items
                items = list(edge_data.items())
                
                if items and isinstance(items[0][0], int):
                    # MultiDiGraph: G[u][v] = {0: {...}, 1: {...}}
                    edge_lengths = []
                    for key, data in items:
                        # Try different ways to get length
                        if 'length' in data:
                            length_val = data['length']
                            # Handle numpy types
                            if hasattr(length_val, 'item'):
                                length_val = float(length_val.item())
                            else:
                                length_val = float(length_val)
                            
                            if length_val > 0:
                                edge_lengths.append(length_val)
                    
                    if edge_lengths:
                        edge_length = min(edge_lengths)
                        if i < 3:
                            print(f"  Edge {i}: {u}->{v}, length={edge_length:.1f}m (MultiDiGraph)")
                else:
                    # Simple DiGraph: G[u][v] = {length: 100, highway: 'path'}
                    if 'length' in edge_data:
                        length_val = edge_data['length']
                        if hasattr(length_val, 'item'):
                            edge_length = float(length_val.item())
                        else:
                            edge_length = float(length_val)
                        
                        if i < 3:
                            print(f"  Edge {i}: {u}->{v}, length={edge_length:.1f}m (DiGraph)")
        except Exception as e:
            if i < 3:
                print(f"  Edge {i}: Error extracting length: {e}")
        
        # Fallback to geodesic if we couldn't get length
        if edge_length is None or edge_length <= 0:
            u_coords = (G.nodes[u]['y'], G.nodes[u]['x'])
            v_coords = (G.nodes[v]['y'], G.nodes[v]['x'])
            edge_length = geodesic(u_coords, v_coords).meters
            if i < 3:
                print(f"  Edge {i}: {u}->{v}, calculated={edge_length:.1f}m (fallback)")
        
        total_distance += edge_length
    
    return total_distance


def get_path(s_lat, s_lon, e_lat, e_lon, weight_key):
    """
    FIXED: Proper routing with accurate distance calculation.
    """
    G = resources["graph"]
    
    # Find nearest nodes
    orig = ox.distance.nearest_nodes(G, s_lon, s_lat)
    dest = ox.distance.nearest_nodes(G, e_lon, e_lat)
    
    print(f"🗺️ Routing from node {orig} ({s_lat:.4f}, {s_lon:.4f}) to {dest} ({e_lat:.4f}, {e_lon:.4f})")
    
    # Handle same node or isolated node
    if orig == dest:
        print("⚠️ Start and end are the same node - creating small loop")
        neighbors = list(G.neighbors(orig))
        if len(neighbors) >= 2:
            route = [orig, neighbors[0], neighbors[1], orig]
        elif len(neighbors) == 1:
            route = [orig, neighbors[0], orig]
        else:
            raise HTTPException(status_code=400, detail="Isolated node with no connections")
    else:
        # Try to find path
        try:
            route = nx.shortest_path(G, orig, dest, weight=weight_key)
        except nx.NetworkXNoPath:
            print("❌ No path found between nodes")
            raise HTTPException(status_code=404, detail=f"No route found between coordinates")
    
    # Calculate ACCURATE distance
    path_length = calculate_route_distance(G, route)
    
    # Extract coordinates
    coords = [{"lat": G.nodes[n]['y'], "lon": G.nodes[n]['x']} for n in route]
    
    print(f"✅ Route: {len(route)} nodes, {path_length:.1f}m (weight: {weight_key})")
    
    return {
        "distance_meters": path_length,
        "geometry": coords,
        "node_count": len(route)
    }


@app.post("/v1/route", response_model=RouteResponse)
async def create_route(req: RouteRequest):
    """Pro routing: direct coordinates with preference weights."""
    print(f"\n📍 /v1/route request: ({req.start_lat}, {req.start_lon}) → ({req.end_lat}, {req.end_lon})")
    
    weight = "athlete_score" if req.preference.lower() in ["hilly", "trail"] else "length"
    
    result = get_path(req.start_lat, req.start_lon, req.end_lat, req.end_lon, weight)
    
    # Dynamic safety score
    route_dist_km = result["distance_meters"] / 1000
    safety_score = min(1.0, 0.8 + (0.15 if weight == "athlete_score" else 0.05) - (0.05 * route_dist_km / 10))
    
    return RouteResponse(
        distance_meters=result["distance_meters"],
        safety_score=round(safety_score, 2),
        geometry=result["geometry"]
    )

def find_target_node_at_distance(G, start_lat, start_lon, target_km):
    """Find node approximately target_km/2 away for round trip."""
    # More accurate: use bearing-based offset
    offset_degrees = (target_km / 2) / 111.0  # 1 degree ≈ 111 km
    
    # Try multiple directions to find valid nodes
    bearings = [45, 135, 225, 315, 0, 90, 180, 270]  # NE, SE, SW, NW, then cardinal
    
    for bearing_deg in bearings:
        bearing_rad = math.radians(bearing_deg)
        target_lat = start_lat + offset_degrees * math.cos(bearing_rad)
        target_lon = start_lon + offset_degrees * math.sin(bearing_rad) / math.cos(math.radians(start_lat))
        
        try:
            target_node = ox.distance.nearest_nodes(G, target_lon, target_lat)
            
            # Verify it's in the graph and connected
            if target_node in G.nodes and G.degree(target_node) > 0:
                print(f"🎯 Target ~{target_km/2:.1f}km away at bearing {bearing_deg}°: node {target_node}")
                return target_node
        except:
            continue
    
    # Fallback: find any node within reasonable distance
    print("⚠️ Using fallback target node selection")
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    neighbors = list(G.neighbors(start_node))
    return neighbors[0] if neighbors else start_node


def get_location_coords_hybrid(location: str, G=None) -> tuple[float, float]:
    """Hybrid: Static landmarks → Dynamic OSMnx → Berlin fallback."""
    location_lower = location.lower().strip()
    
    static_landmarks = {
        "mitte": (52.5213, 13.4125),
        "tiergarten": (52.5194, 13.3539),
        "kreuzberg": (52.4986, 13.3904),
        "pren": (52.5367, 13.4180),
        "prenzlauer": (52.5367, 13.4180),
        "friedrichshain": (52.5127, 13.4359),
        "charlottenburg": (52.5075, 13.3027),
        "neukölln": (52.4740, 13.4325),
        "wedding": (52.5441, 13.3773),
        "moabit": (52.5260, 13.3900),
        "alexanderplatz": (52.5213, 13.4125),
        "brandenburg": (52.5163, 13.3777),
        "potzdamer": (52.5095, 13.3769)
    }
    
    for landmark, coords in static_landmarks.items():
        if landmark in location_lower:
            print(f"⚡ STATIC HIT: '{location}' → {landmark}")
            return coords
    
    try:
        print(f"🔍 OSMnx geocoding: '{location}, Berlin, Germany'")
        point_gdf = ox.geocode_to_gdf(location + ", Berlin, Germany", which_result=1)
        center_lat = point_gdf.geometry.centroid.y.iloc[0]
        center_lon = point_gdf.geometry.centroid.x.iloc[0]
        
        if G is not None:
            nearest_node = ox.distance.nearest_nodes(G, center_lon, center_lat)
            snapped_coords = (G.nodes[nearest_node]['y'], G.nodes[nearest_node]['x'])
            print(f"📍 OSMnx → node snap: {snapped_coords}")
            return snapped_coords
        
        print(f"📍 OSMnx raw: ({center_lat:.4f}, {center_lon:.4f})")
        return (center_lat, center_lon)
        
    except Exception as e:
        print(f"❌ OSMnx failed: {e}")
    
    print("🔄 Walkable Mitte node fallback")
    mitte_node = ox.distance.nearest_nodes(G, 13.4125, 52.5213)
    return (G.nodes[mitte_node]['y'], G.nodes[mitte_node]['x'])


@app.post("/v1/search", response_model=RouteResponse)
async def search_ai_route(req: SearchRequest):
    """AI-powered route search with natural language."""
    print(f"\n🤖 /v1/search request: '{req.prompt}'")
    
    intent = parse_athlete_input(req.prompt)
    G = resources["graph"]
    
    # Get starting location
    start_lat, start_lon = get_location_coords_hybrid(intent.location, G)
    orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    
    # Determine weight before finding target
    terrain = intent.terrain.lower()
    weight = "athlete_score" if any(t in terrain for t in ["hilly", "trail", "undulating"]) else "length"
    
    # Find destination for round trip with actual distance calculation
    dest_node = find_target_node_at_distance(G, start_lat, start_lon, intent.distance_km, weight)
    
    print(f"📊 Parsed: {intent.distance_km}km '{intent.terrain}' in '{intent.location}'")
    print(f"🗺️ Start node {orig_node} → Dest {dest_node}")
    
    terrain = intent.terrain.lower()
    weight = "athlete_score" if any(t in terrain for t in ["hilly", "trail", "undulating"]) else "length"
    
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
        return_route = nx.shortest_path(G, dest_node, orig_node, weight=weight)
        full_route = route + return_route[1:]  # Skip duplicate dest_node
        
        print(f"✅ Path found: {len(full_route)} nodes")
        
    except nx.NetworkXNoPath:
        print("❌ No path - using fallback")
        full_route = [orig_node, dest_node, orig_node]
    
    # ACCURATE distance calculation
    path_length = calculate_route_distance(G, full_route)
    
    coords = [{"lat": G.nodes[n]['y'], "lon": G.nodes[n]['x']} for n in full_route]
    
    actual_km = path_length / 1000
    dist_match = max(0, 1.0 - abs(actual_km - intent.distance_km) / max(intent.distance_km, 1))
    
    print(f"📏 Route distance: {actual_km:.2f}km (requested: {intent.distance_km}km)")
    
    return RouteResponse(
        distance_meters=path_length,
        safety_score=round(min(1.0, 0.85 + 0.12 * dist_match), 2),
        geometry=coords
    )


@app.get("/health")
def health_check():
    G = resources.get("graph")
    return {
        "status": "online",
        "model_version": "v1.0-alpha",
        "graph_loaded": G is not None,
        "nodes": len(G.nodes) if G else 0,
        "edges": len(G.edges) if G else 0
    }