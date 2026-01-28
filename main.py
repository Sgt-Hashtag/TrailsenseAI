import os
import json
import hashlib
from contextlib import asynccontextmanager
from typing import List
from datetime import datetime
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

load_dotenv()

# --- Configuration ---
DEFAULT_LOCATION = "Berlin, Germany"

def get_location_hash(location: str) -> str:
    """Generate a unique hash for a location string."""
    return hashlib.md5(location.encode()).hexdigest()[:12]

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
        """Refined training data to match realistic athlete preferences."""
        np.random.seed(42)
        n_rows = 500 # More data for better generalization
        
        # Generate random features
        incline = np.random.uniform(0, 0.25, n_rows)
        is_trail = np.random.randint(0, 2, n_rows)
        greenery = np.random.uniform(0, 1.0, n_rows)
        
        # Apply the logic: What do athletes ACTUALLY want?
        # Start with baseline
        popularity = 0.4 
        
        # 1. Strong preference for trails
        popularity += is_trail * 0.3
        
        # 2. Strong preference for greenery
        popularity += greenery * 0.2
        
        # 3. Incline "Sweet Spot" (Athletes love 5-10%, hate >15%)
        # Using a Gaussian-like penalty for steepness
        popularity += np.where((incline >= 0.05) & (incline <= 0.10), 0.1, 0)
        popularity -= np.where(incline > 0.15, 0.2, 0)
        
        y = np.clip(popularity, 0, 1)
        
        X = pd.DataFrame({
            'incline': incline,
            'is_trail': is_trail,
            'greenery': greenery
        })
        
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

def parse_athlete_input(user_input: str) -> ParsedIntent:
    """Parse natural language input into structured intent."""
    if not client:
        print("Gemini API key not found, using default intent")
        return ParsedIntent(distance_km=5.0, activity="run", terrain="balanced", location="Berlin")
    
    prompt = f"""
Return ONLY valid JSON with these exact keys: distance_km, activity, terrain, location

User request: {user_input}
JSON:
"""
# Ideally can use any LLM or tool to parse the user input into structured data. would have opted for ollam or even a SLM here but Gemini is easiest to demo with.
    try:
        resp = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw = json.loads(resp.text)
        return ParsedIntent(**raw)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"JSON parsing error: {e}")
        return ParsedIntent(distance_km=5.0, activity="run", terrain="balanced", location="Berlin")
    except Exception as e:
        print(f"Gemini API error: {e}")
        return ParsedIntent(distance_km=5.0, activity="run", terrain="balanced", location="Berlin")

# --- FastAPI App ---
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing SuitabilityModel...")
    ml_engine = SuitabilityModel("trailsense_model.pkl")
    
    # Determine location
    location = os.getenv("TRAILSENSE_LOCATION", DEFAULT_LOCATION)
    location_hash = get_location_hash(location)
    
    print(f"Target location: {location}")
    print(f"Location hash: {location_hash}")
    
    # Cache
    cache_dir = "graph_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file_pkl = os.path.join(cache_dir, f"graph_{location_hash}.pkl")
    metadata_file = os.path.join(cache_dir, f"metadata_{location_hash}.json")
    
    cache_file = cache_file_pkl
    
    print(f"Cache directory: {os.path.abspath(cache_dir)}")
    print(f"Looking for: {os.path.basename(cache_file)}")
    
    # load from cache
    G = None
    if os.path.exists(cache_file) and os.path.exists(metadata_file):
        try:
            print(f"Cache files found, loading...")
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get('location') == location:
                with open(cache_file, 'rb') as f:
                    G = pickle.load(f)
                print(f"Loaded from cache: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
                print(f"Cached on: {metadata.get('cached_at', 'unknown')}")
            else:
                print(f"Location mismatch in cache")
        except Exception as e:
            print(f"Cache load failed: {e}")
            G = None
    else:
        print(f"No cache found, will download from OSM")
    
    # Download and cache if needed
    if G is None:
        print(f"Downloading '{location}' from OpenStreetMap (2-5 min)...")
        
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        try:
            G_full = ox.graph_from_place(location, network_type="walk")
        except Exception as e:
            raise RuntimeError(f"Failed to download graph: {e}")
        
        largest_cc = max(nx.weakly_connected_components(G_full), key=len)
        G = G_full.subgraph(largest_cc).copy()
        
        print(f"Trimmed to largest component: {len(G.nodes):,} nodes")
        
        # Save to cache
        print(f"Saving to cache...")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
            
            file_size = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"Graph cached ({file_size:.1f} MB)")
            
            metadata = {
                "location": location,
                "location_hash": location_hash,
                "nodes": len(G.nodes),
                "edges": len(G.edges),
                "cached_at": datetime.utcnow().isoformat(),
                "network_type": "walk"
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved")
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    # Compute ML scores
    print("Computing ML suitability scores...")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    scores = ml_engine.predict_weights_batch(edges_gdf)
    nx.set_edge_attributes(G, scores, 'athlete_score')
    
    print(f"Ready: {len(G.nodes):,} nodes | {len(G.edges):,} edges")
    
    resources["graph"] = G
    resources["location"] = location
    yield
    resources.clear()

app = FastAPI(title="TrailSense API", lifespan=lifespan)

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    preference: str = "balanced"

class SearchRequest(BaseModel):
    prompt: str

class RouteResponse(BaseModel):
    distance_meters: float
    safety_score: float
    geometry: List[dict]
    route_type: str = "point_to_point"
    intent: dict = {}

def calculate_route_distance(G, route):
    """Fast distance calculation from edge lengths."""
    total = 0.0
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if v in G[u]:
            try:
                first_key = next(iter(G[u][v]))
                length = G[u][v][first_key]['length']
                total += float(length) if not hasattr(length, 'item') else float(length.item())
            except:
                pass
    return total

def get_path(s_lat, s_lon, e_lat, e_lon, weight_key):
    """Calculate shortest path between two points."""
    G = resources["graph"]
    
    orig = ox.distance.nearest_nodes(G, s_lon, s_lat)
    dest = ox.distance.nearest_nodes(G, e_lon, e_lat)
    
    print(f"Routing: node {orig} → {dest}")
    
    if orig == dest:
        neighbors = list(G.neighbors(orig))
        if len(neighbors) >= 2:
            route = [orig, neighbors[0], neighbors[1], orig]
        elif len(neighbors) == 1:
            route = [orig, neighbors[0], orig]
        else:
            raise HTTPException(400, "Isolated node")
    else:
        try:
            route = nx.shortest_path(G, orig, dest, weight=weight_key)
        except nx.NetworkXNoPath:
            raise HTTPException(404, "No route found")
    
    distance = calculate_route_distance(G, route)
    coords = [{"lat": G.nodes[n]['y'], "lon": G.nodes[n]['x']} for n in route]
    
    print(f"Route: {len(route)} nodes, {distance:.0f}m")
    
    return {"distance_meters": distance, "geometry": coords}

@app.post("/v1/route", response_model=RouteResponse)
async def create_route(req: RouteRequest):
    """Direct point-to-point routing."""
    weight = "athlete_score" if req.preference.lower() in ["hilly", "trail"] else "length"
    result = get_path(req.start_lat, req.start_lon, req.end_lat, req.end_lon, weight)
    
    dist_km = result["distance_meters"] / 1000
    safety = min(1.0, 0.8 + (0.15 if weight == "athlete_score" else 0.05) - (0.05 * dist_km / 10))
    
    return RouteResponse(
        distance_meters=result["distance_meters"],
        safety_score=round(safety, 2),
        geometry=result["geometry"],
        route_type="point_to_point",
        intent={"preference": req.preference}
    )

def find_target_node_at_distance(G, start_lat, start_lon, target_km, weight='length'):
    """Find a node creating a round trip close to target_km."""
    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    
    best_node = None
    best_error = float('inf')
    
    # different radii and bearings
    for radius_factor in [0.6, 0.7, 0.5, 0.8]:
        offset_deg = (target_km / 2 * radius_factor) / 111.0
        
        for bearing_deg in [45, 135, 225, 315, 0, 90, 180, 270]:
            bearing_rad = math.radians(bearing_deg)
            lat = start_lat + offset_deg * math.cos(bearing_rad)
            lon = start_lon + offset_deg * math.sin(bearing_rad) / math.cos(math.radians(start_lat))
            
            try:
                candidate = ox.distance.nearest_nodes(G, lon, lat)
                
                if candidate == start_node or G.degree(candidate) == 0:
                    continue
                
                # actual round-trip distance
                try:
                    out = nx.shortest_path(G, start_node, candidate, weight=weight)
                    back = nx.shortest_path(G, candidate, start_node, weight=weight)
                    full = out + back[1:]
                    
                    actual_km = calculate_route_distance(G, full) / 1000
                    error = abs(actual_km - target_km)
                    
                    if error < best_error:
                        best_error = error
                        best_node = candidate
                        
                        if error / target_km < 0.1:  # Within 10%
                            print(f"Match: {actual_km:.1f}km (target: {target_km}km)")
                            return best_node
                except:
                    pass
            except:
                pass
    
    if best_node:
        print(f"Best match: ~{target_km - best_error:.1f}km")
        return best_node
    
    # Fallback
    neighbors = list(G.neighbors(start_node))
    return neighbors[0] if neighbors else start_node

def get_location_coords_hybrid(location: str, G=None):
    """Resolve location to coordinates."""
    location_lower = location.lower().strip()
    
    landmarks = {
        "mitte": (52.5213, 13.4125),
        "tiergarten": (52.5194, 13.3539),
        "kreuzberg": (52.4986, 13.3904),
        "prenzlauer": (52.5367, 13.4180),
        "friedrichshain": (52.5127, 13.4359),
        "charlottenburg": (52.5075, 13.3027),
        "alexanderplatz": (52.5213, 13.4125),
    }
    
    for name, coords in landmarks.items():
        if name in location_lower:
            print(f"Static hit: {name}")
            return coords
    
    # OSMnx geocoding
    try:
        gdf = ox.geocode_to_gdf(f"{location}, Berlin, Germany")
        # using bounds
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        lat = (bounds[1] + bounds[3]) / 2  # Average of min and max y
        lon = (bounds[0] + bounds[2]) / 2  # Average of min and max x
        
        if G:
            node = ox.distance.nearest_nodes(G, lon, lat)
            return (G.nodes[node]['y'], G.nodes[node]['x'])
        return (lat, lon)
    except:
        pass
    
    # Fallback to Mitte
    return (52.5213, 13.4125)

@app.post("/v1/search", response_model=RouteResponse)
async def search_ai_route(req: SearchRequest):
    """AI-powered natural language route search."""
    print(f"Search: '{req.prompt}'")
    
    intent = parse_athlete_input(req.prompt)
    G = resources["graph"]
    
    start_lat, start_lon = get_location_coords_hybrid(intent.location, G)
    orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    
    terrain = intent.terrain.lower()
    weight = "athlete_score" if any(t in terrain for t in ["hilly", "trail"]) else "length"
    
    dest_node = find_target_node_at_distance(G, start_lat, start_lon, intent.distance_km, weight)
    
    if intent.terrain not in ["balanced", "trail", "hilly", "flat"]:
        intent.terrain = "balanced"
    
    print(f"Intent: {intent.distance_km}km '{intent.terrain}' in '{intent.location}'")
    
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
        back = nx.shortest_path(G, dest_node, orig_node, weight=weight)
        full = route + back[1:]
    except:
        full = [orig_node, dest_node, orig_node]
    
    distance = calculate_route_distance(G, full)
    coords = [{"lat": G.nodes[n]['y'], "lon": G.nodes[n]['x']} for n in full]
    
    actual_km = distance / 1000
    match = max(0, 1.0 - abs(actual_km - intent.distance_km) / max(intent.distance_km, 1))
    
    print(f"Route: {actual_km:.2f}km (requested: {intent.distance_km}km)")
    
    return RouteResponse(
        distance_meters=distance,
        safety_score=round(min(1.0, 0.85 + 0.12 * match), 2),
        geometry=coords,
        route_type="round_trip",
        intent={
            "distance_km": intent.distance_km,
            "activity": intent.activity,
            "terrain": intent.terrain,
            "location": intent.location
        }
    )

@app.get("/health")
def health_check():
    G = resources.get("graph")
    location = resources.get("location", "Unknown")
    return {
        "status": "online",
        "location": location,
        "graph_loaded": G is not None,
        "nodes": len(G.nodes) if G else 0,
        "edges": len(G.edges) if G else 0
    }