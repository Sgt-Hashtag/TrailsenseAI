import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static

# Configure the page to look like Strava
st.set_page_config(page_title="TrailSense AI | Route Discovery", page_icon="🏃‍♂️", layout="wide")

# Initialize session state for route data
if 'route_data' not in st.session_state:
    st.session_state.route_data = None
if 'manual_route_data' not in st.session_state:
    st.session_state.manual_route_data = None

st.title("🏃‍♂️ TrailSense AI")
st.markdown("### Intelligent Route Discovery for Berlin Athletes")

# Sidebar for manual controls or API status
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API Endpoint", value="http://localhost:8000")
    
    # # Check API health
    # try:
    #     health = requests.get(f"{api_url}/health", timeout=2).json()
    #     if health.get('status') == 'online':
    #         st.success(f"API Online")
    #         st.caption(f"Nodes: {health.get('nodes', 0):,}")
    #         st.caption(f"Edges: {health.get('edges', 0):,}")
    #     else:
    #         st.warning("API responding but not ready")
    # except:
    #     st.error("API Offline")
    
    st.divider()
    st.info(" This demo uses Gemini 2.5 Flash + Random Forest ML Model")
    
    st.markdown("### Example Prompts")
    st.code("5km run in Tiergarten", language=None)
    st.code("hilly 8km trail in Kreuzberg", language=None)
    st.code("10km cycle around Charlottenburg", language=None)

# Main content - two tabs
tab1, tab2 = st.tabs([" AI Search", " Manual Route"])

# TAB 1: AI-powered search
with tab1:
    st.markdown("#### Natural Language Route Planning")
    prompt = st.text_input(
        "Where do you want to move today?", 
        placeholder="e.g., Give me a hilly 8km run in Mitte",
        key="ai_prompt"
    )

    if st.button(" Generate Route", type="primary", key="ai_generate"):
        if not prompt:
            st.warning("Please enter a route description!")
        else:
            with st.spinner(" AI is analyzing terrain and finding the best path..."):
                try:
                    # Call /v1/search endpoint
                    response = requests.post(
                        f"{api_url}/v1/search", 
                        json={"prompt": prompt},
                        timeout=80
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Validate that we have geometry data
                        if not data.get('geometry') or len(data['geometry']) < 2:
                            st.error(" Route generated but no valid path found. Try a different location or distance.")
                            st.session_state.route_data = None
                        else:
                            # Store in session state
                            st.session_state.route_data = data
                        
                    else:
                        st.error(f" API Error ({response.status_code}): {response.text}")
                        st.session_state.route_data = None

                except requests.exceptions.Timeout:
                    st.error(" Request timed out. Try a shorter route or check if the API is running.")
                    st.session_state.route_data = None
                except requests.exceptions.ConnectionError:
                    st.error(" Cannot connect to API. Make sure FastAPI is running on " + api_url)
                    st.session_state.route_data = None
                except Exception as e:
                    st.error(f" Unexpected error: {str(e)}")
                    st.session_state.route_data = None
    
    # Display route if it exists in session state
    if st.session_state.route_data:
        data = st.session_state.route_data
        
        dist_km = data['distance_meters'] / 1000
        coords = [(p['lat'], p['lon']) for p in data['geometry']]

        # Display Metrics
        st.markdown("#### 📊 Route Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Distance", f"{dist_km:.2f} km", f"{len(coords)} waypoints")
        col2.metric("Safety Score", f"{data['safety_score']*100:.0f}%")
        col3.metric("Route Type", data.get('route_type', 'unknown').replace('_', ' ').title())
        
        # Show terrain from intent if available
        intent_data = data.get('intent', {})
        terrain_label = intent_data.get('terrain', 'balanced').title()
        col4.metric("Terrain", terrain_label)

        # Render the Map
        st.markdown("####  Your Optimized Route")
        
        # Show route type badge
        route_type = data.get('route_type', 'unknown')
        if route_type == 'round_trip':
            st.info(" **Round Trip Route** - Returns to starting point")
        elif route_type == 'point_to_point':
            st.info(" **Point-to-Point Route** - One-way journey")
        
        # Create a Folium map centered on the start point
        center_lat = sum(c[0] for c in coords) / len(coords)
        center_lon = sum(c[1] for c in coords) / len(coords)
        
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=14, 
            tiles="OpenStreetMap"
        )
        
        # Draw the path in Strava Orange
        folium.PolyLine(
            coords, 
            color="#FC4C02", 
            weight=5, 
            opacity=0.8,
            tooltip=f"{dist_km:.2f}km route"
        ).add_to(m)
        
        # Add Start/End Markers
        folium.Marker(
            coords[0], 
            tooltip="🟢 Start", 
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        
        folium.Marker(
            coords[-1], 
            tooltip="🔴 Finish", 
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)

        # Display map
        folium_static(m, width=None, height=500)
        
        # Show raw coordinates in expander
        with st.expander("📍 View Route Details"):
            st.markdown("**Parsed Intent:**")
            if intent_data:
                st.json(intent_data)
            else:
                st.caption("No intent data available")
            
            st.markdown("**Route Coordinates (first 10 points):**")
            st.json(data['geometry'][:10])
            if len(data['geometry']) > 10:
                st.caption(f"... and {len(data['geometry']) - 10} more points")

# TAB 2: Manual route planning
with tab2:
    st.markdown("#### Point-to-Point Route Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📍 Start Location**")
        start_lat = st.number_input("Latitude", value=52.5200, format="%.6f", key="start_lat")
        start_lon = st.number_input("Longitude", value=13.4050, format="%.6f", key="start_lon")
    
    with col2:
        st.markdown("**🎯 End Location**")
        end_lat = st.number_input("Latitude", value=52.5230, format="%.6f", key="end_lat")
        end_lon = st.number_input("Longitude", value=13.4100, format="%.6f", key="end_lon")
    
    preference = st.selectbox(
        "Route Preference",
        ["balanced", "hilly", "trail"],
        help="Balanced: shortest path | Hilly/Trail: optimized for terrain"
    )
    
    if st.button("🎯 Calculate Route", type="primary", key="manual_calculate"):
        with st.spinner("Calculating optimal path..."):
            try:
                response = requests.post(
                    f"{api_url}/v1/route",
                    json={
                        "start_lat": start_lat,
                        "start_lon": start_lon,
                        "end_lat": end_lat,
                        "end_lon": end_lon,
                        "preference": preference
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validate geometry
                    if not data.get('geometry') or len(data['geometry']) < 2:
                        st.error(" No valid route found between these points.")
                        st.session_state.manual_route_data = None
                    else:
                        # Store in session state
                        st.session_state.manual_route_data = data
                    
                else:
                    st.error(f"API Error: {response.text}")
                    st.session_state.manual_route_data = None
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.manual_route_data = None
    
    # Display manual route if it exists
    if st.session_state.manual_route_data:
        data = st.session_state.manual_route_data
        
        dist_km = data['distance_meters'] / 1000
        coords = [(p['lat'], p['lon']) for p in data['geometry']]

        # Metrics
        st.markdown("#### 📊 Route Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Distance", f"{dist_km:.2f} km")
        col2.metric("Safety Score", f"{data['safety_score']*100:.0f}%")
        col3.metric("Route Type", data.get('route_type', 'point_to_point').replace('_', ' ').title())
        col4.metric("Waypoints", len(coords))

        # Map
        st.markdown("#### 🗺️ Route Map")
        
        m = folium.Map(
            location=coords[0], 
            zoom_start=14,
            tiles="OpenStreetMap"
        )
        
        folium.PolyLine(
            coords, 
            color="#FC4C02",
            weight=5,
            opacity=0.8,
            tooltip=f"{dist_km:.2f}km - {preference} route"
        ).add_to(m)
        
        folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

        folium_static(m, width=None, height=500, use_container_width=True, returned_objects=[])

# Footer
st.markdown("---")
st.caption("🏃‍♂️ TrailSense AI - Built for Berlin Athletes | Powered by OSMnx, Gemini 2.5, and Random Forest ML")
