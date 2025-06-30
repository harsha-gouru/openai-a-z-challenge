#!/usr/bin/env python
"""
Amazon Deep Insights - Streamlit Visualization App

This module provides a Streamlit web application for visualizing LiDAR data
and interacting with the RAG system. It includes:
- Interactive maps for exploring LiDAR data and archaeological sites
- Chat interface for querying the knowledge base
- Data visualization tools for terrain analysis
- File upload for analyzing custom LiDAR data
"""

import os
import sys
import json
import logging
import time
import io
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pydeck as pdk
from PIL import Image
import rasterio
from rasterio.plot import show
from dotenv import load_dotenv

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import local modules (with error handling for streamlit's multiple execution model)
try:
    from src.preprocessing.lidar_processing import (
        validate_lidar_file, get_lidar_info, process_lidar_tile
    )
except ImportError:
    # For development mode when running directly
    pass

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amazon_insights.visualization.app")

# Default settings
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8080")
DEFAULT_MAP_CENTER = [float(os.environ.get("DEFAULT_LAT", "-3.4653")), 
                     float(os.environ.get("DEFAULT_LON", "-62.2159"))]
DEFAULT_ZOOM = int(os.environ.get("DEFAULT_ZOOM", "10"))
DEFAULT_MAP_STYLE = os.environ.get("MAP_STYLE", "mapbox://styles/mapbox/satellite-v9")
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")

# App configuration
st.set_page_config(
    page_title="Amazon Deep Insights",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f7ff;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        padding: 10px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #ffffcc;
        border-left: 6px solid #ffeb3b;
        padding: 10px;
        margin-bottom: 15px;
    }
    .error-box {
        background-color: #ffdddd;
        border-left: 6px solid #f44336;
        padding: 10px;
        margin-bottom: 15px;
    }
    .success-box {
        background-color: #ddffdd;
        border-left: 6px solid #4CAF50;
        padding: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "lidar_files" not in st.session_state:
    st.session_state.lidar_files = {}
if "current_view" not in st.session_state:
    st.session_state.current_view = "Home"
if "map_center" not in st.session_state:
    st.session_state.map_center = DEFAULT_MAP_CENTER
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = DEFAULT_ZOOM
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"
if "collections" not in st.session_state:
    st.session_state.collections = []
if "current_collection" not in st.session_state:
    st.session_state.current_collection = "amazon_insights"

# Helper functions
def check_api_health(api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Check the health of the RAG API."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        response.raise_for_status()
        st.session_state.api_status = "healthy"
        data = response.json()
        st.session_state.collections = data.get("collections", [])
        if st.session_state.collections and not st.session_state.current_collection in st.session_state.collections:
            st.session_state.current_collection = st.session_state.collections[0]
        return data
    except Exception as e:
        st.session_state.api_status = "unhealthy"
        logger.error(f"API health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

def query_rag_api(query: str, collection_name: str = None, top_k: int = 5) -> Dict[str, Any]:
    """Query the RAG API for an answer."""
    if not collection_name and st.session_state.collections:
        collection_name = st.session_state.current_collection
    
    try:
        response = requests.post(
            f"{DEFAULT_API_URL}/generate",
            json={
                "query": query,
                "collection_name": collection_name,
                "top_k": top_k,
                "include_context": True
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"RAG API query failed: {e}")
        return {"error": str(e)}

def display_chat_message(message: str, is_user: bool = False):
    """Display a chat message with styling."""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user">
            <img class="avatar" src="https://avataaars.io/?avatarStyle=Circle&topType=ShortHairShortFlat&accessoriesType=Blank&hairColor=BrownDark&facialHairType=Blank&clotheType=Hoodie&clotheColor=Blue&eyeType=Default&eyebrowType=Default&mouthType=Default&skinColor=Light" alt="User Avatar">
            <div class="message">{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <img class="avatar" src="https://avataaars.io/?avatarStyle=Circle&topType=WinterHat2&accessoriesType=Round&hatColor=Green&facialHairType=Blank&clotheType=Overall&clotheColor=Gray&eyeType=Happy&eyebrowType=Default&mouthType=Smile&skinColor=Light" alt="Bot Avatar">
            <div class="message">{message}</div>
        </div>
        """, unsafe_allow_html=True)

def create_base_map(center=None, zoom=None, tiles="OpenStreetMap"):
    """Create a base Folium map."""
    if center is None:
        center = st.session_state.map_center
    if zoom is None:
        zoom = st.session_state.map_zoom
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=tiles
    )
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add measure tool
    plugins.MeasureControl(position='bottomleft', primary_length_unit='meters').add_to(m)
    
    return m

def display_raster(raster_path: str, colormap: str = "viridis", title: str = None):
    """Display a raster using matplotlib."""
    try:
        with rasterio.open(raster_path) as src:
            fig, ax = plt.subplots(figsize=(10, 8))
            image = src.read(1, masked=True)
            show(image, ax=ax, cmap=colormap, title=title or os.path.basename(raster_path))
            plt.colorbar(ax.images[0], ax=ax, shrink=0.8)
            st.pyplot(fig)
            
            # Display statistics
            valid_data = image[~image.mask] if hasattr(image, 'mask') else image
            stats = {
                "Min": float(np.min(valid_data)),
                "Max": float(np.max(valid_data)),
                "Mean": float(np.mean(valid_data)),
                "Median": float(np.median(valid_data)),
                "Std Dev": float(np.std(valid_data))
            }
            return stats
    except Exception as e:
        st.error(f"Error displaying raster: {e}")
        return {}

def process_uploaded_lidar(uploaded_file):
    """Process an uploaded LiDAR file."""
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Validate the file
        if not validate_lidar_file(tmp_path):
            st.error("Invalid LiDAR file. Please upload a valid .las or .laz file.")
            os.unlink(tmp_path)
            return None
        
        # Get file info
        file_info = get_lidar_info(tmp_path)
        
        # Process the file
        output_dir = tempfile.mkdtemp()
        results = process_lidar_tile(
            tmp_path,
            output_dir=output_dir,
            resolution=1.0,
            generate_products=["dem", "dsm", "chm"]
        )
        
        # Store results in session state
        file_key = f"file_{len(st.session_state.lidar_files) + 1}"
        st.session_state.lidar_files[file_key] = {
            "path": tmp_path,
            "info": file_info,
            "results": results,
            "output_dir": output_dir,
            "name": uploaded_file.name,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return file_key
    except Exception as e:
        st.error(f"Error processing LiDAR file: {e}")
        return None

def create_3d_terrain_view(dem_path: str, colormap: str = "terrain"):
    """Create a 3D terrain view using PyDeck."""
    try:
        with rasterio.open(dem_path) as src:
            # Read the data
            elevation = src.read(1)
            
            # Get bounds and transform
            bounds = src.bounds
            transform = src.transform
            
            # Create a grid of coordinates
            height, width = elevation.shape
            x_coords = np.arange(width)
            y_coords = np.arange(height)
            xx, yy = np.meshgrid(x_coords, y_coords)
            
            # Convert pixel coordinates to world coordinates
            lons, lats = rasterio.transform.xy(transform, yy.flatten(), xx.flatten())
            
            # Create a dataframe for PyDeck
            df = pd.DataFrame({
                'lon': lons,
                'lat': lats,
                'elevation': elevation.flatten()
            })
            
            # Filter out nodata values
            if src.nodata is not None:
                df = df[df['elevation'] != src.nodata]
            
            # Normalize elevation for color
            min_elev = df['elevation'].min()
            max_elev = df['elevation'].max()
            df['elevation_norm'] = (df['elevation'] - min_elev) / (max_elev - min_elev)
            
            # Create PyDeck layer
            terrain_layer = pdk.Layer(
                'HexagonLayer',
                data=df,
                get_position=['lon', 'lat'],
                get_elevation='elevation',
                elevation_scale=50,
                elevation_range=[min_elev, max_elev],
                extruded=True,
                coverage=1,
                get_fill_color=['255 * elevation_norm', '120 * (1 - elevation_norm)', '0', '200'],
                pickable=True
            )
            
            # Create PyDeck view
            view_state = pdk.ViewState(
                longitude=np.mean(lons),
                latitude=np.mean(lats),
                zoom=12,
                pitch=45
            )
            
            # Create PyDeck deck
            deck = pdk.Deck(
                layers=[terrain_layer],
                initial_view_state=view_state,
                map_style='mapbox://styles/mapbox/satellite-v9',
                mapbox_key=MAPBOX_TOKEN
            )
            
            return deck
    except Exception as e:
        st.error(f"Error creating 3D terrain view: {e}")
        return None

def display_known_archaeological_sites():
    """Display known archaeological sites on a map."""
    # Sample data for demonstration
    sites = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Geoglyph Site 1",
                    "type": "Geometric earthwork",
                    "discovered": 2009
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-67.4, -9.8]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "Geoglyph Site 2",
                    "type": "Circular earthwork",
                    "discovered": 2016
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-67.5, -9.9]
                }
            },
            {
                "type": "Feature",
                "properties": {
                    "name": "Terra Preta Site",
                    "type": "Dark earth settlement",
                    "discovered": 2005
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [-62.2, -3.4]
                }
            }
        ]
    }
    
    # Create map
    m = create_base_map(center=[-8.0, -65.0], zoom=5, tiles="CartoDB positron")
    
    # Add sites to map
    for feature in sites["features"]:
        coords = feature["geometry"]["coordinates"]
        props = feature["properties"]
        popup_text = f"""
        <b>{props['name']}</b><br>
        Type: {props['type']}<br>
        Discovered: {props['discovered']}
        """
        folium.Marker(
            location=[coords[1], coords[0]],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=props['name'],
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)
    
    # Display map
    st_folium(m, width=800, height=600)

# App layout and navigation
def sidebar():
    """Create the sidebar navigation."""
    st.sidebar.title("Amazon Deep Insights")
    st.sidebar.image("https://i.imgur.com/8PgZnRG.png", use_column_width=True)
    
    # Navigation
    st.sidebar.header("Navigation")
    views = ["Home", "Chat", "LiDAR Visualization", "Archaeological Sites", "Upload Data", "About"]
    selected_view = st.sidebar.radio("Go to", views, index=views.index(st.session_state.current_view))
    
    if selected_view != st.session_state.current_view:
        st.session_state.current_view = selected_view
    
    # API Status
    st.sidebar.header("System Status")
    api_health = check_api_health()
    
    if st.session_state.api_status == "healthy":
        st.sidebar.success(f"API: ✅ Online")
    else:
        st.sidebar.error(f"API: ❌ Offline")
    
    # Collections
    if st.session_state.collections:
        st.sidebar.header("Collections")
        st.session_state.current_collection = st.sidebar.selectbox(
            "Select Collection",
            st.session_state.collections,
            index=st.session_state.collections.index(st.session_state.current_collection) if st.session_state.current_collection in st.session_state.collections else 0
        )
    
    # Settings
    st.sidebar.header("Settings")
    api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
    if api_url != DEFAULT_API_URL:
        st.sidebar.button("Update API URL", on_click=lambda: setattr(st.session_state, "api_url", api_url))
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2023 Amazon Deep Insights")

def home_view():
    """Display the home view."""
    st.title("🌳 Amazon Deep Insights")
    st.subheader("Exploring the Amazon Rainforest with LiDAR & AI")
    
    st.markdown("""
    Welcome to the **Amazon Deep Insights** platform! This tool combines high-resolution LiDAR data
    with AI to help researchers explore and understand the Amazon rainforest's ecological complexity
    and hidden archaeological features.
    
    ### What You Can Do Here
    
    * **Chat with the Knowledge Base**: Ask questions about the Amazon rainforest, LiDAR technology,
      and archaeological discoveries using our RAG-powered AI assistant.
    
    * **Visualize LiDAR Data**: Explore Digital Elevation Models (DEMs), Canopy Height Models (CHMs),
      and other derived products from LiDAR scans.
    
    * **Discover Archaeological Sites**: View known archaeological sites and explore potential
      new discoveries identified through LiDAR analysis.
    
    * **Upload Your Own Data**: Process and analyze your own LiDAR data using our tools.
    
    ### Getting Started
    
    Select a view from the sidebar to begin exploring the data.
    """)
    
    # Display key stats
    st.header("Key Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("LiDAR Coverage", "1.2M km²", "5%")
    
    with col2:
        st.metric("Archaeological Sites", "458", "+24")
    
    with col3:
        st.metric("Giant Trees Detected", "12,547", "+102")
    
    # Featured visualizations
    st.header("Featured Visualizations")
    tab1, tab2 = st.tabs(["Canopy Height", "Archaeological Sites"])
    
    with tab1:
        st.image("https://i.imgur.com/JLCBQpY.jpg", caption="Sample Canopy Height Model", use_column_width=True)
    
    with tab2:
        st.image("https://i.imgur.com/V5DTTYq.jpg", caption="LiDAR-revealed Geoglyphs", use_column_width=True)

def chat_view():
    """Display the chat interface for RAG queries."""
    st.title("💬 Chat with Amazon Deep Insights")
    
    st.markdown("""
    Ask questions about the Amazon rainforest, LiDAR data, archaeological sites, and more.
    Our AI assistant will retrieve relevant information from the knowledge base and provide answers.
    """)
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message["content"], message["is_user"])
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your question:", height=100, max_chars=500)
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.form_submit_button("Send")
        with col2:
            st.markdown("Examples: *What are geoglyphs?* | *How tall can trees grow in the Amazon?* | *What is Terra Preta?*")
    
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"content": user_input, "is_user": True})
        
        # Display user message
        display_chat_message(user_input, is_user=True)
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = query_rag_api(user_input, st.session_state.current_collection)
            
            if "error" in response:
                ai_message = f"Sorry, I encountered an error: {response['error']}"
            else:
                ai_message = response.get("answer", "Sorry, I couldn't generate an answer.")
                
                # Add context sources if available
                contexts = response.get("contexts", [])
                if contexts:
                    sources = []
                    for i, context in enumerate(contexts[:3], 1):
                        metadata = context.get("metadata", {})
                        source = metadata.get("source", "Unknown source")
                        if isinstance(source, str) and source != "Unknown source":
                            sources.append(f"[{i}] {os.path.basename(source)}")
                    
                    if sources:
                        ai_message += "\n\n**Sources:**\n" + "\n".join(sources)
        
        # Add AI message to chat history
        st.session_state.chat_history.append({"content": ai_message, "is_user": False})
        
        # Display AI message
        display_chat_message(ai_message, is_user=False)
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

def lidar_visualization_view():
    """Display the LiDAR visualization view."""
    st.title("🗺️ LiDAR Visualization")
    
    # Check if there are any processed LiDAR files
    if not st.session_state.lidar_files:
        st.info("No LiDAR data available. Please upload data in the 'Upload Data' section.")
        if st.button("Go to Upload Data"):
            st.session_state.current_view = "Upload Data"
            st.experimental_rerun()
        
        # Show sample visualization
        st.subheader("Sample Visualization")
        st.image("https://i.imgur.com/JLCBQpY.jpg", caption="Sample Canopy Height Model", use_column_width=True)
        return
    
    # Select LiDAR file
    file_options = {info["name"]: key for key, info in st.session_state.lidar_files.items()}
    selected_file_name = st.selectbox("Select LiDAR File", list(file_options.keys()))
    selected_file_key = file_options[selected_file_name]
    file_data = st.session_state.lidar_files[selected_file_key]
    
    # Display file info
    st.subheader("File Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**File:** {file_data['name']}")
        st.write(f"**Upload Time:** {file_data['upload_time']}")
        st.write(f"**Point Count:** {file_data['info']['point_count']:,}")
    
    with col2:
        bounds = file_data['info']['bounds']
        st.write(f"**X Range:** {bounds['minx']:.2f} to {bounds['maxx']:.2f}")
        st.write(f"**Y Range:** {bounds['miny']:.2f} to {bounds['maxy']:.2f}")
        st.write(f"**Z Range:** {bounds['minz']:.2f} to {bounds['maxz']:.2f}")
    
    # Display visualizations
    st.subheader("Visualizations")
    tabs = st.tabs(["Digital Elevation Model", "Canopy Height Model", "3D View"])
    
    with tabs[0]:
        if "dem" in file_data["results"]:
            dem_path = file_data["results"]["dem"]
            stats = display_raster(dem_path, colormap="terrain", title="Digital Elevation Model (DEM)")
            
            # Display statistics
            st.subheader("Elevation Statistics")
            st.write(f"**Min:** {stats.get('Min', 'N/A'):.2f} m")
            st.write(f"**Max:** {stats.get('Max', 'N/A'):.2f} m")
            st.write(f"**Mean:** {stats.get('Mean', 'N/A'):.2f} m")
            st.write(f"**Median:** {stats.get('Median', 'N/A'):.2f} m")
        else:
            st.warning("No DEM available for this file.")
    
    with tabs[1]:
        if "chm" in file_data["results"]:
            chm_path = file_data["results"]["chm"]
            stats = display_raster(chm_path, colormap="viridis", title="Canopy Height Model (CHM)")
            
            # Display statistics
            st.subheader("Canopy Height Statistics")
            st.write(f"**Min:** {stats.get('Min', 'N/A'):.2f} m")
            st.write(f"**Max:** {stats.get('Max', 'N/A'):.2f} m")
            st.write(f"**Mean:** {stats.get('Mean', 'N/A'):.2f} m")
            st.write(f"**Median:** {stats.get('Median', 'N/A'):.2f} m")
            
            # Height distribution
            if stats:
                with rasterio.open(chm_path) as src:
                    data = src.read(1, masked=True)
                    valid_data = data[~data.mask] if hasattr(data, 'mask') else data
                    
                    fig = px.histogram(
                        x=valid_data.flatten(),
                        nbins=50,
                        labels={"x": "Height (m)"},
                        title="Canopy Height Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No CHM available for this file.")
    
    with tabs[2]:
        if "dem" in file_data["results"]:
            dem_path = file_data["results"]["dem"]
            deck = create_3d_terrain_view(dem_path)
            if deck:
                st.pydeck_chart(deck)
            else:
                st.warning("Could not create 3D view.")
        else:
            st.warning("No DEM available for 3D view.")

def archaeological_sites_view():
    """Display the archaeological sites view."""
    st.title("🏛️ Archaeological Sites")
    
    st.markdown("""
    This map shows known archaeological sites in the Amazon rainforest, including geoglyphs,
    terra preta sites, and other features identified through LiDAR scanning and traditional
    archaeological methods.
    """)
    
    # Display map of sites
    display_known_archaeological_sites()
    
    # Site information
    st.subheader("About Amazon Archaeological Sites")
    st.markdown("""
    The Amazon rainforest contains thousands of archaeological sites, many of which were only
    recently discovered thanks to LiDAR technology. These sites include:
    
    * **Geoglyphs**: Large geometric earthworks, often only visible from the air
    * **Terra Preta**: Patches of dark, fertile soil created by ancient civilizations
    * **Raised Fields**: Agricultural systems built to manage water and soil fertility
    * **Settlement Mounds**: Elevated areas where ancient communities lived
    
    LiDAR has revolutionized Amazonian archaeology by allowing researchers to "see through"
    the dense canopy and identify human-made structures that would otherwise remain hidden.
    """)
    
    # Featured sites
    st.subheader("Featured Sites")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://i.imgur.com/V5DTTYq.jpg", caption="Geometric Geoglyphs in Acre, Brazil")
        st.markdown("""
        **Geometric Geoglyphs (Acre, Brazil)**
        
        These massive geometric earthworks were constructed between 2,000 and 650 years ago.
        They include squares, circles, and octagons, some spanning over 300 meters in diameter.
        Their purpose remains debated - they may have been ceremonial centers, defensive
        structures, or astronomical markers.
        """)
    
    with col2:
        st.image("https://i.imgur.com/8PgZnRG.png", caption="Terra Preta Site near Santarém")
        st.markdown("""
        **Terra Preta Site (Santarém, Brazil)**
        
        Terra Preta ("black earth" in Portuguese) is a type of very dark, fertile anthropogenic
        soil found throughout the Amazon Basin. Created by ancient indigenous populations,
        these soils are rich in charcoal, organic matter, and nutrients, demonstrating
        sophisticated agricultural knowledge and long-term settlement.
        """)

def upload_data_view():
    """Display the data upload view."""
    st.title("📤 Upload Data")
    
    st.markdown("""
    Upload your own LiDAR data for processing and visualization. The system supports
    .las and .laz files and will automatically generate Digital Elevation Models (DEMs),
    Digital Surface Models (DSMs), and Canopy Height Models (CHMs).
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload LiDAR File", type=["las", "laz"])
    
    if uploaded_file:
        with st.spinner("Processing LiDAR file..."):
            file_key = process_uploaded_lidar(uploaded_file)
            
            if file_key:
                st.success(f"File {uploaded_file.name} processed successfully!")
                
                # Display quick preview
                file_data = st.session_state.lidar_files[file_key]
                st.subheader("File Information")
                st.write(f"**File:** {file_data['name']}")
                st.write(f"**Point Count:** {file_data['info']['point_count']:,}")
                
                # Show visualization button
                if st.button("View Visualizations"):
                    st.session_state.current_view = "LiDAR Visualization"
                    st.experimental_rerun()
    
    # List processed files
    if st.session_state.lidar_files:
        st.subheader("Processed Files")
        
        for key, file_data in st.session_state.lidar_files.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{file_data['name']}** ({file_data['upload_time']})")
            
            with col2:
                if st.button(f"View", key=f"view_{key}"):
                    st.session_state.current_view = "LiDAR Visualization"
                    st.experimental_rerun()
            
            with col3:
                if st.button(f"Delete", key=f"delete_{key}"):
                    # Clean up temporary files
                    try:
                        os.unlink(file_data['path'])
                        for result_path in file_data['results'].values():
                            if os.path.exists(result_path):
                                os.unlink(result_path)
                        if os.path.exists(file_data['output_dir']):
                            import shutil
                            shutil.rmtree(file_data['output_dir'])
                    except Exception as e:
                        logger.error(f"Error cleaning up files: {e}")
                    
                    # Remove from session state
                    del st.session_state.lidar_files[key]
                    st.experimental_rerun()

def about_view():
    """Display the about view."""
    st.title("ℹ️ About Amazon Deep Insights")
    
    st.markdown("""
    **Amazon Deep Insights** is a project developed for the OpenAI A-to-Z Challenge. It combines
    LiDAR data processing, geospatial analysis, and Retrieval-Augmented Generation (RAG) to help
    researchers explore and understand the Amazon rainforest.
    
    ### Project Goals
    
    1. **Unify Diverse Datasets**: Integrate LiDAR point clouds, remote sensing imagery, and
       scientific literature into a searchable knowledge base.
    
    2. **Accelerate Discovery**: Identify potential archaeological sites and ecological features
       that would be difficult to detect through traditional methods.
    
    3. **Democratize Access**: Make complex geospatial data accessible through natural language
       queries and intuitive visualizations.
    
    ### Technologies Used
    
    * **LiDAR Processing**: PDAL, GDAL, laspy
    * **Geospatial Analysis**: Rasterio, GeoPandas, Shapely
    * **Machine Learning**: scikit-learn, XGBoost
    * **RAG System**: OpenAI Embeddings, ChromaDB, LangChain
    * **Visualization**: Streamlit, Folium, PyDeck, Plotly
    
    ### Data Sources
    
    The project uses publicly available datasets from:
    
    * ORNL DAAC LiDAR Forest Inventory
    * OpenTopography
    * Zenodo repositories
    * Published research articles
    
    ### Team
    
    This project was developed by the OpenAI A-to-Z Challenge team.
    """)
    
    # Contact information
    st.subheader("Contact")
    st.markdown("""
    For more information or to contribute to the project, please visit our GitHub repository:
    [github.com/harsha-gouru/openai-a-z-challenge](https://github.com/harsha-gouru/openai-a-z-challenge)
    """)
    
    # Acknowledgements
    st.subheader("Acknowledgements")
    st.markdown("""
    We would like to thank OpenAI for organizing the A-to-Z Challenge and providing the
    opportunity to work on this project. We also thank the researchers and organizations
    who have made their data publicly available.
    """)

# Main app function
def main():
    """Main application function."""
    # Display sidebar
    sidebar()
    
    # Display the selected view
    if st.session_state.current_view == "Home":
        home_view()
    elif st.session_state.current_view == "Chat":
        chat_view()
    elif st.session_state.current_view == "LiDAR Visualization":
        lidar_visualization_view()
    elif st.session_state.current_view == "Archaeological Sites":
        archaeological_sites_view()
    elif st.session_state.current_view == "Upload Data":
        upload_data_view()
    elif st.session_state.current_view == "About":
        about_view()

if __name__ == "__main__":
    main()
