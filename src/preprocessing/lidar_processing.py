#!/usr/bin/env python
"""
Amazon Deep Insights - LiDAR Processing Module

This module provides functions to preprocess LiDAR data, including:
- Loading and validating LiDAR files
- Generating Digital Elevation Models (DEMs)
- Generating Digital Surface Models (DSMs)
- Creating Canopy Height Models (CHMs)
- Filtering point clouds
- Running custom PDAL pipelines

The module is designed to work with LAS/LAZ files from various sources
and supports batch processing of multiple tiles.
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import pdal
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Polygon, MultiPolygon
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amazon_insights.preprocessing.lidar_processing")

# Load environment variables
load_dotenv()

# Default paths
DEFAULT_PROCESSED_DIR = os.environ.get(
    "PROCESSED_DATA_DIR", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
)


class LidarProcessingError(Exception):
    """Exception raised for errors during LiDAR processing."""
    pass


def validate_lidar_file(file_path: str) -> bool:
    """
    Validate a LiDAR file by checking if it can be opened and read.
    
    Args:
        file_path (str): Path to the LiDAR file (.las or .laz)
        
    Returns:
        bool: True if the file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Check file extension
    if not file_path.lower().endswith(('.las', '.laz')):
        logger.error(f"Invalid file extension: {file_path}")
        return False
    
    try:
        # Try to read the file using PDAL
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [
                file_path,
                {
                    "type": "filters.stats"
                }
            ]
        }))
        
        pipeline.execute()
        metadata = pipeline.metadata
        
        # Check if we got valid metadata
        if not metadata or 'metadata' not in metadata:
            logger.error(f"Failed to read metadata from {file_path}")
            return False
        
        # Check if the file has points
        stats = json.loads(pipeline.metadata)['metadata']['filters.stats']
        point_count = stats['statistic'][0]['count']
        
        if point_count == 0:
            logger.warning(f"LiDAR file contains no points: {file_path}")
            return False
        
        logger.info(f"Valid LiDAR file: {file_path} ({point_count} points)")
        return True
    
    except Exception as e:
        logger.error(f"Error validating LiDAR file {file_path}: {e}")
        return False


def get_lidar_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a LiDAR file including bounds, point count, and classification.
    
    Args:
        file_path (str): Path to the LiDAR file (.las or .laz)
        
    Returns:
        Dict[str, Any]: Dictionary with LiDAR file information
        
    Raises:
        LidarProcessingError: If the file cannot be read
    """
    if not os.path.exists(file_path):
        raise LidarProcessingError(f"File not found: {file_path}")
    
    try:
        # Use PDAL to get file info
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [
                file_path,
                {
                    "type": "filters.info"
                }
            ]
        }))
        
        pipeline.execute()
        metadata = json.loads(pipeline.metadata)
        
        # Extract relevant information
        info = metadata['metadata']['filters.info']
        
        # Get bounds
        bounds = info['bounds']
        
        # Get point count
        point_count = info['num_points']
        
        # Get coordinate system
        srs = info.get('srs', {}).get('wkt', 'Unknown')
        
        # Get dimensions
        dimensions = info.get('dimensions', [])
        
        # Check if classification is available
        has_classification = 'Classification' in [dim['name'] for dim in dimensions]
        
        # Get classification counts if available
        classification_counts = {}
        if has_classification:
            try:
                # Run a stats filter to get classification counts
                stats_pipeline = pdal.Pipeline(json.dumps({
                    "pipeline": [
                        file_path,
                        {
                            "type": "filters.stats",
                            "dimensions": "Classification"
                        }
                    ]
                }))
                
                stats_pipeline.execute()
                stats_metadata = json.loads(stats_pipeline.metadata)
                
                # Extract classification counts
                stats = stats_metadata['metadata']['filters.stats']
                for stat in stats['statistic']:
                    if stat['name'] == 'Classification':
                        for count in stat['counts']:
                            classification_counts[count['value']] = count['count']
            
            except Exception as e:
                logger.warning(f"Failed to get classification counts: {e}")
        
        # Format the result
        result = {
            "file_path": file_path,
            "point_count": point_count,
            "bounds": {
                "minx": bounds['minx'],
                "miny": bounds['miny'],
                "minz": bounds['minz'],
                "maxx": bounds['maxx'],
                "maxy": bounds['maxy'],
                "maxz": bounds['maxz']
            },
            "srs": srs,
            "dimensions": [dim['name'] for dim in dimensions],
            "has_classification": has_classification,
            "classification_counts": classification_counts
        }
        
        return result
    
    except Exception as e:
        raise LidarProcessingError(f"Error getting LiDAR info for {file_path}: {e}")


def create_dem_pipeline(input_file: str, output_file: str, resolution: float = 1.0,
                       bounds: Optional[Dict[str, float]] = None, 
                       output_type: str = "idw", window_size: int = 10) -> Dict:
    """
    Create a PDAL pipeline definition for generating a Digital Elevation Model (DEM).
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output DEM file (GeoTIFF)
        resolution (float): Output resolution in the same units as input data (default: 1.0)
        bounds (Optional[Dict[str, float]]): Optional bounds dictionary with minx, miny, maxx, maxy
        output_type (str): Interpolation method ('idw', 'mean', 'min', 'max')
        window_size (int): Window size for interpolation
        
    Returns:
        Dict: PDAL pipeline definition as a dictionary
    """
    # Create the pipeline stages
    pipeline_stages = [
        input_file,
        {
            "type": "filters.assign",
            "assignment": "Classification[:]=0"
        },
        {
            "type": "filters.elm"
        },
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": 8,
            "multiplier": 3.0
        },
        {
            "type": "filters.pmf"
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        }
    ]
    
    # Add writer with bounds if provided
    writer_stage = {
        "type": "writers.gdal",
        "filename": output_file,
        "output_type": output_type,
        "resolution": resolution,
        "window_size": window_size
    }
    
    if bounds:
        writer_stage["bounds"] = f"([{bounds['minx']}, {bounds['maxx']}], [{bounds['miny']}, {bounds['maxy']}])"
    
    pipeline_stages.append(writer_stage)
    
    return {"pipeline": pipeline_stages}


def create_dsm_pipeline(input_file: str, output_file: str, resolution: float = 1.0,
                       bounds: Optional[Dict[str, float]] = None,
                       output_type: str = "max", window_size: int = 10) -> Dict:
    """
    Create a PDAL pipeline definition for generating a Digital Surface Model (DSM).
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output DSM file (GeoTIFF)
        resolution (float): Output resolution in the same units as input data (default: 1.0)
        bounds (Optional[Dict[str, float]]): Optional bounds dictionary with minx, miny, maxx, maxy
        output_type (str): Interpolation method ('max' recommended for DSM)
        window_size (int): Window size for interpolation
        
    Returns:
        Dict: PDAL pipeline definition as a dictionary
    """
    # Create the pipeline stages
    pipeline_stages = [
        input_file,
        {
            "type": "filters.outlier",
            "method": "statistical",
            "mean_k": 8,
            "multiplier": 3.0
        }
    ]
    
    # Add writer with bounds if provided
    writer_stage = {
        "type": "writers.gdal",
        "filename": output_file,
        "output_type": output_type,
        "resolution": resolution,
        "window_size": window_size
    }
    
    if bounds:
        writer_stage["bounds"] = f"([{bounds['minx']}, {bounds['maxx']}], [{bounds['miny']}, {bounds['maxy']}])"
    
    pipeline_stages.append(writer_stage)
    
    return {"pipeline": pipeline_stages}


def create_normalized_height_pipeline(input_file: str, output_file: str) -> Dict:
    """
    Create a PDAL pipeline definition for normalizing point heights (z) relative to the ground.
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output normalized LiDAR file
        
    Returns:
        Dict: PDAL pipeline definition as a dictionary
    """
    return {
        "pipeline": [
            input_file,
            {
                "type": "filters.assign",
                "assignment": "Classification[:]=0"
            },
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": 8,
                "multiplier": 3.0
            },
            {
                "type": "filters.pmf"
            },
            {
                "type": "filters.hag_nn"
            },
            {
                "type": "writers.las",
                "filename": output_file,
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
    }


def create_classification_pipeline(input_file: str, output_file: str, 
                                 ground_max_angle: float = 2.0,
                                 ground_max_window_size: float = 33.0) -> Dict:
    """
    Create a PDAL pipeline definition for classifying points (ground, vegetation, etc.).
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output classified LiDAR file
        ground_max_angle (float): Maximum angle for ground classification
        ground_max_window_size (float): Maximum window size for ground classification
        
    Returns:
        Dict: PDAL pipeline definition as a dictionary
    """
    return {
        "pipeline": [
            input_file,
            {
                "type": "filters.assign",
                "assignment": "Classification[:]=0"
            },
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": 8,
                "multiplier": 3.0
            },
            {
                "type": "filters.smrf",
                "ignore": "Classification[7:7]",
                "slope": ground_max_angle,
                "window": ground_max_window_size,
                "threshold": 0.15,
                "scalar": 1.2
            },
            {
                "type": "filters.hag_nn",
                "count": 8
            },
            {
                "type": "filters.approximatecoplanar",
                "knn": 10,
                "thresh1": 25,
                "thresh2": 6
            },
            {
                "type": "filters.csf"
            },
            {
                "type": "filters.pmf"
            },
            {
                "type": "filters.range",
                "limits": "HeightAboveGround[0.5:2.0],Coplanar[0:0.95]",
                "assignment": "Classification[:]=3"  # Low vegetation
            },
            {
                "type": "filters.range",
                "limits": "HeightAboveGround[2.0:5.0],Coplanar[0:0.95]",
                "assignment": "Classification[:]=4"  # Medium vegetation
            },
            {
                "type": "filters.range",
                "limits": "HeightAboveGround[5.0:)",
                "assignment": "Classification[:]=5"  # High vegetation
            },
            {
                "type": "writers.las",
                "filename": output_file,
                "extra_dims": "HeightAboveGround=float32,Coplanar=float32"
            }
        ]
    }


def run_pdal_pipeline(pipeline_def: Dict) -> pdal.Pipeline:
    """
    Execute a PDAL pipeline.
    
    Args:
        pipeline_def (Dict): PDAL pipeline definition as a dictionary
        
    Returns:
        pdal.Pipeline: Executed PDAL pipeline object
        
    Raises:
        LidarProcessingError: If the pipeline fails to execute
    """
    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        count = pipeline.execute()
        logger.info(f"PDAL pipeline executed successfully: {count} points processed")
        return pipeline
    
    except Exception as e:
        raise LidarProcessingError(f"Error executing PDAL pipeline: {e}")


def generate_dem(input_file: str, output_file: str, resolution: float = 1.0,
                bounds: Optional[Dict[str, float]] = None) -> str:
    """
    Generate a Digital Elevation Model (DEM) from a LiDAR file.
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output DEM file (GeoTIFF)
        resolution (float): Output resolution in the same units as input data
        bounds (Optional[Dict[str, float]]): Optional bounds dictionary with minx, miny, maxx, maxy
        
    Returns:
        str: Path to the output DEM file
        
    Raises:
        LidarProcessingError: If DEM generation fails
    """
    logger.info(f"Generating DEM from {input_file} at {resolution}m resolution")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create and run the DEM pipeline
        pipeline_def = create_dem_pipeline(
            input_file=input_file,
            output_file=output_file,
            resolution=resolution,
            bounds=bounds
        )
        
        run_pdal_pipeline(pipeline_def)
        
        # Verify the output file exists
        if not os.path.exists(output_file):
            raise LidarProcessingError(f"DEM file was not created: {output_file}")
        
        logger.info(f"DEM generated successfully: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error generating DEM: {e}")


def generate_dsm(input_file: str, output_file: str, resolution: float = 1.0,
                bounds: Optional[Dict[str, float]] = None) -> str:
    """
    Generate a Digital Surface Model (DSM) from a LiDAR file.
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output DSM file (GeoTIFF)
        resolution (float): Output resolution in the same units as input data
        bounds (Optional[Dict[str, float]]): Optional bounds dictionary with minx, miny, maxx, maxy
        
    Returns:
        str: Path to the output DSM file
        
    Raises:
        LidarProcessingError: If DSM generation fails
    """
    logger.info(f"Generating DSM from {input_file} at {resolution}m resolution")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create and run the DSM pipeline
        pipeline_def = create_dsm_pipeline(
            input_file=input_file,
            output_file=output_file,
            resolution=resolution,
            bounds=bounds
        )
        
        run_pdal_pipeline(pipeline_def)
        
        # Verify the output file exists
        if not os.path.exists(output_file):
            raise LidarProcessingError(f"DSM file was not created: {output_file}")
        
        logger.info(f"DSM generated successfully: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error generating DSM: {e}")


def generate_chm(dem_file: str, dsm_file: str, output_file: str) -> str:
    """
    Generate a Canopy Height Model (CHM) by subtracting DEM from DSM.
    
    Args:
        dem_file (str): Path to the DEM file
        dsm_file (str): Path to the DSM file
        output_file (str): Path to the output CHM file
        
    Returns:
        str: Path to the output CHM file
        
    Raises:
        LidarProcessingError: If CHM generation fails
    """
    logger.info(f"Generating CHM from DEM {dem_file} and DSM {dsm_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Open the DEM and DSM
        with rasterio.open(dem_file) as dem_src, rasterio.open(dsm_file) as dsm_src:
            # Check if they have the same dimensions and transform
            if dem_src.shape != dsm_src.shape:
                raise LidarProcessingError(
                    f"DEM and DSM have different dimensions: {dem_src.shape} vs {dsm_src.shape}"
                )
            
            # Read the data
            dem_data = dem_src.read(1, masked=True)
            dsm_data = dsm_src.read(1, masked=True)
            
            # Calculate CHM (DSM - DEM)
            chm_data = dsm_data - dem_data
            
            # Set negative values to zero (sometimes happens due to interpolation artifacts)
            chm_data = np.where(chm_data < 0, 0, chm_data)
            
            # Write CHM to file
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=dem_src.height,
                width=dem_src.width,
                count=1,
                dtype=chm_data.dtype,
                crs=dem_src.crs,
                transform=dem_src.transform,
                nodata=dem_src.nodata
            ) as chm_dst:
                chm_dst.write(chm_data, 1)
        
        logger.info(f"CHM generated successfully: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error generating CHM: {e}")


def classify_points(input_file: str, output_file: str) -> str:
    """
    Classify points in a LiDAR file (ground, vegetation, etc.).
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output classified LiDAR file
        
    Returns:
        str: Path to the output classified LiDAR file
        
    Raises:
        LidarProcessingError: If classification fails
    """
    logger.info(f"Classifying points in {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create and run the classification pipeline
        pipeline_def = create_classification_pipeline(
            input_file=input_file,
            output_file=output_file
        )
        
        run_pdal_pipeline(pipeline_def)
        
        # Verify the output file exists
        if not os.path.exists(output_file):
            raise LidarProcessingError(f"Classified file was not created: {output_file}")
        
        logger.info(f"Point classification completed: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error classifying points: {e}")


def normalize_heights(input_file: str, output_file: str) -> str:
    """
    Normalize point heights relative to the ground.
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output normalized LiDAR file
        
    Returns:
        str: Path to the output normalized LiDAR file
        
    Raises:
        LidarProcessingError: If normalization fails
    """
    logger.info(f"Normalizing heights in {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create and run the normalization pipeline
        pipeline_def = create_normalized_height_pipeline(
            input_file=input_file,
            output_file=output_file
        )
        
        run_pdal_pipeline(pipeline_def)
        
        # Verify the output file exists
        if not os.path.exists(output_file):
            raise LidarProcessingError(f"Normalized file was not created: {output_file}")
        
        logger.info(f"Height normalization completed: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error normalizing heights: {e}")


def filter_by_bounds(input_file: str, output_file: str, 
                    bounds: Dict[str, float]) -> str:
    """
    Filter a LiDAR file to include only points within specified bounds.
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output filtered LiDAR file
        bounds (Dict[str, float]): Bounds dictionary with minx, miny, maxx, maxy
        
    Returns:
        str: Path to the output filtered LiDAR file
        
    Raises:
        LidarProcessingError: If filtering fails
    """
    logger.info(f"Filtering {input_file} by bounds: {bounds}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create the bounds filter pipeline
        pipeline_def = {
            "pipeline": [
                input_file,
                {
                    "type": "filters.crop",
                    "bounds": f"([{bounds['minx']}, {bounds['maxx']}], [{bounds['miny']}, {bounds['maxy']}])"
                },
                {
                    "type": "writers.las",
                    "filename": output_file
                }
            ]
        }
        
        run_pdal_pipeline(pipeline_def)
        
        # Verify the output file exists
        if not os.path.exists(output_file):
            raise LidarProcessingError(f"Filtered file was not created: {output_file}")
        
        logger.info(f"Filtering completed: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error filtering by bounds: {e}")


def filter_by_classification(input_file: str, output_file: str, 
                           keep_classes: List[int]) -> str:
    """
    Filter a LiDAR file to include only points with specified classifications.
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output filtered LiDAR file
        keep_classes (List[int]): List of classification values to keep
        
    Returns:
        str: Path to the output filtered LiDAR file
        
    Raises:
        LidarProcessingError: If filtering fails
    """
    logger.info(f"Filtering {input_file} by classification: {keep_classes}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create the classification filter pipeline
        class_range = ",".join([f"{cls}:{cls}" for cls in keep_classes])
        
        pipeline_def = {
            "pipeline": [
                input_file,
                {
                    "type": "filters.range",
                    "limits": f"Classification[{class_range}]"
                },
                {
                    "type": "writers.las",
                    "filename": output_file
                }
            ]
        }
        
        run_pdal_pipeline(pipeline_def)
        
        # Verify the output file exists
        if not os.path.exists(output_file):
            raise LidarProcessingError(f"Filtered file was not created: {output_file}")
        
        logger.info(f"Classification filtering completed: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error filtering by classification: {e}")


def thin_points(input_file: str, output_file: str, resolution: float = 1.0) -> str:
    """
    Thin a LiDAR point cloud to reduce density.
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_file (str): Path to the output thinned LiDAR file
        resolution (float): Grid cell size for thinning
        
    Returns:
        str: Path to the output thinned LiDAR file
        
    Raises:
        LidarProcessingError: If thinning fails
    """
    logger.info(f"Thinning points in {input_file} with resolution {resolution}m")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create the thinning pipeline
        pipeline_def = {
            "pipeline": [
                input_file,
                {
                    "type": "filters.voxelgrid",
                    "leaf_x": resolution,
                    "leaf_y": resolution,
                    "leaf_z": resolution
                },
                {
                    "type": "writers.las",
                    "filename": output_file
                }
            ]
        }
        
        run_pdal_pipeline(pipeline_def)
        
        # Verify the output file exists
        if not os.path.exists(output_file):
            raise LidarProcessingError(f"Thinned file was not created: {output_file}")
        
        logger.info(f"Point thinning completed: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error thinning points: {e}")


def merge_lidar_files(input_files: List[str], output_file: str) -> str:
    """
    Merge multiple LiDAR files into a single file.
    
    Args:
        input_files (List[str]): List of input LiDAR file paths
        output_file (str): Path to the output merged LiDAR file
        
    Returns:
        str: Path to the output merged LiDAR file
        
    Raises:
        LidarProcessingError: If merging fails
    """
    if len(input_files) < 2:
        raise ValueError("At least two input files are required for merging")
    
    logger.info(f"Merging {len(input_files)} LiDAR files")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Create the merge pipeline
        pipeline_def = {
            "pipeline": [
                *input_files,
                {
                    "type": "filters.merge"
                },
                {
                    "type": "writers.las",
                    "filename": output_file
                }
            ]
        }
        
        run_pdal_pipeline(pipeline_def)
        
        # Verify the output file exists
        if not os.path.exists(output_file):
            raise LidarProcessingError(f"Merged file was not created: {output_file}")
        
        logger.info(f"Merging completed: {output_file}")
        return output_file
    
    except Exception as e:
        raise LidarProcessingError(f"Error merging LiDAR files: {e}")


def process_lidar_tile(input_file: str, output_dir: str = DEFAULT_PROCESSED_DIR,
                      resolution: float = 1.0, generate_products: List[str] = None) -> Dict[str, str]:
    """
    Process a single LiDAR tile to generate various products.
    
    Args:
        input_file (str): Path to the input LiDAR file
        output_dir (str): Base directory for output files
        resolution (float): Output resolution in the same units as input data
        generate_products (List[str]): List of products to generate ('dem', 'dsm', 'chm', 'normalized', 'classified')
        
    Returns:
        Dict[str, str]: Dictionary mapping product names to output file paths
        
    Raises:
        LidarProcessingError: If processing fails
    """
    if not os.path.exists(input_file):
        raise LidarProcessingError(f"Input file not found: {input_file}")
    
    # Default products if not specified
    if generate_products is None:
        generate_products = ['dem', 'dsm', 'chm']
    
    # Get the base name of the input file without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create output directories
    dem_dir = os.path.join(output_dir, "dem")
    dsm_dir = os.path.join(output_dir, "dsm")
    chm_dir = os.path.join(output_dir, "chm")
    las_dir = os.path.join(output_dir, "las")
    
    os.makedirs(dem_dir, exist_ok=True)
    os.makedirs(dsm_dir, exist_ok=True)
    os.makedirs(chm_dir, exist_ok=True)
    os.makedirs(las_dir, exist_ok=True)
    
    # Define output file paths
    dem_file = os.path.join(dem_dir, f"{base_name}_dem_{resolution}m.tif")
    dsm_file = os.path.join(dsm_dir, f"{base_name}_dsm_{resolution}m.tif")
    chm_file = os.path.join(chm_dir, f"{base_name}_chm_{resolution}m.tif")
    normalized_file = os.path.join(las_dir, f"{base_name}_normalized.las")
    classified_file = os.path.join(las_dir, f"{base_name}_classified.las")
    
    # Process the file
    results = {}
    
    try:
        # Generate DEM if requested
        if 'dem' in generate_products:
            dem_path = generate_dem(input_file, dem_file, resolution)
            results['dem'] = dem_path
        
        # Generate DSM if requested
        if 'dsm' in generate_products:
            dsm_path = generate_dsm(input_file, dsm_file, resolution)
            results['dsm'] = dsm_path
        
        # Generate CHM if both DEM and DSM were generated
        if 'chm' in generate_products and 'dem' in results and 'dsm' in results:
            chm_path = generate_chm(results['dem'], results['dsm'], chm_file)
            results['chm'] = chm_path
        
        # Generate normalized point cloud if requested
        if 'normalized' in generate_products:
            normalized_path = normalize_heights(input_file, normalized_file)
            results['normalized'] = normalized_path
        
        # Generate classified point cloud if requested
        if 'classified' in generate_products:
            classified_path = classify_points(input_file, classified_file)
            results['classified'] = classified_path
        
        return results
    
    except Exception as e:
        raise LidarProcessingError(f"Error processing LiDAR tile {input_file}: {e}")


def batch_process_lidar(input_dir: str, output_dir: str = DEFAULT_PROCESSED_DIR,
                       resolution: float = 1.0, generate_products: List[str] = None,
                       file_pattern: str = "*.la[sz]") -> Dict[str, Dict[str, str]]:
    """
    Process multiple LiDAR tiles in a directory.
    
    Args:
        input_dir (str): Directory containing input LiDAR files
        output_dir (str): Base directory for output files
        resolution (float): Output resolution in the same units as input data
        generate_products (List[str]): List of products to generate
        file_pattern (str): Glob pattern to match input files
        
    Returns:
        Dict[str, Dict[str, str]]: Dictionary mapping input files to their output products
        
    Raises:
        LidarProcessingError: If processing fails
    """
    # Find all LiDAR files in the input directory
    input_files = list(Path(input_dir).glob(file_pattern))
    
    if not input_files:
        logger.warning(f"No LiDAR files found in {input_dir} matching pattern {file_pattern}")
        return {}
    
    logger.info(f"Found {len(input_files)} LiDAR files to process")
    
    # Process each file
    results = {}
    
    for input_file in input_files:
        try:
            file_results = process_lidar_tile(
                str(input_file),
                output_dir=output_dir,
                resolution=resolution,
                generate_products=generate_products
            )
            
            results[str(input_file)] = file_results
            logger.info(f"Successfully processed {input_file}")
        
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
    
    return results


def main():
    """Command-line interface for LiDAR processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process LiDAR data to generate DEMs, DSMs, and CHMs")
    
    # Input/output options
    parser.add_argument("--input", "-i", required=True, help="Input LiDAR file or directory")
    parser.add_argument("--output-dir", "-o", default=DEFAULT_PROCESSED_DIR, help="Output directory")
    parser.add_argument("--resolution", "-r", type=float, default=1.0, help="Output resolution in meters")
    
    # Processing options
    parser.add_argument("--products", "-p", nargs="+", default=["dem", "dsm", "chm"],
                      choices=["dem", "dsm", "chm", "normalized", "classified"],
                      help="Products to generate")
    parser.add_argument("--batch", "-b", action="store_true", help="Process all LiDAR files in the input directory")
    parser.add_argument("--pattern", default="*.la[sz]", help="File pattern for batch processing")
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            if not os.path.isdir(args.input):
                parser.error("--batch requires input to be a directory")
            
            results = batch_process_lidar(
                args.input,
                args.output_dir,
                args.resolution,
                args.products,
                args.pattern
            )
            
            # Print summary
            print("\nProcessing Summary:")
            for input_file, products in results.items():
                print(f"\nInput: {input_file}")
                for product, output_file in products.items():
                    print(f"  - {product}: {output_file}")
        
        else:
            if not os.path.isfile(args.input):
                parser.error("Input must be a file when not using --batch")
            
            results = process_lidar_tile(
                args.input,
                args.output_dir,
                args.resolution,
                args.products
            )
            
            # Print results
            print("\nProcessing Results:")
            for product, output_file in results.items():
                print(f"  - {product}: {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
