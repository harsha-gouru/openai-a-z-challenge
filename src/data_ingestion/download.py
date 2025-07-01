#!/usr/bin/env python
"""
Amazon Deep Insights - Data Ingestion Module

This module provides functions to download LiDAR and other datasets from various sources
listed in the project's Links.md file. It supports downloading from:
- ORNL DAAC datasets
- OpenTopography
- Zenodo repositories
- Other HTTP/HTTPS sources

Usage:
    As a module:
        from amazon_insights.data_ingestion.download import download_dataset
        download_dataset("ornl", dataset_id="1644", output_dir="data/raw/lidar")
    
    From command line:
        python -m amazon_insights.data_ingestion.download --source ornl --id 1644
"""

import os
import sys
import re
import hashlib
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse, parse_qs
import time
import shutil

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("amazon_insights.data_ingestion.download")

# Default paths
DEFAULT_LINKS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Links.md")
DEFAULT_OUTPUT_DIR = os.environ.get("RAW_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw"))
DEFAULT_LIDAR_DIR = os.environ.get("LIDAR_TILES_DIR", os.path.join(DEFAULT_OUTPUT_DIR, "lidar"))
DEFAULT_RASTER_DIR = os.environ.get("RASTER_DATA_DIR", os.path.join(DEFAULT_OUTPUT_DIR, "rasters"))
DEFAULT_VECTOR_DIR = os.environ.get("VECTOR_DATA_DIR", os.path.join(DEFAULT_OUTPUT_DIR, "vectors"))
DEFAULT_TEXT_DIR = os.environ.get("TEXT_CORPUS_DIR", os.path.join(DEFAULT_OUTPUT_DIR, "corpus"))

# API keys and credentials
ORNL_DAAC_API_KEY = os.environ.get("ORNL_DAAC_API_KEY", "")
OPENTOPO_API_KEY = os.environ.get("OPENTOPO_API_KEY", "")
ZENODO_ACCESS_TOKEN = os.environ.get("ZENODO_ACCESS_TOKEN", "")


class DownloadError(Exception):
    """Exception raised for errors during download operations."""
    pass


def create_session() -> requests.Session:
    """
    Create a requests session with retry capabilities.
    
    Returns:
        requests.Session: Configured session object
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def parse_links_file(links_file: str = DEFAULT_LINKS_FILE) -> Dict[str, List[str]]:
    """
    Parse the Links.md file and categorize URLs by source.
    
    Args:
        links_file (str): Path to the Links.md file
        
    Returns:
        Dict[str, List[str]]: Dictionary of URLs categorized by source
    """
    if not os.path.exists(links_file):
        logger.error(f"Links file not found: {links_file}")
        return {}
    
    try:
        with open(links_file, 'r') as f:
            content = f.read()
        
        # Extract all URLs
        urls = re.findall(r'https?://[^\s\)]+', content)
        
        # Categorize URLs by source
        categorized_urls = {
            "ornl": [url for url in urls if "daac.ornl.gov" in url],
            "opentopo": [url for url in urls if "opentopography.org" in url],
            "zenodo": [url for url in urls if "zenodo.org" in url],
            "kaggle": [url for url in urls if "kaggle.com" in url],
            "science": [url for url in urls if "science.org" in url],
            "other": [url for url in urls if not any(
                domain in url for domain in ["daac.ornl.gov", "opentopography.org", "zenodo.org", "kaggle.com", "science.org"]
            )]
        }
        
        return categorized_urls
    
    except Exception as e:
        logger.error(f"Error parsing links file: {e}")
        return {}


def extract_dataset_id(url: str) -> Optional[str]:
    """
    Extract dataset ID from a URL.
    
    Args:
        url (str): URL to parse
        
    Returns:
        Optional[str]: Dataset ID if found, None otherwise
    """
    # Extract ORNL DAAC dataset ID
    ornl_match = re.search(r'ds_id=(\d+)', url)
    if ornl_match:
        return ornl_match.group(1)
    
    # Extract OpenTopography dataset ID
    opentopo_match = re.search(r'otCollectionID=([\w\.]+)', url)
    if opentopo_match:
        return opentopo_match.group(1)
    
    # Extract Zenodo record ID
    zenodo_match = re.search(r'records/(\d+)', url)
    if zenodo_match:
        return zenodo_match.group(1)
    
    # Extract Kaggle dataset ID
    kaggle_match = re.search(r'datasets/([\w\-/]+)', url)
    if kaggle_match:
        return kaggle_match.group(1)
    
    return None


def download_file(url: str, output_path: str, session: Optional[requests.Session] = None,
                 chunk_size: int = 8192, overwrite: bool = False) -> str:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url (str): URL to download
        output_path (str): Path to save the file
        session (Optional[requests.Session]): Requests session to use
        chunk_size (int): Size of chunks to download
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        str: Path to the downloaded file
    
    Raises:
        DownloadError: If download fails
    """
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"File already exists: {output_path}")
        return output_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use provided session or create a new one
    if session is None:
        session = create_session()
    
    try:
        # Stream download with progress bar
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                bar.update(size)
        
        # Calculate and save checksum
        checksum = calculate_checksum(output_path)
        checksum_path = f"{output_path}.sha256"
        with open(checksum_path, 'w') as f:
            f.write(f"{checksum} *{os.path.basename(output_path)}")
        
        logger.info(f"Downloaded: {output_path}")
        logger.info(f"Checksum: {checksum}")
        
        return output_path
    
    except Exception as e:
        # Clean up partial download
        if os.path.exists(output_path):
            os.remove(output_path)
        
        logger.error(f"Download failed: {e}")
        raise DownloadError(f"Failed to download {url}: {e}")


def calculate_checksum(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculate checksum for a file.
    
    Args:
        file_path (str): Path to the file
        algorithm (str): Hash algorithm to use
        
    Returns:
        str: Hexadecimal checksum
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def verify_checksum(file_path: str, expected_checksum: str, algorithm: str = 'sha256') -> bool:
    """
    Verify file checksum.
    
    Args:
        file_path (str): Path to the file
        expected_checksum (str): Expected checksum
        algorithm (str): Hash algorithm to use
        
    Returns:
        bool: True if checksum matches, False otherwise
    """
    actual_checksum = calculate_checksum(file_path, algorithm)
    return actual_checksum.lower() == expected_checksum.lower()


def download_ornl_dataset(dataset_id: str, output_dir: str = DEFAULT_LIDAR_DIR,
                         api_key: str = ORNL_DAAC_API_KEY, overwrite: bool = False) -> List[str]:
    """
    Download a dataset from ORNL DAAC.
    
    Args:
        dataset_id (str): Dataset ID
        output_dir (str): Directory to save files
        api_key (str): API key for authentication
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        List[str]: Paths to downloaded files
    """
    logger.info(f"Downloading ORNL DAAC dataset {dataset_id}")
    
    # Create directory for this dataset
    dataset_dir = os.path.join(output_dir, f"ornl_{dataset_id}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # ORNL DAAC API endpoint for dataset metadata
    metadata_url = f"https://daac.ornl.gov/api/datasets/{dataset_id}/metadata"
    
    session = create_session()
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        # Get dataset metadata
        response = session.get(metadata_url, headers=headers)
        response.raise_for_status()
        metadata = response.json()
        
        # Save metadata
        metadata_path = os.path.join(dataset_dir, f"metadata_{dataset_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Get download URLs
        files_url = f"https://daac.ornl.gov/api/datasets/{dataset_id}/files"
        response = session.get(files_url, headers=headers)
        response.raise_for_status()
        files_data = response.json()
        
        downloaded_files = []
        
        # Download each file
        for file_info in files_data.get("files", []):
            file_url = file_info.get("url")
            file_name = file_info.get("name")
            
            if not file_url or not file_name:
                continue
            
            # Skip non-LiDAR files unless it's metadata
            if not any(ext in file_name.lower() for ext in [".las", ".laz", ".tif", ".tiff", ".json", ".xml", ".csv"]):
                logger.debug(f"Skipping non-data file: {file_name}")
                continue
            
            output_path = os.path.join(dataset_dir, file_name)
            
            try:
                downloaded_file = download_file(file_url, output_path, session=session, overwrite=overwrite)
                downloaded_files.append(downloaded_file)
            except DownloadError as e:
                logger.error(f"Failed to download {file_name}: {e}")
        
        return downloaded_files
    
    except Exception as e:
        logger.error(f"Error downloading ORNL DAAC dataset {dataset_id}: {e}")
        raise DownloadError(f"Failed to download ORNL DAAC dataset {dataset_id}: {e}")


def download_opentopo_dataset(dataset_id: str, output_dir: str = DEFAULT_LIDAR_DIR,
                             api_key: str = OPENTOPO_API_KEY, overwrite: bool = False) -> List[str]:
    """
    Download a dataset from OpenTopography.
    
    Args:
        dataset_id (str): Dataset ID
        output_dir (str): Directory to save files
        api_key (str): API key for authentication
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        List[str]: Paths to downloaded files
    """
    logger.info(f"Downloading OpenTopography dataset {dataset_id}")
    
    # Create directory for this dataset
    dataset_dir = os.path.join(output_dir, f"opentopo_{dataset_id}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # OpenTopography API endpoint
    metadata_url = f"https://portal.opentopography.org/API/metadata?otCollectionID={dataset_id}"
    if api_key:
        metadata_url += f"&apiKey={api_key}"
    
    session = create_session()
    
    try:
        # Get dataset metadata
        response = session.get(metadata_url)
        response.raise_for_status()
        metadata = response.json()
        
        # Save metadata
        metadata_path = os.path.join(dataset_dir, f"metadata_{dataset_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Get download URLs - OpenTopography requires specific API calls for each file type
        # This is a simplified version - in practice, you'd need to handle different file types
        
        # Example: Get point cloud data
        pointcloud_url = f"https://portal.opentopography.org/API/pointcloud?otCollectionID={dataset_id}"
        if api_key:
            pointcloud_url += f"&apiKey={api_key}"
        
        response = session.get(pointcloud_url)
        if response.status_code == 200:
            pointcloud_data = response.json()
            
            downloaded_files = []
            
            # Process each tile
            for tile in pointcloud_data.get("tiles", []):
                tile_url = tile.get("url")
                tile_name = tile.get("name", f"tile_{tile.get('id', 'unknown')}.laz")
                
                if not tile_url:
                    continue
                
                output_path = os.path.join(dataset_dir, tile_name)
                
                try:
                    downloaded_file = download_file(tile_url, output_path, session=session, overwrite=overwrite)
                    downloaded_files.append(downloaded_file)
                except DownloadError as e:
                    logger.error(f"Failed to download {tile_name}: {e}")
            
            return downloaded_files
        else:
            logger.warning(f"Failed to get point cloud data: {response.status_code}")
            return [metadata_path]
    
    except Exception as e:
        logger.error(f"Error downloading OpenTopography dataset {dataset_id}: {e}")
        raise DownloadError(f"Failed to download OpenTopography dataset {dataset_id}: {e}")


def download_zenodo_record(record_id: str, output_dir: str = DEFAULT_LIDAR_DIR,
                          access_token: str = ZENODO_ACCESS_TOKEN, overwrite: bool = False) -> List[str]:
    """
    Download files from a Zenodo record.
    
    Args:
        record_id (str): Zenodo record ID
        output_dir (str): Directory to save files
        access_token (str): Access token for authentication
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        List[str]: Paths to downloaded files
    """
    logger.info(f"Downloading Zenodo record {record_id}")
    
    # Create directory for this record
    record_dir = os.path.join(output_dir, f"zenodo_{record_id}")
    os.makedirs(record_dir, exist_ok=True)
    
    # Zenodo API endpoint
    record_url = f"https://zenodo.org/api/records/{record_id}"
    
    session = create_session()
    params = {}
    if access_token:
        params["access_token"] = access_token
    
    try:
        # Get record metadata
        response = session.get(record_url, params=params)
        response.raise_for_status()
        record_data = response.json()
        
        # Save metadata
        metadata_path = os.path.join(record_dir, f"metadata_{record_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(record_data, f, indent=2)
        
        downloaded_files = [metadata_path]
        
        # Download each file
        for file_info in record_data.get("files", []):
            file_url = file_info.get("links", {}).get("self")
            file_name = file_info.get("key")
            
            if not file_url or not file_name:
                continue
            
            # Skip non-data files
            if not any(ext in file_name.lower() for ext in [".las", ".laz", ".tif", ".tiff", ".zip", ".tar", ".gz", ".csv", ".json"]):
                logger.debug(f"Skipping non-data file: {file_name}")
                continue
            
            output_path = os.path.join(record_dir, file_name)
            
            try:
                downloaded_file = download_file(file_url, output_path, session=session, overwrite=overwrite)
                downloaded_files.append(downloaded_file)
                
                # If it's a compressed file, extract it
                if any(ext in file_name.lower() for ext in [".zip", ".tar.gz", ".tgz"]):
                    extract_dir = os.path.join(record_dir, os.path.splitext(file_name)[0])
                    extract_archive(output_path, extract_dir)
            except DownloadError as e:
                logger.error(f"Failed to download {file_name}: {e}")
        
        return downloaded_files
    
    except Exception as e:
        logger.error(f"Error downloading Zenodo record {record_id}: {e}")
        raise DownloadError(f"Failed to download Zenodo record {record_id}: {e}")


def extract_archive(archive_path: str, extract_dir: str) -> None:
    """
    Extract an archive file.
    
    Args:
        archive_path (str): Path to the archive file
        extract_dir (str): Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {extract_dir}")
    
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        if archive_path.lower().endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        
        elif archive_path.lower().endswith(('.tar.gz', '.tgz')):
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        else:
            logger.warning(f"Unsupported archive format: {archive_path}")
            return
        
        logger.info(f"Extraction complete: {extract_dir}")
    
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")


def download_kaggle_dataset(dataset_id: str, output_dir: str = DEFAULT_LIDAR_DIR,
                           overwrite: bool = False) -> List[str]:
    """
    Download a dataset from Kaggle.
    
    Note: Requires Kaggle API credentials to be set up (~/.kaggle/kaggle.json)
    
    Args:
        dataset_id (str): Kaggle dataset ID (username/dataset-name)
        output_dir (str): Directory to save files
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        List[str]: Paths to downloaded files
    """
    logger.info(f"Downloading Kaggle dataset {dataset_id}")
    
    try:
        # Check if kaggle module is installed
        import kaggle
    except ImportError:
        logger.error("Kaggle API not installed. Install with 'pip install kaggle'")
        raise DownloadError("Kaggle API not installed")
    
    # Create directory for this dataset
    dataset_dir = os.path.join(output_dir, f"kaggle_{dataset_id.replace('/', '_')}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_id,
            path=dataset_dir,
            unzip=True,
            force=overwrite
        )
        
        # List downloaded files
        downloaded_files = []
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                downloaded_files.append(os.path.join(root, file))
        
        return downloaded_files
    
    except Exception as e:
        logger.error(f"Error downloading Kaggle dataset {dataset_id}: {e}")
        raise DownloadError(f"Failed to download Kaggle dataset {dataset_id}: {e}")


def download_dataset(source: str, dataset_id: str = None, url: str = None,
                    output_dir: str = None, overwrite: bool = False) -> List[str]:
    """
    Download a dataset from a specified source.
    
    Args:
        source (str): Data source ('ornl', 'opentopo', 'zenodo', 'kaggle', 'url')
        dataset_id (str, optional): Dataset ID
        url (str, optional): Direct URL to download
        output_dir (str, optional): Directory to save files
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        List[str]: Paths to downloaded files
    
    Raises:
        ValueError: If required parameters are missing
        DownloadError: If download fails
    """
    if source not in ['ornl', 'opentopo', 'zenodo', 'kaggle', 'url']:
        raise ValueError(f"Unsupported source: {source}")
    
    if not dataset_id and not url:
        raise ValueError("Either dataset_id or url must be provided")
    
    # Set appropriate output directory based on source
    if output_dir is None:
        if source in ['ornl', 'opentopo']:
            output_dir = DEFAULT_LIDAR_DIR
        elif source == 'zenodo':
            output_dir = DEFAULT_LIDAR_DIR
        elif source == 'kaggle':
            output_dir = DEFAULT_LIDAR_DIR
        else:
            output_dir = DEFAULT_OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download based on source
    if source == 'ornl':
        return download_ornl_dataset(dataset_id, output_dir, overwrite=overwrite)
    
    elif source == 'opentopo':
        return download_opentopo_dataset(dataset_id, output_dir, overwrite=overwrite)
    
    elif source == 'zenodo':
        return download_zenodo_record(dataset_id, output_dir, overwrite=overwrite)
    
    elif source == 'kaggle':
        return download_kaggle_dataset(dataset_id, output_dir, overwrite=overwrite)
    
    elif source == 'url':
        if not url:
            raise ValueError("URL must be provided for source 'url'")
        
        # Determine filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename:
            filename = f"download_{int(time.time())}"
        
        output_path = os.path.join(output_dir, filename)
        
        return [download_file(url, output_path, overwrite=overwrite)]


def download_all_from_links(links_file: str = DEFAULT_LINKS_FILE,
                           output_dir: str = DEFAULT_OUTPUT_DIR,
                           sources: List[str] = None,
                           limit: int = None,
                           overwrite: bool = False) -> Dict[str, List[str]]:
    """
    Download all datasets from the Links.md file.
    
    Args:
        links_file (str): Path to the Links.md file
        output_dir (str): Base directory to save files
        sources (List[str], optional): List of sources to download from
        limit (int, optional): Maximum number of datasets to download per source
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        Dict[str, List[str]]: Dictionary of downloaded files by source
    """
    # Parse links file
    categorized_urls = parse_links_file(links_file)
    
    if not categorized_urls:
        logger.error("No URLs found in links file")
        return {}
    
    # Filter sources if specified
    if sources:
        categorized_urls = {k: v for k, v in categorized_urls.items() if k in sources}
    
    downloaded_files = {}
    
    # Download from each source
    for source, urls in categorized_urls.items():
        if limit:
            urls = urls[:limit]
        
        downloaded_files[source] = []
        
        for url in urls:
            try:
                dataset_id = extract_dataset_id(url)
                
                if dataset_id:
                    # Download using appropriate function
                    if source == 'ornl':
                        source_dir = os.path.join(output_dir, "lidar")
                        files = download_ornl_dataset(dataset_id, source_dir, overwrite=overwrite)
                    
                    elif source == 'opentopo':
                        source_dir = os.path.join(output_dir, "lidar")
                        files = download_opentopo_dataset(dataset_id, source_dir, overwrite=overwrite)
                    
                    elif source == 'zenodo':
                        source_dir = os.path.join(output_dir, "lidar")
                        files = download_zenodo_record(dataset_id, source_dir, overwrite=overwrite)
                    
                    elif source == 'kaggle':
                        source_dir = os.path.join(output_dir, "lidar")
                        files = download_kaggle_dataset(dataset_id, source_dir, overwrite=overwrite)
                    
                    else:
                        # For other sources, download directly from URL
                        parsed_url = urlparse(url)
                        filename = os.path.basename(parsed_url.path)
                        
                        if not filename:
                            filename = f"download_{int(time.time())}"
                        
                        if "lidar" in url.lower() or any(ext in url.lower() for ext in [".las", ".laz"]):
                            source_dir = os.path.join(output_dir, "lidar")
                        elif any(ext in url.lower() for ext in [".tif", ".tiff", ".png", ".jpg"]):
                            source_dir = os.path.join(output_dir, "rasters")
                        elif any(ext in url.lower() for ext in [".shp", ".geojson", ".kml"]):
                            source_dir = os.path.join(output_dir, "vectors")
                        elif any(ext in url.lower() for ext in [".pdf", ".txt", ".html"]):
                            source_dir = os.path.join(output_dir, "corpus")
                        else:
                            source_dir = os.path.join(output_dir, "misc")
                        
                        os.makedirs(source_dir, exist_ok=True)
                        output_path = os.path.join(source_dir, filename)
                        
                        files = [download_file(url, output_path, overwrite=overwrite)]
                    
                    downloaded_files[source].extend(files)
                
                else:
                    # If no dataset ID found, download directly from URL
                    parsed_url = urlparse(url)
                    filename = os.path.basename(parsed_url.path)
                    
                    if not filename:
                        filename = f"download_{int(time.time())}"
                    
                    if "lidar" in url.lower() or any(ext in url.lower() for ext in [".las", ".laz"]):
                        source_dir = os.path.join(output_dir, "lidar")
                    elif any(ext in url.lower() for ext in [".tif", ".tiff", ".png", ".jpg"]):
                        source_dir = os.path.join(output_dir, "rasters")
                    elif any(ext in url.lower() for ext in [".shp", ".geojson", ".kml"]):
                        source_dir = os.path.join(output_dir, "vectors")
                    elif any(ext in url.lower() for ext in [".pdf", ".txt", ".html"]):
                        source_dir = os.path.join(output_dir, "corpus")
                    else:
                        source_dir = os.path.join(output_dir, "misc")
                    
                    os.makedirs(source_dir, exist_ok=True)
                    output_path = os.path.join(source_dir, filename)
                    
                    files = [download_file(url, output_path, overwrite=overwrite)]
                    downloaded_files[source].extend(files)
            
            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")
    
    return downloaded_files


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Download LiDAR and other datasets for Amazon Deep Insights")
    
    # Command options
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download a specific dataset
    download_parser = subparsers.add_parser("download", help="Download a specific dataset")
    download_parser.add_argument("--source", "-s", required=True, choices=["ornl", "opentopo", "zenodo", "kaggle", "url"],
                              help="Data source")
    download_parser.add_argument("--id", "-i", help="Dataset ID")
    download_parser.add_argument("--url", "-u", help="Direct URL to download")
    download_parser.add_argument("--output", "-o", help="Output directory")
    download_parser.add_argument("--overwrite", "-f", action="store_true", help="Overwrite existing files")
    
    # Download all datasets from Links.md
    download_all_parser = subparsers.add_parser("download-all", help="Download all datasets from Links.md")
    download_all_parser.add_argument("--links", "-l", default=DEFAULT_LINKS_FILE, help="Path to Links.md file")
    download_all_parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR, help="Base output directory")
    download_all_parser.add_argument("--sources", "-s", nargs="+", help="Sources to download from")
    download_all_parser.add_argument("--limit", "-n", type=int, help="Maximum number of datasets to download per source")
    download_all_parser.add_argument("--overwrite", "-f", action="store_true", help="Overwrite existing files")
    
    # Parse links file
    parse_parser = subparsers.add_parser("parse", help="Parse Links.md file")
    parse_parser.add_argument("--links", "-l", default=DEFAULT_LINKS_FILE, help="Path to Links.md file")
    
    args = parser.parse_args()
    
    if args.command == "download":
        if not args.id and not args.url:
            parser.error("Either --id or --url must be provided")
        
        try:
            files = download_dataset(
                source=args.source,
                dataset_id=args.id,
                url=args.url,
                output_dir=args.output,
                overwrite=args.overwrite
            )
            
            logger.info(f"Downloaded {len(files)} files")
            for file in files:
                logger.info(f"  - {file}")
        
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "download-all":
        try:
            downloaded_files = download_all_from_links(
                links_file=args.links,
                output_dir=args.output,
                sources=args.sources,
                limit=args.limit,
                overwrite=args.overwrite
            )
            
            for source, files in downloaded_files.items():
                logger.info(f"Source: {source}")
                logger.info(f"  Downloaded {len(files)} files")
        
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "parse":
        try:
            categorized_urls = parse_links_file(args.links)
            
            for source, urls in categorized_urls.items():
                print(f"Source: {source}")
                print(f"  URLs: {len(urls)}")
                for url in urls[:5]:  # Show first 5 URLs
                    print(f"    - {url}")
                if len(urls) > 5:
                    print(f"    ... and {len(urls) - 5} more")
                print()
        
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
