#!/usr/bin/env python
"""
Amazon Deep Insights - Integration Test Script

This script tests the integration of different components of the Amazon Deep Insights system:
- Data ingestion
- LiDAR processing
- RAG system
- Visualization capabilities

Run this script to verify that all components work together correctly.
"""

import os
import sys
import logging
import tempfile
import unittest
import shutil
from pathlib import Path
import time
import json
import requests
from unittest.mock import patch, MagicMock

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amazon_insights.test_integration")

# Import local modules
try:
    from src.data_ingestion.download import (
        download_file, validate_checksum, parse_links_file, extract_dataset_id
    )
    from src.preprocessing.lidar_processing import (
        validate_lidar_file, get_lidar_info, process_lidar_tile,
        generate_dem, generate_dsm, generate_chm
    )
    from src.rag.embeddings import (
        create_chroma_client, get_openai_embedding_function,
        get_or_create_collection, chunk_text, build_knowledge_base
    )
    
    # Mock visualization components since they require UI
    # We'll just test the functions that don't require UI interaction
    
    MODULES_LOADED = True
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    MODULES_LOADED = False


class TestIntegration(unittest.TestCase):
    """Integration tests for Amazon Deep Insights system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directories for test data
        cls.temp_dir = tempfile.mkdtemp()
        cls.raw_dir = os.path.join(cls.temp_dir, "raw")
        cls.processed_dir = os.path.join(cls.temp_dir, "processed")
        cls.vector_db_dir = os.path.join(cls.temp_dir, "vector_db")
        cls.corpus_dir = os.path.join(cls.temp_dir, "corpus")
        
        # Create directories
        os.makedirs(cls.raw_dir, exist_ok=True)
        os.makedirs(cls.processed_dir, exist_ok=True)
        os.makedirs(cls.vector_db_dir, exist_ok=True)
        os.makedirs(cls.corpus_dir, exist_ok=True)
        
        # Create a sample text file for the corpus
        cls.sample_text_path = os.path.join(cls.corpus_dir, "sample.txt")
        with open(cls.sample_text_path, "w") as f:
            f.write("""
            # Amazon Rainforest and LiDAR Technology
            
            The Amazon rainforest is the world's largest tropical rainforest, covering much of northwestern Brazil
            and extending into Colombia, Peru, and other South American countries. It's famous for its biodiversity,
            with one in ten known species living in the Amazon.
            
            LiDAR (Light Detection and Ranging) technology has revolutionized the study of the Amazon rainforest.
            By sending laser pulses from aircraft and measuring their return times, LiDAR creates detailed 3D maps
            of the forest structure, even penetrating the dense canopy to reveal the ground beneath.
            
            Recent LiDAR surveys have revealed numerous archaeological sites hidden beneath the canopy,
            including geometric earthworks known as geoglyphs. These discoveries suggest that pre-Columbian
            civilizations in the Amazon were more numerous and sophisticated than previously thought.
            
            Additionally, LiDAR has helped identify exceptionally tall trees, with some reaching heights of over
            80 meters, making them among the tallest trees in the world.
            """)
        
        # URL for a sample LAS file (publicly available)
        cls.sample_lidar_url = "https://github.com/PDAL/data/raw/master/autzen/autzen-classified.las"
        cls.sample_lidar_path = os.path.join(cls.raw_dir, "sample.las")
        
        # Download sample LiDAR file if modules loaded successfully
        if MODULES_LOADED:
            try:
                download_file(cls.sample_lidar_url, cls.sample_lidar_path)
                logger.info(f"Downloaded sample LiDAR file to {cls.sample_lidar_path}")
                cls.lidar_available = True
            except Exception as e:
                logger.error(f"Error downloading sample LiDAR file: {e}")
                cls.lidar_available = False
        else:
            cls.lidar_available = False
        
        # Check if API is running
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            cls.api_running = response.status_code == 200
        except:
            cls.api_running = False
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        if not MODULES_LOADED:
            self.skipTest("Required modules not loaded")
    
    def test_data_ingestion(self):
        """Test data ingestion functionality."""
        # Test file download
        test_url = "https://raw.githubusercontent.com/PDAL/data/master/README.md"
        test_path = os.path.join(self.raw_dir, "test_readme.md")
        
        downloaded_path = download_file(test_url, test_path)
        self.assertTrue(os.path.exists(downloaded_path))
        self.assertEqual(downloaded_path, test_path)
        
        # Test URL parsing
        test_links = """
        https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1644
        https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.042013.4326.1
        https://zenodo.org/records/7689909
        """
        
        with open(os.path.join(self.temp_dir, "test_links.md"), "w") as f:
            f.write(test_links)
        
        parsed_links = parse_links_file(os.path.join(self.temp_dir, "test_links.md"))
        self.assertIn("ornl", parsed_links)
        self.assertIn("opentopo", parsed_links)
        self.assertIn("zenodo", parsed_links)
        
        # Test dataset ID extraction
        ornl_url = "https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1644"
        opentopo_url = "https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.042013.4326.1"
        zenodo_url = "https://zenodo.org/records/7689909"
        
        self.assertEqual(extract_dataset_id(ornl_url), "1644")
        self.assertEqual(extract_dataset_id(opentopo_url), "OT.042013.4326.1")
        self.assertEqual(extract_dataset_id(zenodo_url), "7689909")
    
    def test_lidar_processing(self):
        """Test LiDAR processing functionality."""
        if not self.lidar_available:
            self.skipTest("Sample LiDAR file not available")
        
        # Test LiDAR validation
        self.assertTrue(validate_lidar_file(self.sample_lidar_path))
        
        # Test getting LiDAR info
        info = get_lidar_info(self.sample_lidar_path)
        self.assertIn("point_count", info)
        self.assertIn("bounds", info)
        self.assertIn("dimensions", info)
        
        # Test DEM generation
        dem_path = os.path.join(self.processed_dir, "dem", "sample_dem.tif")
        os.makedirs(os.path.dirname(dem_path), exist_ok=True)
        
        try:
            generated_dem = generate_dem(self.sample_lidar_path, dem_path)
            self.assertTrue(os.path.exists(generated_dem))
            
            # Test DSM generation
            dsm_path = os.path.join(self.processed_dir, "dsm", "sample_dsm.tif")
            os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
            
            generated_dsm = generate_dsm(self.sample_lidar_path, dsm_path)
            self.assertTrue(os.path.exists(generated_dsm))
            
            # Test CHM generation
            chm_path = os.path.join(self.processed_dir, "chm", "sample_chm.tif")
            os.makedirs(os.path.dirname(chm_path), exist_ok=True)
            
            generated_chm = generate_chm(generated_dem, generated_dsm, chm_path)
            self.assertTrue(os.path.exists(generated_chm))
        except Exception as e:
            logger.error(f"Error in LiDAR processing: {e}")
            self.fail(f"LiDAR processing failed: {e}")
    
    @patch('openai.OpenAI')
    def test_rag_system(self, mock_openai):
        """Test RAG system functionality."""
        # Mock OpenAI embedding function
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_data = [MagicMock()]
        mock_data[0].embedding = [0.1] * 1536  # Mock embedding vector
        mock_response.data = mock_data
        
        # Test text chunking
        text = "This is a test document. It has multiple sentences. We want to split it into chunks."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        self.assertGreater(len(chunks), 1)
        
        # Test ChromaDB client creation
        try:
            client = create_chroma_client(persist_directory=self.vector_db_dir)
            self.assertIsNotNone(client)
            
            # Test collection creation
            with patch('src.rag.embeddings.get_openai_embedding_function') as mock_embedding_func:
                mock_embedding_func.return_value = lambda texts: [[0.1] * 1536 for _ in texts]
                
                collection = get_or_create_collection(
                    client,
                    "test_collection",
                    mock_embedding_func.return_value
                )
                self.assertIsNotNone(collection)
                self.assertEqual(collection.name, "test_collection")
        except Exception as e:
            logger.error(f"Error in RAG system test: {e}")
            self.fail(f"RAG system test failed: {e}")
    
    def test_api_integration(self):
        """Test API integration."""
        if not self.api_running:
            self.skipTest("API not running")
        
        # Test health endpoint
        response = requests.get("http://localhost:8080/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        
        # Test query endpoint
        response = requests.post(
            "http://localhost:8080/query",
            json={
                "query": "What is LiDAR technology?",
                "collection_name": "amazon_insights",
                "top_k": 3
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("query", data)
        self.assertIn("contexts", data)
        
        # Test generate endpoint
        response = requests.post(
            "http://localhost:8080/generate",
            json={
                "query": "What is LiDAR technology?",
                "collection_name": "amazon_insights",
                "top_k": 3,
                "model": "gpt-4o-mini",
                "max_tokens": 100,
                "temperature": 0.2
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("query", data)
        self.assertIn("answer", data)
    
    def test_end_to_end(self):
        """Test end-to-end workflow."""
        if not self.lidar_available:
            self.skipTest("Sample LiDAR file not available")
        
        if not self.api_running:
            self.skipTest("API not running")
        
        try:
            # 1. Process LiDAR data
            results = process_lidar_tile(
                self.sample_lidar_path,
                output_dir=self.processed_dir,
                resolution=1.0,
                generate_products=["dem", "dsm", "chm"]
            )
            
            self.assertIn("dem", results)
            self.assertIn("dsm", results)
            self.assertIn("chm", results)
            
            # 2. Create a text file with information about the processed data
            info_file_path = os.path.join(self.corpus_dir, "processed_data_info.txt")
            with open(info_file_path, "w") as f:
                f.write(f"""
                # Processed LiDAR Data Information
                
                File: {os.path.basename(self.sample_lidar_path)}
                Processing Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
                
                ## Generated Products
                
                - DEM: {results['dem']}
                - DSM: {results['dsm']}
                - CHM: {results['chm']}
                
                ## Description
                
                This LiDAR data shows a sample area with various terrain features and vegetation.
                The Digital Elevation Model (DEM) represents the bare-earth terrain.
                The Digital Surface Model (DSM) includes vegetation and other above-ground features.
                The Canopy Height Model (CHM) shows the height of vegetation above the ground.
                """)
            
            # 3. Build knowledge base with the corpus
            with patch('src.rag.embeddings.get_openai_embedding_function') as mock_embedding_func:
                mock_embedding_func.return_value = lambda texts: [[0.1] * 1536 for _ in texts]
                
                # Mock the add_documents_to_collection function to avoid actual embedding
                with patch('src.rag.embeddings.add_documents_to_collection') as mock_add_docs:
                    mock_add_docs.return_value = ["doc1", "doc2"]
                    
                    collection = build_knowledge_base(
                        corpus_dir=self.corpus_dir,
                        persist_dir=self.vector_db_dir,
                        collection_name="test_collection",
                        force_rebuild=True
                    )
            
            # 4. Query the API with a relevant question
            response = requests.post(
                "http://localhost:8080/generate",
                json={
                    "query": "What does a Canopy Height Model show?",
                    "collection_name": "amazon_insights",
                    "top_k": 3,
                    "model": "gpt-4o-mini",
                    "max_tokens": 100,
                    "temperature": 0.2
                }
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("answer", data)
            
            logger.info("End-to-end test completed successfully")
            
        except Exception as e:
            logger.error(f"Error in end-to-end test: {e}")
            self.fail(f"End-to-end test failed: {e}")


if __name__ == "__main__":
    unittest.main()
