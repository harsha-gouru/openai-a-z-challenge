from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Core dependencies
core_deps = [
    'numpy>=1.24.0',
    'pandas>=2.0.0',
    'geopandas>=0.13.0',
    'shapely>=2.0.0',
    'pyproj>=3.5.0',
    'rasterio>=1.3.6',
    'GDAL>=3.6.0',
    'pdal>=2.5.0',
    'pylas>=0.4.3',
    'laspy>=2.3.0',
    'dvc>=3.0.0',
    'python-dotenv>=1.0.0',
    'requests>=2.28.0',
]

# ML dependencies
ml_deps = [
    'scikit-learn>=1.2.0',
    'xgboost>=1.7.0',
    'foresttools>=0.2.0',
    'scipy>=1.10.0',
]

# RAG dependencies
rag_deps = [
    'chromadb>=0.4.6',
    'langchain>=0.0.267',
    'langchain-openai>=0.0.2',
    'openai>=1.3.0',
    'tiktoken>=0.5.0',
    'unstructured>=0.10.0',
    'pypdf>=3.15.0',
    'beautifulsoup4>=4.12.0',
    'lxml>=4.9.0',
]

# Visualization dependencies
viz_deps = [
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'folium>=0.14.0',
    'keplergl>=0.3.2',
    'plotly>=5.14.0',
    'pydeck>=0.8.0',
    'ipywidgets>=8.0.0',
]

# Web dependencies
web_deps = [
    'fastapi>=0.103.0',
    'uvicorn>=0.23.0',
    'streamlit>=1.26.0',
    'streamlit-folium>=0.13.0',
    'streamlit-chat>=0.1.0',
    'pydantic>=2.3.0',
]

# Dev dependencies
dev_deps = [
    'pytest>=7.4.0',
    'mkdocs>=1.5.0',
    'mkdocs-material>=9.2.0',
    'jupyter>=1.0.0',
    'notebook>=7.0.0',
]

setup(
    name="amazon_insights",
    version="0.1.0",
    description="Amazon Deep Insights - Analyzing LiDAR data with AI for ecological and archaeological discovery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="OpenAI A-Z Challenge Team",
    author_email="harsha.gouru@example.com",
    url="https://github.com/harsha-gouru/openai-a-z-challenge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "core": core_deps,
        "ml": ml_deps,
        "rag": rag_deps,
        "viz": viz_deps,
        "web": web_deps,
        "dev": dev_deps,
        "all": core_deps + ml_deps + rag_deps + viz_deps + web_deps + dev_deps,
    },
    entry_points={
        "console_scripts": [
            "amazon-insights-api=amazon_insights.rag.api:main",
            "amazon-insights-app=amazon_insights.visualization.app:main",
            "amazon-insights-download=amazon_insights.data_ingestion.download:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="lidar, amazon, rainforest, archaeology, machine learning, rag, openai",
)
