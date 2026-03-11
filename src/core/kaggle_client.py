import os
import logging
from typing import List, Dict, Optional
import subprocess

logger = logging.getLogger(__name__)

class KaggleClient:
    """
    Wrapper for Kaggle API to discover and download datasets for benchmarking.
    """
    def __init__(self, username: str, key: str):
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        
        # Verify connectivity/credentials
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info(f"Kaggle API authenticated for user: {username}")
        except Exception as e:
            logger.error(f"Kaggle authentication failed: {e}")
            self.api = None

    def search_datasets(self, query: str = "tabular classification", max_limit: int = 10) -> List[Dict]:
        """Search for datasets matching a query."""
        if not self.api:
            return []
            
        try:
            datasets = self.api.dataset_list(search=query, sort_by='votes')
            results = []
            for ds in datasets[:max_limit]:
                results.append({
                    "ref": ds.ref,
                    "title": ds.title,
                    "size": getattr(ds, "total_bytes", 0),
                    "download_count": getattr(ds, "download_count", 0),
                    "vote_count": getattr(ds, "vote_count", 0)
                })
            return results
        except Exception as e:
            logger.error(f"Kaggle search failed: {e}")
            return []

    def download_dataset(self, dataset_ref: str, path: str = "benchmarks/kaggle") -> Optional[str]:
        """Download and unzip a dataset into the specified path."""
        if not self.api:
            return None

        if not os.path.exists(path):
            os.makedirs(path)

        try:
            logger.info(f"Downloading dataset: {dataset_ref} to {path}")
            self.api.dataset_download_files(dataset_ref, path=path, unzip=True)
            
            # Find the largest CSV file in the downloaded path
            csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
            if not csv_files:
                logger.warning(f"No CSV files found in {dataset_ref}")
                return None
            
            # Heuristic: largest CSV is usually the main dataset
            csv_files.sort(key=lambda x: os.path.getsize(os.path.join(path, x)), reverse=True)
            return os.path.join(path, csv_files[0])
            
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            return None
