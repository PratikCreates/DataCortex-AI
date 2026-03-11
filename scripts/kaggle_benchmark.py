import os
import time
import logging
from src.core.kaggle_client import KaggleClient
from benchmark_sync import run_benchmark_sync
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if not username or not key:
        logger.error("Kaggle credentials not found in environment.")
        return

    client = KaggleClient(username, key)
    
    # 1. Search for high-quality tabular datasets
    logger.info("Searching for Kaggle datasets...")
    datasets = client.search_datasets("tabular binary classification", max_limit=3)
    
    if not datasets:
        logger.error("No datasets found.")
        return

    all_results = []
    
    for ds in datasets:
        ref = ds["ref"]
        title = ds["title"]
        logger.info(f"Targeting dataset: {title} ({ref})")
        
        # 2. Download and get path
        local_path = client.download_dataset(ref, path=f"benchmarks/kaggle/{ref.replace('/', '_')}")
        if not local_path:
            logger.warning(f"Skipping {title} - download failed.")
            continue
            
        # 3. Guess target column (IngestionAgent will do this normally, but we can pass hint if known)
        # For Kaggle datasets, the last column is often the target
        logger.info(f"Running benchmark for {title}...")
        res = run_benchmark_sync(title, local_path, f"Predict the target for {title}", None)
        if res:
            all_results.append(res)
        
        time.sleep(5) # Cooldown between datasets

    # 4. Summary Table
    print("\n" + "="*85)
    print("KAGGLE SOTA BENCHMARK SUMMARY")
    print(f"{'Dataset':<25} | {'Status':<10} | {'Metric':<10} | {'Score':<10} | {'Iters':<5}")
    print("-"*85)
    for r in all_results:
        status = r.get('status', 'unknown')
        metric = r.get('metric', 'N/A')
        score = f"{r.get('score', 0):.4f}" if status == 'completed' else "N/A"
        iters = r.get('iters', 'N/A')
        print(f"{r['name']:<25} | {status:<10} | {metric:<10} | {score:<10} | {iters:<5}")
    print("="*85)

if __name__ == "__main__":
    main()
