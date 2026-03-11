import requests
import time
import sys
import logging
import os
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

API_BASE_URL = "http://127.0.0.1:8000"

def get_file_hash(filepath):
    """Compute MD5 hash of a file for Data Intactness check."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_benchmark_sync(dataset_name, filepath, query, target):
    logger.info(f"=== Starting Sync Benchmark for {dataset_name} ===")
    start_time = time.time()
    
    # SOTA Check: Data Intactness (Before)
    initial_hash = get_file_hash(filepath)
    
    files = {
        'file': (dataset_name, open(filepath, 'rb'), 'text/csv')
    }
    data = {
        "user_query": query,
        "target_column": target
    }
    
    logger.info("Sending POST request...")
    try:
        resp = requests.post(f"{API_BASE_URL}/analyze", data=data, files=files, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"POST failed for {dataset_name}: {e}")
        return {"name": dataset_name, "status": "failed", "error": str(e), "duration": round(time.time() - start_time, 2)}
        
    session_id = resp.json()["session_id"]
    logger.info(f"Pipeline started successfully. Session ID: {session_id}")
    
    max_wait = 300 # 5 minutes for complex datasets
    final_result = {"name": dataset_name, "session_id": session_id, "status": "timeout"}
    
    while max_wait > 0:
        try:
            status_resp = requests.get(f"{API_BASE_URL}/status/{session_id}", timeout=10)
            if status_resp.status_code == 200:
                cp = status_resp.json()
                status = cp.get("status", "unknown")
                if status == "completed":
                    logger.info(f"[{dataset_name}] Pipeline FINISHED.")
                    final_result["status"] = "completed"
                    break
                elif status == "failed":
                    logger.error(f"[{dataset_name}] Pipeline FAILED.")
                    final_result["status"] = "failed"
                    break
        except Exception as e:
            logger.warning(f"Status poll failed: {e}")
            
        time.sleep(5)
        max_wait -= 5
        
    if final_result["status"] == "completed":
        try:
            art_resp = requests.get(f"{API_BASE_URL}/artifacts/{session_id}", timeout=10)
            if art_resp.status_code == 200:
                registry = art_resp.json()
                arts = registry.get("artifacts", [])
                
                # SOTA Check: Plot Validity
                plots_found = any(a.get("artifact_type") == "plot" for a in arts)
                final_result["plots"] = plots_found
                
                for art in arts:
                    if art["phase"] == "training" and art["artifact_type"] == "model_comparison":
                        d = art.get("data", {})
                        rl_iters = d.get("rl_iterations", 1)
                        models = d.get("models", [])
                        if models:
                            best = models[0]
                            final_result.update({
                                "model": best.get('name'),
                                "score": best.get('test_score'),
                                "metric": best.get('scoring'),
                                "iters": rl_iters
                            })
                            logger.info(f"[{dataset_name}] AGENTIC RL RESULTS: {best.get('name')} | Metric={best.get('test_score')} | Iters={rl_iters}")
            else:
                logger.error("Failed to load artifacts.")
        except Exception as e:
            logger.error(f"Artifact fetch failed: {e}")

    # SOTA Check: Data Intactness (After)
    final_hash = get_file_hash(filepath)
    final_result["intact"] = (initial_hash == final_hash)
    if not final_result["intact"]:
        logger.warning(f"[{dataset_name}] DATA INTACTNESS FAILED! File was modified.")

    final_result["duration"] = round(time.time() - start_time, 2)
    return final_result

if __name__ == "__main__":
    benchmarks = [
        ("iris.csv", "tests/fixtures/iris.csv", "Predict the species", "Species"),
        ("housing_benchmark.csv", "tests/fixtures/housing_benchmark.csv", "Predict house values", "MedianHouseValue"),
        ("churn_benchmark.csv", "tests/fixtures/churn_benchmark.csv", "Predict churn probability", "Churn"),
        ("credit_benchmark.csv", "tests/fixtures/credit_benchmark.csv", "Predict credit risk", "Risk"),
        ("hard_benchmark.csv", "tests/fixtures/hard_benchmark.csv", "Predict high income", "HighIncome"),
    ]
    
    all_results = []
    for name, path, query, target in benchmarks:
        if not os.path.exists(path):
            # Try absolute path if relative fails
            abs_path = os.path.join(os.getcwd(), path)
            if not os.path.exists(abs_path):
                # Try data/ directory or root
                if os.path.exists(name):
                    path = name
                else:
                    logger.error(f"Path not found: {path} or {name}")
                    continue
            else:
                path = abs_path
            
        res = run_benchmark_sync(name, path, query, target)
        all_results.append(res)
        time.sleep(2)
    
    print("\n" + "="*95)
    print("DATACORTEX AI SOTA BENCHMARK SUMMARY")
    print(f"{'Dataset':<25} | {'Status':<10} | {'Metric':<10} | {'Score':<10} | {'Iters':<5} | {'Intact':<6} | {'Plots':<5}")
    print("-" * 95)
    for r in all_results:
        status = r['status']
        metric = r.get('metric', 'N/A')
        score = f"{r.get('score', 0):.4f}" if status == 'completed' else 'N/A'
        iters = r.get('iters', 'N/A')
        intact = "Yes" if r.get('intact') else "NO"
        plots = "Yes" if r.get('plots') else "No"
        print(f"{r['name']:<25} | {status:<10} | {metric:<10} | {score:<10} | {iters:<5} | {intact:<6} | {plots:<5}")
    print("="*95)
