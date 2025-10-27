from src.pipeline_cls import run_classification_pipeline

class DataExecutor:
    def __call__(self, state):
        df = state["data"]
        print("[Executor] Starting classification pipeline...")
        metrics, model = run_classification_pipeline(df)
        if "auc" in metrics:
            print(f"[Executor] Pipeline finished. AUC={metrics['auc']:.4f}")
        return {"metrics": metrics, "model": model, "data": df}
