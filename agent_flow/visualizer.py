import os
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

class Visualizer:
    def __call__(self, state):
        metrics = state["metrics"]
        y_true = metrics.get("y_true")
        y_pred = metrics.get("y_pred")
        if y_true is None or y_pred is None:
            print("[Visualizer] y_true / y_pred missing in metrics; skip ROC.")
            return {}

        os.makedirs("artifacts/classification", exist_ok=True)
        out_path = "artifacts/classification/roc_curve_agent.png"
        RocCurveDisplay.from_predictions(y_true, y_pred)
        plt.title(f"ROC Curve (AUC={metrics.get('auc', 0):.3f})")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Visualizer] ROC saved -> {out_path}")
        return {"roc_path": out_path}
