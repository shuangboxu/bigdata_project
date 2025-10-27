class DataPlanner:
    def __call__(self, state):
        plan = {
            "steps": [
                "data_preprocessing",
                "feature_engineering",
                "model_training",
                "evaluation",
                "visualization",
                "llm_report"
            ]
        }
        print("[Planner] Created execution plan:", plan["steps"])
        return {"plan": plan, "data": state["data"]}
