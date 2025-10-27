class DataPlanner:
    def __call__(self, state):
        plan = {
            "steps": [
                "data_preprocessing",
                "feature_engineering",
                "model_training",
                "evaluation",
                "visualization",
                "llm_report",
            ]
        }
        print("[Planner] Created execution plan:", plan["steps"])
        # propagate the raw data forward so downstream nodes have the
        # original frame available without having to reload it.
        return {**state, "plan": plan}
