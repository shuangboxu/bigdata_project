from typing import TypedDict, Any
from langgraph.graph import StateGraph

from agent_flow.planner import DataPlanner
from agent_flow.executor import DataExecutor
from agent_flow.visualizer import Visualizer
from agent_flow.reporter import Reporter
from agent_flow.utils import load_data


class WorkflowState(TypedDict, total=False):
    data: Any
    plan: Any
    metrics: Any
    model: Any
    roc_path: str
    report_text: str


def run():
    df = load_data("data/数据示例.xlsx")

    graph = StateGraph(WorkflowState)

    planner = DataPlanner()
    executor = DataExecutor()
    visualizer = Visualizer()
    reporter = Reporter()

    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("visualizer", visualizer)
    graph.add_node("reporter", reporter)

    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "visualizer")
    graph.add_edge("visualizer", "reporter")

    graph.set_entry_point("planner")
    graph.set_finish_point("reporter")

    app = graph.compile()
    print("[Workflow] Starting LangGraph pipeline...")
    result = app.invoke({"data": df})
    print("[Workflow] Finished. Results:", {k: type(v).__name__ for k, v in result.items()})


if __name__ == "__main__":
    run()
