"""Generate JSON data for the risk dashboard from the risk score CSV."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List

RISK_LEVEL_ORDER = {"低风险": 0, "中风险": 1, "高风险": 2}


def load_risk_scores(csv_path: Path) -> List[Dict[str, object]]:
    """Read the CSV file and normalise the rows."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        records: List[Dict[str, object]] = []
        for row in reader:
            record = {
                "id": row["id"],
                "risk_score": float(row["risk_score"]),
                "risk_level": row["risk_level"],
                "components": {
                    key.replace("component_", ""): float(value)
                    for key, value in row.items()
                    if key.startswith("component_")
                },
            }
            records.append(record)
    return records


def build_summary(records: List[Dict[str, object]]) -> Dict[str, object]:
    scores = [record["risk_score"] for record in records]
    level_counts: Dict[str, int] = {}
    for record in records:
        level_counts[record["risk_level"]] = level_counts.get(record["risk_level"], 0) + 1

    return {
        "count": len(records),
        "average_score": round(mean(scores), 4) if scores else None,
        "min_score": round(min(scores), 4) if scores else None,
        "max_score": round(max(scores), 4) if scores else None,
        "level_counts": level_counts,
    }


def generate_dashboard_data(records: List[Dict[str, object]]) -> Dict[str, object]:
    sorted_records = sorted(
        records,
        key=lambda record: (
            RISK_LEVEL_ORDER.get(record["risk_level"], 99),
            record["risk_score"],
        ),
    )
    for index, record in enumerate(sorted_records, start=1):
        record["rank"] = index
    summary = build_summary(sorted_records)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": sorted_records,
        "summary": summary,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "artifacts" / "risk_scores.csv"
    default_output = repo_root / "artifacts" / "risk_scores.json"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to the CSV file with risk scores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination path for the generated JSON file.",
    )
    args = parser.parse_args()

    records = load_risk_scores(args.input)
    dashboard_data = generate_dashboard_data(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(dashboard_data, fh, ensure_ascii=False, indent=2)

    print(f"Generated {len(dashboard_data['records'])} records into {args.output}")


if __name__ == "__main__":
    main()
