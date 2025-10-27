from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

# Default input Excel
DEFAULT_INPUT = DATA_DIR / "数据示例.xlsx"
DEFAULT_SHEET = "Sheet1"

# Random seed
SEED = 42

# Columns & domain knowledge
# Adjust if your data schema differs.
NUMERIC_CANDIDATES = [
    "loanAmnt","term","interestRate","installment","annualIncome","dti",
    "revolUtil","n0","n1","n2","n3","n4","n5","n6","n7","n8","n9","n10",
    "n11","n12","n13","n14"
]

CATEGORICAL_CANDIDATES = [
    "grade","subGrade","employmentLength","homeOwnership",
    "verificationStatus","purpose","postCode","regionCode",
    "initialListStatus","applicationType","employmentTitle"
]

DATE_COLUMNS = ["issueDate", "earliesCreditLine"]

ID_COLUMNS = ["id"]

# When predicting interestRate, it's safer to drop grade/subGrade to avoid leakage.
LEAKY_FOR_INTEREST = ["grade", "subGrade"]

# Plot style
MPL_DPI = 140
