import pandas as pd

def load_data(path: str):
    print(f"[Utils] Loading data from {path}")
    try:
        return pd.read_excel(path)
    except Exception as e_xlsx:
        try:
            print(f"[Utils] Excel load failed ({e_xlsx}); trying CSV...")
            return pd.read_csv(path)
        except Exception as e_csv:
            raise RuntimeError(f"Failed to load data from {path}. Excel error: {e_xlsx}; CSV error: {e_csv}")
