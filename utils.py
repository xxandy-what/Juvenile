import pandas as pd
import numpy as np

def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    d2 = d.replace(0, np.nan)
    out = n / d2
    return out.replace([np.inf, -np.inf], np.nan)

def sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'