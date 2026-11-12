import numpy as np
import pandas as pd

def clean_origination(orig: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = orig.copy()
    numeric_cols = cfg["features"]["numeric"]
    categorical_cols = cfg["features"]["categorical"]
    not_avail = cfg["data"]["not_available"]

    # 1. Convert numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Strip strings on categorical columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "None", "<NA>", "NaN"]), col] = np.nan

    # 3. DP-05: Replace documented Not Available sentinels with NaN
    for col, codes in not_avail.items():
        if col in df.columns:
            if col in numeric_cols:
                codes_num = [float(c) for c in codes]
                df.loc[df[col].isin(codes_num), col] = np.nan
            else:
                codes_str = [str(c).strip() for c in codes]
                df.loc[df[col].isin(codes_str), col] = np.nan

    # 4. SRS 5.9: borrower count capped at 2 (1 borrower vs 2 or more)
    if "n_borrowers" in df.columns:
        df["n_borrowers"] = df["n_borrowers"].clip(upper=cfg["features"]["borrower_cap"])

    # 5. Vintage extraction
    if "loan_id" in df.columns:
        def extract_vintage_info(lid, fpd):
            lid_str = str(lid).strip()
            if len(lid_str) >= 2 and lid_str[:2].isdigit():
                v = 2000 + int(lid_str[:2])
            elif "F" in lid_str and len(lid_str) >= 4 and lid_str[1:3].isdigit():
                v = 2000 + int(lid_str[1:3])
            elif pd.notna(fpd) and len(str(fpd)) >= 4 and str(fpd)[:4].isdigit():
                v = int(str(fpd)[:4])
            else:
                v = 2018

            if pd.notna(fpd) and len(str(fpd)) >= 6 and str(fpd)[:6].isdigit():
                m = int(str(fpd)[4:6])
                q = (m - 1) // 3 + 1
            else:
                q = 1
            return v, f"{v}Q{q}"

        fp_col = df["first_payment_date"] if "first_payment_date" in df.columns else [None] * len(df)
        v_vq = [extract_vintage_info(lid, fpd) for lid, fpd in zip(df["loan_id"], fp_col)]
        df["vintage"] = [x[0] for x in v_vq]
        df["vintage_quarter"] = [x[1] for x in v_vq]

    return df

def compute_quarter_median(df_all: pd.DataFrame) -> dict:
    if "vintage_quarter" not in df_all.columns or "orig_rate" not in df_all.columns:
        return {}
    return df_all.groupby("vintage_quarter")["orig_rate"].median().dropna().to_dict()

def add_derived(df: pd.DataFrame, quarter_median: dict) -> pd.DataFrame:
    df = df.copy()
    if "orig_cltv" in df.columns and "orig_ltv" in df.columns:
        df["cltv_minus_ltv"] = df["orig_cltv"] - df["orig_ltv"]
    if "orig_rate" in df.columns and "vintage_quarter" in df.columns:
        mapped = df["vintage_quarter"].map(quarter_median)
        df["rate_vs_quarter_median"] = df["orig_rate"] - mapped
    return df

def apply_categories(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    df = df.copy()
    for col, levels in categories.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=levels)
    return df

def feature_matrix(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    if cfg is None:
        from src.common.config import load_config
        cfg = load_config()
    cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    df_copy = df.copy()
    for c in cols:
        if c not in df_copy.columns:
            df_copy[c] = np.nan
    return df_copy[cols]
