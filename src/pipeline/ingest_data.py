import io
import json
import zipfile
from pathlib import Path
import pandas as pd
from src.common.config import load_config, load_schema, project_path

def main():
    cfg = load_config()
    schema_o = load_schema("origination")
    schema_p = load_schema("performance")

    raw = project_path(cfg, "raw")
    interim = project_path(cfg, "interim")
    interim.mkdir(parents=True, exist_ok=True)

    years = sorted(list(set(sum(cfg["data"]["vintages"].values(), []))))
    manifest = {}

    print(f"Ingesting raw vintage files for years: {years}")

    for year in years:
        zip_path = raw / f"sample_{year}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing {zip_path}; check raw data in {raw} (DP-02)")

        with zipfile.ZipFile(zip_path, "r") as zf:
            orig_name = f"sample_orig_{year}.txt"
            perf_name = f"sample_perf_{year}.txt"
            if orig_name not in zf.namelist() or perf_name not in zf.namelist():
                raise FileNotFoundError(f"Missing {orig_name} or {perf_name} in {zip_path}")

            orig_txt = io.BytesIO(zf.read(orig_name))
            perf_txt = io.BytesIO(zf.read(perf_name))

        orig = pd.read_csv(orig_txt, sep="|", header=None, names=schema_o, dtype=str)

        # DP-03: Column count validation
        if orig.shape[1] != len(schema_o):
            raise ValueError(f"{year}: Origination column count {orig.shape[1]} does not match schema length {len(schema_o)}")

        # DP-03: Duplicate loan ID validation
        if not orig["loan_id"].is_unique:
            raise ValueError(f"{year}: Loan identifier appears more than once in origination file (DP-03)")

        # Stream / read performance file keeping label and loss fields
        PERF_KEEP = ["loan_id", "period", "delinq_status", "zero_bal_code", "actual_loss"]
        perf = pd.read_csv(perf_txt, sep="|", header=None, names=schema_p, dtype=str, usecols=PERF_KEEP)

        orig.to_parquet(interim / f"orig_{year}.parquet", index=False)
        perf.to_parquet(interim / f"perf_{year}.parquet", index=False)

        # DP-03: Censoring check
        last_first_pay = orig["first_payment_date"].dropna().astype(str).str.strip().str[:6].max()
        last_period = perf["period"].dropna().astype(str).str.strip().str[:6].max()

        if last_first_pay and last_period:
            lp_yr, lp_mo = int(last_period[:4]), int(last_period[4:6])
            lfp_yr, lfp_mo = int(last_first_pay[:4]), int(last_first_pay[4:6])
            diff_months = (lp_yr - lfp_yr) * 12 + (lp_mo - lfp_mo)
            if diff_months < cfg["data"]["label_window_months"]:
                raise ValueError(f"{year}: Performance data ends less than 24 months after latest first payment date ({diff_months} < 24) (DP-03)")

        manifest[year] = {
            "orig_rows": len(orig),
            "perf_rows": len(perf),
            "last_period": last_period,
            "release": cfg["data"]["release"]
        }
        print(f"  Vintage {year}: {len(orig)} orig loans, {len(perf)} perf records, last period {last_period}")

    manifest_path = interim / "ingest_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Ingestion complete. Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
