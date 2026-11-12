import json
from pathlib import Path
from src.common.config import load_config, project_path

def test_simulation_psi_properties():
    """TS-06: Within each scenario, PSI on the shocked feature never decreases as severity rises, and at least one severe scenario exceeds 0.20."""
    cfg = load_config()
    artifacts = project_path(cfg, "artifacts")
    sim_path = artifacts / "psi_simulated.json"
    assert sim_path.exists(), "psi_simulated.json must exist"

    with open(sim_path, "r", encoding="utf-8") as f:
        sim = json.load(f)

    severe_exceeds_threshold = False
    for scenario_name, severities in sim.items():
        # Score PSI should not decrease with severity
        s_mild = severities["mild"]["score_psi"]
        s_mod = severities["moderate"]["score_psi"]
        s_sev = severities["severe"]["score_psi"]
        assert s_mild <= s_mod + 1e-5, f"{scenario_name}: mild score PSI ({s_mild}) > mod ({s_mod})"
        assert s_mod <= s_sev + 1e-5, f"{scenario_name}: mod score PSI ({s_mod}) > sev ({s_sev})"

        # Check shocked feature PSI
        shock_cfg = cfg["drift"]["shocks"][scenario_name]
        col = shock_cfg.get("column") or shock_cfg.get("columns")[0]
        mild_psi = next(f["psi"] for f in severities["mild"]["features"] if f["name"] == col)
        sev_psi = next(f["psi"] for f in severities["severe"]["features"] if f["name"] == col)
        assert mild_psi <= sev_psi + 1e-5, f"{scenario_name}: mild PSI ({mild_psi}) > sev ({sev_psi})"

        if s_sev > 0.20 or sev_psi > 0.20:
            severe_exceeds_threshold = True

    assert severe_exceeds_threshold, "At least one severe scenario must exceed PSI 0.20 (TS-06)"
