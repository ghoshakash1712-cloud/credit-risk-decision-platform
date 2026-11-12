from pathlib import Path
import json
import yaml

def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def load_config(path: str = "config.yaml") -> dict:
    repo_root = get_repo_root()
    config_file = repo_root / path
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def project_path(cfg: dict, key: str) -> Path:
    repo_root = get_repo_root()
    rel = cfg.get("paths", {}).get(key, key)
    return repo_root / rel

def load_schema(name: str) -> list:
    repo_root = get_repo_root()
    schema_file = repo_root / "schema" / f"{name}_columns.json"
    with open(schema_file, "r", encoding="utf-8") as f:
        return json.load(f)
