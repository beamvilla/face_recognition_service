import yaml
import json
from typing import Dict, Any


def load_yaml_file(yaml_file_path: str) -> Dict[str, Any]:
    with open(yaml_file_path) as f:
        try:
            yaml_file = yaml.safe_load(f)
        except yaml.YAMLError:
            raise
    return yaml_file


def save_json(data: Dict[str, Any], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)