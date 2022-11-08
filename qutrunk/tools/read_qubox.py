"""Get Qubox configure info."""

import os
from pathlib import Path

import yaml


def get_qubox_setting():
    """Get Qubox configure info."""
    cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_file = Path(cur_path) / "config" / "qubox.yaml"

    with open(yaml_file) as f:
        yaml_content = yaml.load(f, Loader=yaml.FullLoader)
        return yaml_content
