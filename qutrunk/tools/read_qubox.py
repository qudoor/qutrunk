"""Get Qubox configure info."""

from pathlib import Path

import yaml


def get_qubox_setting():
    """Get Qubox configure info."""
    yaml_file = Path.cwd().parent / "config" / "qubox.yaml"
    with open(yaml_file) as f:
        yaml_content = yaml.load(f, Loader=yaml.FullLoader)
        return yaml_content
