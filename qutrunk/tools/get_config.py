"""Get Qubox IP and Port."""

from pathlib import Path

import yaml


def get_qubox_setting():
    """Get Qubox IP and Port."""
    # the path of qubox.yaml
    yaml_file = Path.cwd().parent / "config" / "qubox.yaml"
    with open(yaml_file) as f:
        yaml_content = yaml.load(f, Loader=yaml.FullLoader)
        return f"{yaml_content['ip']}:{yaml_content['port']}"
