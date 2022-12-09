import os.path

import yaml


def _get_config():
    config_file = os.path.dirname(__file__) + "/config.yaml"
    with open(config_file, "r") as f:
        return yaml.full_load(f)


configuration = _get_config()
