import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_qubox_setting():
    """Get Qubox configure info."""
    with open(BASE_DIR + "/config/qubox.yaml") as f:
        yaml_content = yaml.load(f, Loader=yaml.FullLoader)
        return yaml_content
