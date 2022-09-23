import os

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_qubox_setting():
    """Get Qubox configure info."""
    yaml_file = os.path.join(BASE_DIR, "config", "qubox.yaml")  # yaml文件路径
    with open(yaml_file) as f:
        yaml_content = yaml.load(f, Loader=yaml.FullLoader)
        return "{ip}:{port}".format(ip=yaml_content["ip"], port=yaml_content["port"])
