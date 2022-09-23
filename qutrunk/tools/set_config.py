import os
import argparse


def set_config(yaml_file, json_str):
    """Write the json to file config/qubox.yaml.

    Args:
        yaml_file: Yaml file path.
        json_str: Json content.
    """
    # 将json转成要写入的内容
    file_str = ""
    for key in json_str:
        value = json_str[key]
        file_str += f"{key}: {value}\n"

    # 修改yaml文件内容
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(file_str)


def str_to_dict(args_json):
    """Convert strng to dict.

    Args:
        args_json: Json string.

    Return:
        Dict object.
    """
    json_str = {}
    args_json = args_json.split(":")
    json_str["ip"], json_str["port"] = args_json[0], int(args_json[1])
    return json_str


def set_box_ip(args_json=None):
    """Set Qubox IP.

    Args:
        args_json: QuBox config.

    Raises:
        Exception: If no parameters are passed in.
    """
    # 1 调用该函数，接收参数。
    if args_json:  # 判断是否传入参数
        # 2 读取传入字符内容并转为JSON格式

        # 3 修改config下面的yaml文件
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_file = os.path.join(BASE_DIR, "config", "qubox.yaml")  # 获取yaml文件路径
        json_str = str_to_dict(args_json)  # 获取json
        set_config(yaml_file, json_str)
        print("Done")
    else:
        raise Exception("未传参数")


if __name__ == "__main__":
    # 接收参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str)
    args = parser.parse_args()
    # 执行修改
    set_box_ip((args.ip).replace(" ", "").replace("：", ":"))
