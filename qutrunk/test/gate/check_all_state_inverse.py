import math

from qutrunk.test.global_parameters import PRECISION


def check_all_state_inverse(res, res_box):
    if len(res) != len(res_box):
        return False

    for index in range(len(res)):
        amp_str = res[index]
        amp_str_box = res_box[index]
        real_str, image_str = amp_str.split(",")
        real_str_box, image_str_box = amp_str_box.split(",")
        test = float(image_str) - float(image_str_box)
        test1 = math.fabs(test)
        a = test1 > PRECISION
        if (
            math.fabs(float(real_str) - float(real_str_box)) > PRECISION
            or math.fabs(float(image_str) - float(image_str_box)) > PRECISION
        ):
            return False

    return True
