import math

FLOAT_PRECISION = 1e-7

def equal(a, b):
    return math.fabs(a - b) < FLOAT_PRECISION