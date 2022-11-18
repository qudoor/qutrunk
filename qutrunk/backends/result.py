"""The result of measure."""


class MeasureQubit:
    def __init__(self, idx=0, value=0):
        self.idx = idx
        self.value = value


class MeasureQubits:
    def __init__(self):
        self.measure = []


class MeasureCount:
    def __init__(self, bit_str="", count=0):
        self.bit_str = bit_str
        self.count = count


class Result:
    def __init__(self):
        # store the object of MeasureQubits.
        self.measures = []
        # store the object of MeasureCount.
        self.measure_counts = []
