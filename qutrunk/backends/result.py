# TODO: need to improve.
class MeasureQubit:
    def __init__(self, idx=0, value=0):
        self.idx = idx
        self.value = value


class MeasureQubits:
    def __init__(self):
        self.measure = []

    def __getitem__(self, index):
        return self.measure[index]

    def simplify(self, indexs: set=None):
        meas = []
        for m in self.measure:
            if indexs is None or m.idx in indexs:
                meas.append({"idx": m.idx, "val": m.value})
        return meas

    def bit_str(self, indexs: set=None):
        bitstr = "0b"
        for m in self.measure[::-1]:
            if indexs is None or m.idx in indexs:
                bitstr += str(m.value)
        return bitstr

class MeasureCount:
    def __init__(self, bit_str="", count=0):
        self.bitstr = bit_str
        self.count = count


class MeasureResult:
    def __init__(self):
        self.measures = []
        self.measure_counts = []
        
    def add_measures(self, measure_qubits):
        self.measures.append(measure_qubits)
        
    def get_measure_counts(self, indexs: set=None) -> MeasureCount:
        if len(self.measure_counts) > 0:
            return self.measure_counts
        
        measure_counts = {}
        for meas in self.measures:
            bitstr = ""
            for mea in meas.measure:
                if indexs is None or mea.idx in indexs:
                    bitstr += str(mea.value)
                
            if bitstr in measure_counts:
                measure_counts[bitstr] += 1
            else:
                measure_counts[bitstr] = 1
            
        measure_counts = dict(sorted(measure_counts.items(), key=lambda x : x[0]))
        for bitstr, count in measure_counts.items():
            mc = MeasureCount(bitstr, count)
            self.measure_counts.append(mc)
            
        return self.measure_counts
    
    def get_bitstrs(self, indexs: set=None):
        bit_strs = []
        for m in self.measures:
            bit_strs.append(m.bit_str(indexs))
            
        return bit_strs