# TODO: need to improve.
class MeasureQubit:
    """Measure Result Single Qubit.

    Save the result of single measure qubit.

    Args:
        idx: The index of qubits.
        value: The index of qubits result.
    """
    
    def __init__(self, idx=0, value=0):
        self.idx = idx
        self.value = value


# TODO: need to improve.
class MeasureQubits:
    """Measure Result of all qubits.

    Save the all qubits result of measure circuit running.

    Args:
        measure: The measure result of MeasureQubit.
    """
    
    def __init__(self):
        self.measure = []

    def __getitem__(self, idx):
        """Return a MeasureQubit instance.

        Arg:
            idx: The index of MeasureQubit.

        Returns:
            MeasureQubit instance.

        Raises:
          ValueError: expected integer index into measure.
        """
        if not isinstance(idx, int):
            raise ValueError("expected integer index into measure")
        return self.measure[idx]

    def simplify(self, idxs: set = None):
        """Get the measure result in dict format."""
        meas = []
        for m in self.measure:
            if idxs is None or m.idx in idxs:
                meas.append({"idx": m.idx, "val": m.value})
        return meas

    def bit_str(self, idxs: set = None):
        """Get the measure result in str format."""
        bit_str = ""
        for m in reversed(self.measure):
            if idxs is None or m.idx in idxs:
                bit_str += str(m.value)
        return bit_str


class MeasureCount:
    """Counts Measure Result data.

    Save the Counts result of measure circuit running.

    Args:
        bitstr: The str of all qubits.
        count: The counts of measure result.
    """
    
    def __init__(self, bit_str="", count=0):
        self.bitstr = bit_str
        self.count = count


class MeasureResult:
    """Multi Measure Result data.

    Save the result of multi measure circuit running.

    Args:
        measures: The measure result of MeasureQubits.
        measure_counts: The counts measure result of measures.
    """
    
    def __init__(self):
        self.measures = []
        self.measure_counts = []

    def add_measures(self, measure_qubits: MeasureQubits):
        """add measurement results."""
        self.measures.append(measure_qubits)

    # TODO: have some problem.
    def get_measure_counts(self, idxs: set = None) -> MeasureCount:
        """Get the number of times the measurement results appear."""
        if self.measure_counts:
            return self.measure_counts

        measure_counts = {}
        for meas in self.measures:
            bitstr = "0b"
            for mea in meas.measure:
                if idxs is None or mea.idx in idxs:
                    bitstr += str(mea.value)

            if bitstr in measure_counts:
                measure_counts[bitstr] += 1
            else:
                measure_counts[bitstr] = 1

        measure_counts = dict(sorted(measure_counts.items(), key=lambda x: x[0]))
        for bitstr, count in measure_counts.items():
            mc = MeasureCount(bitstr, count)
            self.measure_counts.append(mc)

        return self.measure_counts

    def get_bitstrs(self, idxs: set = None):
        """Get the measure result in binary format."""
        bit_strs = []
        for m in self.measures:
            bit_strs.append("0b"+m.bit_str(idxs))

        return bit_strs
    
    def get_values(self, idxs: set = None):
        """Get the measure result of int."""
        bit_strs = []
        for m in self.measures:
            bitstr = m.bit_str(idxs)
            if bitstr:
                bit_strs.append(int(bitstr, base=2))

        return bit_strs