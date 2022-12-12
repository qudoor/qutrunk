# TODO: need to improve.
class MeasureQubits:
    """Measure Result of all qubits.

    Save the all qubits result of measure circuit running.

    Args:
        measure: 
            key: The target index of qubits.
            value: The index of qubits result.
    """
    
    def __init__(self):
        self.measure = {}

    def __getitem__(self, idx):
        """Return a Measure Result instance.

        Arg:
            idx: The target index of qubits.

        Returns:
            The index of qubits result.

        Raises:
          ValueError: expected integer index into measure.
        """
        if not isinstance(idx, int):
            raise ValueError("expected integer index into measure")
        return self.measure.get(idx, None)

    def simplify(self, idxs: set = None):
        """Get the measure result in dict format."""
        meas = []
        for key, value in self.measure.items():
            if idxs is None or key in idxs:
                meas.append({"idx": key, "val": value})
        return meas

    def bit_str(self, idxs: set = None):
        """Get the measure result in str format."""
        bitstr = ""
        measures = dict(sorted(self.measure.items(), key=lambda x: x[0], reverse=True))
        for key, value in measures.items():
            if idxs is None or key in idxs:
                bitstr += str(value)
        return bitstr

    def add_measure(self, idx: int, value: float):
        """Add the measure result."""
        self.measure[idx] = value

    def sort(self):
        """Add the measure result."""
        self.measure = dict(sorted(self.measure.items(), key=lambda x: x[0]))
        
class MeasureResult:
    """Multi Measure Result data.

    Save the result of multi measure circuit running.

    Args:
        measures: The measure result of MeasureQubits.
        measure_counts: 
            key: The str of all qubits.
            value: The counts of measure result.
    """
    
    def __init__(self):
        self.measures = []
        self.measure_counts = {}

    def add_measures(self, measure_qubits: MeasureQubits):
        """add measurement results."""
        self.measures.append(measure_qubits)

    def get_measure_counts(self, idxs: set = None) -> dict:
        """Get the number of times the measurement results appear."""
        if len(self.measure_counts) > 0:
            return self.measure_counts

        for meas in self.measures:
            bitstr = "0b"
            for key, value in meas.measure.items():
                if idxs is None or key in idxs:
                    bitstr += str(value)

            if bitstr in self.measure_counts:
                self.measure_counts[bitstr] += 1
            else:
                self.measure_counts[bitstr] = 1

        self.measure_counts = dict(sorted(self.measure_counts.items(), key=lambda x: x[0]))
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