import numpy as np
class MLInterface:

    def __init__(self, iface):
        self._interface = iface

        if self._interface == 'pytorch':
            import torch as tc
            self.ml_backend = tc
        if self._interface == 'paddle':
            import paddle as pd
            self.ml_backend = pd

    def tensor(self, data: any, requires_grad=False):
        if self._interface == 'pytorch':
            if requires_grad:
                return self.ml_backend.tensor(data, requires_grad=True)
            return self.ml_backend.tensor(np.array(data))
        if self._interface == 'paddle':
            return self.ml_backend.to_tensor(np.array(data))