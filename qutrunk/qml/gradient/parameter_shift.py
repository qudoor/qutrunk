import numpy as np

def parameter_shift(circuit_func, input_data, shift=np.pi/2):
    """ 
    Backward pass computation, calculate the gradient of quantum circuit by parameter shift rule.
    """
    input_list = np.array(input_data.tolist())
    
    shift_right = input_list + np.ones(input_list.shape) * shift
    shift_left = input_list - np.ones(input_list.shape) * shift
    
    gradients = []
    for i in range(len(input_list)):
        expectation_right = circuit_func(shift_right[i])
        expectation_left  = circuit_func(shift_left[i])
        
        gradient = np.array([expectation_right]) - np.array([expectation_left])
        gradients.append(gradient)
    gradients = np.array(gradients).T
    return gradients