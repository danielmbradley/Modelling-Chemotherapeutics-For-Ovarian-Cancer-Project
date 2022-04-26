import ctypes
import numpy

class GLMethodAccelerated:

    def __init__(self):
        self._GLMethod = ctypes.CDLL('GLMethodAccelerated.dylib')
        self._GLMethod.fractional_order_dynamics_solution.argtypes = (ctypes.POINTER(ctypes.c_double),
                                                                      ctypes.c_int,
                                                                      ctypes.c_double,
                                                                      ctypes.c_double,
                                                                      ctypes.POINTER(ctypes.c_double))
        self._GLMethod.cost.argtypes = (ctypes.POINTER(ctypes.c_double),
                                        ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.c_int,
                                        ctypes.c_double,
                                        ctypes.c_double)
        self._GLMethod.cost.restype = ctypes.c_double

    def fractional_order_solution(self, alpha, initial_conditions, integration_step, ending_point)->([float], [[float]]):
        parameters = list(initial_conditions)
        parameters.insert(0, alpha)
        parameters = numpy.array(parameters)
        c_parameters = ctypes.c_double * len(parameters)
        empty_array = [0] * (2*int(numpy.ceil(ending_point / integration_step)))
        result_allocated = ctypes.c_double * (2*int(numpy.ceil(ending_point / integration_step)))
        c_result = result_allocated(*empty_array)
        self._GLMethod.fractional_order_dynamics_solution(c_parameters(*parameters),
                                                          ctypes.c_int(int(numpy.ceil(parameters[0])+1)),
                                                          ctypes.c_double(integration_step),
                                                          ctypes.c_double(ending_point),
                                                          c_result)
        c_result = list(c_result)
        c_result = numpy.array(c_result)

        return (numpy.array(c_result[0:int(numpy.ceil(ending_point / integration_step))]), numpy.array([c_result[int(numpy.ceil(ending_point / integration_step)):]]))


    def cost_with_fractional_order_solution(self, parameters, dataset, integration_step, ending_point, x_dataset=None) -> float:
        if x_dataset == None:
            x_dataset = [0]
            for x in range(1, len(dataset)):
                x_dataset.append(x_dataset[x-1]+integration_step)

        c_parameters = ctypes.c_double * len(parameters)
        c_y_dataset = ctypes.c_double * len(dataset)
        c_x_dataset = ctypes.c_double * len(x_dataset)

        cost = ctypes.c_double(self._GLMethod.cost(c_parameters(*parameters),
                                                   ctypes.c_int(int(numpy.ceil(parameters[0])+1)),
                                                   c_y_dataset(*dataset),
                                                   c_x_dataset(*x_dataset),
                                                   ctypes.c_int(len(dataset)),
                                                   ctypes.c_double(integration_step),
                                                   ctypes.c_double(ending_point))).value
        return cost