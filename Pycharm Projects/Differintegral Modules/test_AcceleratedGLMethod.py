import math
import time
import unittest
import numpy
import os
import timeit
import matplotlib.pyplot as plt

from GLMethodAccelerated import GLMethodAccelerated
from GLMethod import GLMethod


class TestGLMethodAccelerated(unittest.TestCase):

    integration_step = 0.001
    number_of_points = 5000

    # Important Note: Verify GLMethodAccelerated.cpp contains the following equation in the function "dynamics_equation"
    # This is an assumption that has been made in all following tests
    # differential_equation = "-5 * math.pow(numpy.e, x)"

    number_of_verifications = 100

    def verification_first_order_solution(self, x_values):
        return -1 * numpy.log(-5 * ((-1 * (1 / (5 * numpy.e))) - x_values))

    def verification_second_order_solution(self, x_values):
        z = 1 + 10 * numpy.e
        sqrt_z = numpy.sqrt(z)
        return numpy.log(-0.1 * z * (-1 + (numpy.tanh(0.5 * (sqrt_z * x_values - 2 * numpy.arctanh(1 / sqrt_z))) ** 2)))

    def test_function_init(self):
        if os.path.exists("GLMethodAccelerated.dylib"):
            os.remove("GLMethodAccelerated.dylib")
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        else:
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        model = GLMethodAccelerated()
        self.assertIsInstance(model, GLMethodAccelerated, "GLMethodAccelerated Object Not Created Successfully")

    def test_fractional_order_solution(self):
        if os.path.exists("GLMethodAccelerated.dylib"):
            os.remove("GLMethodAccelerated.dylib")
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        else:
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')

        model = GLMethodAccelerated()

        for step in numpy.divide(numpy.random.randint(low=1, high=100000, size=self.number_of_verifications),
                                 100000000):
            print("Step size: {step_size}".format(step_size=step))
            print("Validate Fixed 1st Order and 2nd Order Solutions")
            expected_x_values = numpy.multiply(
                numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                step)
            expected_first_order_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))
            expected_second_order_y_values = self.verification_second_order_solution(numpy.array(expected_x_values))
            x_values_first_order, y_values_first_order = model.fractional_order_solution(alpha=1.0,
                                                                                         integration_step=step,
                                                                                         ending_point=self.number_of_points * step,
                                                                                         initial_conditions=[1.0])
            x_values_second_order, y_values_second_order = model.fractional_order_solution(alpha=2.0,
                                                                                           integration_step=step,
                                                                                           ending_point=self.number_of_points * step,
                                                                                           initial_conditions=[1.0,
                                                                                                               1.0])
            numpy.testing.assert_allclose(y_values_first_order[-1, :],
                                          expected_first_order_y_values,
                                          rtol=0,
                                          atol=step * 10,
                                          err_msg="First Order Y values do not align with the expected y values.\n"
                                                  "The step size is: {step_size}\n".format(step_size=step))
            numpy.testing.assert_allclose(y_values_second_order[-1, :],
                                          expected_second_order_y_values,
                                          rtol=0,
                                          atol=step * 100,
                                          err_msg="Second Order Y values do not align with the expected y values.\n"
                                                  "The step size is: {step_size}\n".format(step_size=step))

            print("Validate Non-Fixed 0.1st Order to 3.0rd Order Solutions")
            for alpha in numpy.linspace(0.1, 3.1, num=30, endpoint=False):
                print("Alpha: {alpha_value}".format(alpha_value=alpha))
                initial_conditions = numpy.ones(math.ceil(alpha)).tolist()
                x_values, y_values = model.fractional_order_solution(alpha=alpha,
                                                                     integration_step=step,
                                                                     ending_point=self.number_of_points * step,
                                                                     initial_conditions=initial_conditions)

                plt.plot(x_values, y_values[-1, :])
            expected_x_values = numpy.multiply(
                numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                step)
            expected_first_order_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))
            expected_second_order_y_values = self.verification_second_order_solution(numpy.array(expected_x_values))
            plt.plot(x_values, expected_first_order_y_values, linestyle='dashed', markersize=24)
            plt.plot(x_values, expected_second_order_y_values, linestyle='dashed', markersize=24)
            plt.savefig(
                'C++ Accelerated Plot Of All Alpha Values 0.1-3.0 In 0.1 Iterations With Step {step_values} - {identify}.png'
                    .format(step_values=step, identify=int(time.time())), format='png', dpi=2400)
            plt.show()

    def test_speed_test(self):
        if os.path.exists("GLMethodAccelerated.dylib"):
            os.remove("GLMethodAccelerated.dylib")
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        else:
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')

        accelerated_model = GLMethodAccelerated()
        model = GLMethod("-5 * math.pow(numpy.e, x)", "x")

        def run_accelerated():
            x_values, y_values = accelerated_model.fractional_order_solution(alpha=3.7,
                                                                             integration_step=self.integration_step,
                                                                             ending_point=self.number_of_points * self.integration_step,
                                                                             initial_conditions=[1.0, 1.0, 1.0, 1.0])

        def run_standard():
            parameters = [3.7, 1.0, 1.0, 1.0, 1.0]
            x_values, y_values = model.optimise_accelerated_fractional_order_solution(parameters=parameters,
                                                                                      integration_step=self.integration_step,
                                                                                      ending_point=self.number_of_points * self.integration_step)

        accelerated_speed = timeit.timeit(run_accelerated, number=10)
        standard_speed = timeit.timeit(run_standard, number=10)

        print("Accelerated function speed: {speed}".format(speed=accelerated_speed))
        print("Standard function speed: {speed}".format(speed=standard_speed))
        print("Speed up of {factor} times".format(factor=standard_speed/accelerated_speed))

        self.assertGreater(standard_speed, accelerated_speed, "Accelerated function is slower than standard version.")