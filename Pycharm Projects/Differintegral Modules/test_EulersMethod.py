import unittest
import math
import numpy
from EulersMethod import EulersMethod
import matplotlib.pyplot as plt


class TestEulersMethod(unittest.TestCase):

    def verification_dynamics(self, x):
        return -5 * math.pow(numpy.e, x)

    def verification_first_order_solution(self, x_values):
        return -1 * numpy.log(-5 * ((-1 * (1 / (5 * numpy.e))) - x_values))

    def verification_second_order_solution(self, x_values):
        z = 1 + 10 * numpy.e
        sqrt_z = numpy.sqrt(z)
        return numpy.log(-0.1 * z * (-1 + (numpy.tanh(0.5 * (sqrt_z * x_values - 2 * numpy.arctanh(1 / sqrt_z))) ** 2)))

    integration_step = 0.000001
    number_of_points = 1000000

    differential_equation = "-5 * math.pow(numpy.e, x)"
    variable = "x"

    def test_string_init(self):

        model = EulersMethod(equation=self.differential_equation, variable="x")

        self.assertEqual(model.system_dynamics, self.differential_equation, "System dynamics should contain the "
                                                                            "dynamics equation")
        self.assertEqual(model.dynamics_variable, self.variable, " Dynamics variable should contain the variable "
                                                                "used in the system dynamics equation")

    def test_function_init(self):

        model = EulersMethod(equation=self.verification_dynamics)

        self.assertEqual(model.system_dynamics, self.verification_dynamics, "System dynamics should contain the function "
                                                                            "that returns the result of the dynamics "
                                                                            "equation at a particular point")

    def test_first_order_point_solution(self):
        model = EulersMethod(equation=self.differential_equation, variable="x")

        random_y_values = numpy.random.randint(low=0, high=9223372036854775807, size=10)
        random_gradient_values = numpy.random.randint(low=0, high=9223372036854775807, size=10)
        random_step_values = numpy.random.randint(low=0, high=9223372036854775807, size=10)

        for y in random_y_values:
            for gradient in random_gradient_values:
                for step in random_step_values:
                    self.assertEqual(model.first_order_point_solution(y=y, gradient=gradient, step=step),
                                     y+(gradient*step), "Point soluton is not correct for variables "
                                                        "y: {y}, gradient: {gradient}, step: {step}"
                                     .format(y=y, gradient=gradient, step=step))

    def test_euler_first_order_solution_dynamics_function(self):
        expected_x_values = numpy.multiply(numpy.linspace(0, self.number_of_points-1, num=self.number_of_points), self.integration_step)
        expected_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))

        model = EulersMethod(equation=self.verification_dynamics)

        x_values, y_values = model.euler_first_order_solution_function_dynamics(number_of_iterations=self.number_of_points,
                                                                                integration_step=self.integration_step,
                                                                                starting_point=0,
                                                                                initial_condition=1.0)
        numpy.testing.assert_allclose(x_values, expected_x_values,
                                      rtol=0,
                                      atol=self.integration_step*(1*(10**-3)),
                                      err_msg="X values do not align with the expected x values")

        numpy.testing.assert_allclose(y_values, expected_y_values, rtol=0, atol=self.integration_step*10,
                                      err_msg="Y values do not align with the expected y values")

    def test_euler_first_order_solution_string_evaluation(self):
        expected_x_values = numpy.multiply(numpy.linspace(0, self.number_of_points-1, num=self.number_of_points), self.integration_step)
        expected_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))

        model = EulersMethod(equation=self.differential_equation, variable=self.variable)

        x_values, y_values = model.euler_first_order_solution_string_dynamics(number_of_iterations=self.number_of_points,
                                                                              integration_step=self.integration_step,
                                                                              starting_point=0,
                                                                              initial_condition=1.0)

        numpy.testing.assert_allclose(x_values, expected_x_values,
                                      rtol=0,
                                      atol=self.integration_step*(1*(10**-3)),
                                      err_msg="X values do not align with the expected x values")

        numpy.testing.assert_allclose(y_values,
                                      expected_y_values,
                                      rtol=0,
                                      atol=self.integration_step*10,
                                      err_msg="Y values do not align with the expected y values")

    def test_euler_nth_order_solution_function_dynamics(self):
        expected_first_order_x_values = numpy.multiply(numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points), self.integration_step)
        expected_first_order_y_values = self.verification_first_order_solution(numpy.array(expected_first_order_x_values))

        expected_second_order_x_values = numpy.multiply(numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points), self.integration_step)
        expected_second_order_y_values = self.verification_second_order_solution(numpy.array(expected_second_order_x_values))

        expected_x_values = [expected_first_order_x_values, expected_second_order_x_values]
        expected_y_values = [expected_first_order_y_values, expected_second_order_y_values]

        model = EulersMethod(equation=self.verification_dynamics)

        for order in [1, 2]:
            x_values, y_values = model.euler_nth_order_solution_function_dynamics(number_of_iterations=self.number_of_points,
                                                                                  integration_step=self.integration_step,
                                                                                  order=order,
                                                                                  initial_conditions=numpy.ones(order))
            plt.plot(x_values, y_values[-1, :])
            plt.show()
            numpy.testing.assert_allclose(x_values,
                                          expected_x_values[order-1],
                                          rtol=0,
                                          atol=self.integration_step*(1*(10**-3)),
                                          err_msg="X values do not align with the expected x values")

            numpy.testing.assert_allclose(y_values[-1],
                                  expected_y_values[order-1],
                                  rtol=0,
                                  atol=self.integration_step * 100,
                                  err_msg="Y values do not align with the expected y values")


    def test_euler_nth_order_solution_string_dynamics(self):
        expected_first_order_x_values = numpy.multiply(numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points), self.integration_step)
        expected_first_order_y_values = self.verification_first_order_solution(numpy.array(expected_first_order_x_values))

        expected_second_order_x_values = numpy.multiply(numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points), self.integration_step)
        expected_second_order_y_values = self.verification_second_order_solution(numpy.array(expected_second_order_x_values))

        expected_x_values = [expected_first_order_x_values, expected_second_order_x_values]
        expected_y_values = [expected_first_order_y_values, expected_second_order_y_values]

        model = EulersMethod(equation=self.differential_equation, variable=self.variable)

        for order in [1, 2]:
            x_values, y_values = model.euler_nth_order_solution_string_dynamics(number_of_iterations=self.number_of_points,
                                                                                integration_step=self.integration_step,
                                                                                order=order,
                                                                                initial_conditions=numpy.ones(order))

            error = numpy.log(numpy.abs(((y_values[-1]-expected_y_values[order-1])/numpy.maximum(0.1, numpy.abs(expected_y_values[order-1])))*100))

            plt.plot(x_values, y_values[-1])
            plt.plot(expected_x_values[order-1], expected_y_values[order-1])
            plt.show()

            plt.plot(x_values, error)
            plt.show()

            numpy.testing.assert_allclose(x_values,
                                          expected_x_values[order-1],
                                          rtol=0,
                                          atol=self.integration_step*(1*(10**-3)),
                                          err_msg="X values do not align with the expected x values")

            numpy.testing.assert_allclose(y_values[-1],
                                  expected_y_values[order-1],
                                  rtol=0,
                                  atol=self.integration_step * 100,
                                  err_msg="Y values do not align with the expected y values")
