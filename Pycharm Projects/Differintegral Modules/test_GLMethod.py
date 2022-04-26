import math
import time
import unittest
import numpy
import matplotlib.pyplot as plt

from tqdm import tqdm

from GLMethod import GLMethod


class TestGLMethod(unittest.TestCase):

    integration_step = 0.001
    number_of_points = 5000

    differential_equation = "-5 * math.pow(numpy.e, x)"
    variable = "x"

    number_of_verifications = 1

    def verification_dynamics(self, x):
        return -5 * math.pow(numpy.e, x)

    def verification_first_order_solution(self, x_values):
        return -1 * numpy.log(-5 * ((-1 * (1 / (5 * numpy.e))) - x_values))

    def verification_second_order_solution(self, x_values):
        z = 1 + 10 * numpy.e
        sqrt_z = numpy.sqrt(z)
        return numpy.log(-0.1 * z * (-1 + (numpy.tanh(0.5 * (sqrt_z * x_values - 2 * numpy.arctanh(1 / sqrt_z))) ** 2)))

    def test_string_init(self):

        model = GLMethod(equation=self.differential_equation, variable="x")

        self.assertEqual(model.system_dynamics, self.differential_equation, "System dynamics should contain the "
                                                                            "dynamics equation")
        self.assertEqual(model.dynamics_variable, self.variable, "Dynamics variable should contain the variable "
                                                                 "used in the system dynamics equation")

    def test_function_init(self):

        model = GLMethod(equation=self.verification_dynamics)

        self.assertEqual(model.system_dynamics, self.verification_dynamics,
                         "System dynamics should contain the function "
                         "that returns the result of the dynamics "
                         "equation at a particular point")

    def test_fractional_order_point_solution(self):
        models = [GLMethod(equation=self.differential_equation, variable=self.variable),
                  GLMethod(equation=self.verification_dynamics)]

        for model_count, model in enumerate(models):

            for step in numpy.divide(numpy.random.randint(low=1, high=100000, size=self.number_of_verifications),
                                     100000000):

                print("Step size: {step_size}".format(step_size=step))
                solution_cache = numpy.zeros(self.number_of_points)
                solution_cache[0] = 1.0
                for point in tqdm(range(1, self.number_of_points)):
                    solution_cache[point] = model.fractional_order_point_solution(1,
                                                                                  step=step,
                                                                                  iteration=point,
                                                                                  y_cache=solution_cache[:point],
                                                                                  previous_solution=solution_cache[
                                                                                      point - 1])

                expected_x_values = numpy.multiply(
                    numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                    step)
                expected_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))

                numpy.testing.assert_allclose(solution_cache,
                                              expected_y_values,
                                              rtol=0,
                                              atol=step * 10,
                                              err_msg="Y values do not align with the expected y values.\n"
                                                      "The step size is: {step_size}\n"
                                                      "The model is index number: {model_number}".format(step_size=step,
                                                                                                         model_number=model_count))

    def test_provided_dynamics_fractional_order_point_solution(self):
        models = [GLMethod(equation=self.differential_equation, variable=self.variable),
                  GLMethod(equation=self.verification_dynamics)]

        for model_count, model in enumerate(models):

            for step in numpy.divide(numpy.random.randint(low=1, high=100000, size=self.number_of_verifications),
                                     100000000):
                print("Step size: {step_size}".format(step_size=step))
                first_order_solution = numpy.zeros(self.number_of_points)
                first_order_solution[0] = 1.0
                second_order_solution = numpy.zeros(self.number_of_points)
                second_order_solution[0] = 1.0

                for point in tqdm(range(1, self.number_of_points)):
                    first_order_solution[point] = \
                        model.provided_dynamics_fractional_order_point_solution(alpha=1,
                                                                                step=step,
                                                                                iteration=point,
                                                                                dynamics=self.verification_dynamics(
                                                                                    first_order_solution[point - 1]),
                                                                                y_cache=first_order_solution[:point])

                expected_x_values = numpy.multiply(
                    numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                    step)
                expected_first_order_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))

                numpy.testing.assert_allclose(first_order_solution,
                                              expected_first_order_y_values,
                                              rtol=0,
                                              atol=step * 10,
                                              err_msg="Y values do not align with the expected y values.\n"
                                                      "The step size is: {step_size}\n"
                                                      "The model is index number: {model_number}".format(step_size=step,
                                                                                                         model_number=model_count))

    def test_fractional_order_solution_to_first_order(self):
        models = [GLMethod(equation=self.differential_equation, variable=self.variable),
                  GLMethod(equation=self.verification_dynamics)]

        for model_count, model in enumerate(models):
            for step in numpy.divide(numpy.random.randint(low=1, high=100000, size=self.number_of_verifications),
                                     100000000):
                print("Step size: {step_size}".format(step_size=step))

                x_values, y_values = model.fractional_order_solution_to_first_order(alpha=1,
                                                                                    step=step,
                                                                                    number_of_iterations=self.number_of_points,
                                                                                    initial_conditions=[1.0])

                expected_x_values = numpy.multiply(
                    numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                    step)
                expected_first_order_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))

                numpy.testing.assert_allclose(y_values[-1, :],
                                              expected_first_order_y_values,
                                              rtol=0,
                                              atol=step * 10,
                                              err_msg="Y values do not align with the expected y values.\n"
                                                      "The step size is: {step_size}\n"
                                                      "The model is index number: {model_number}".format(step_size=step,
                                                                                                         model_number=model_count))

    def test_fractional_order_solution_from_first_to_nth_order(self):
        models = [GLMethod(equation=self.differential_equation, variable=self.variable),
                  GLMethod(equation=self.verification_dynamics)]

        for model_count, model in enumerate(models):
            for step in numpy.divide(numpy.random.randint(low=1, high=100000, size=self.number_of_verifications),
                                     100000000):
                print("Step size: {step_size}".format(step_size=step))

                x_values, y_values = model.fractional_order_solution_from_first_to_nth_order(alpha=2.0,
                                                                                             step=step,
                                                                                             number_of_iterations=self.number_of_points,
                                                                                             initial_conditions=[1.0,
                                                                                                                 1.0])

                expected_x_values = numpy.multiply(
                    numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                    step)
                expected_second_order_y_values = self.verification_second_order_solution(numpy.array(expected_x_values))

            numpy.testing.assert_allclose(y_values[-1, :],
                                          expected_second_order_y_values,
                                          rtol=0,
                                          atol=step * 100,
                                          err_msg="Y values do not align with the expected y values.\n"
                                                  "The step size is: {step_size}\n"
                                                  "The model is index number: {model_number}".format(step_size=step,
                                                                                                     model_number=1))

    def test_fractional_order_solution(self):
        models = [GLMethod(equation=self.differential_equation, variable=self.variable),
                  GLMethod(equation=self.verification_dynamics)]

        for model_count, model in enumerate(models):
            for step in numpy.divide(numpy.random.randint(low=1, high=100000, size=self.number_of_verifications),
                                     100000000):
                print("Step size: {step_size}".format(step_size=step))
                print("Validate Fixed 1st Order and 2nd Order Solutions".format(step_size=step))
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
                                                                                               initial_conditions=[1.0, 1.0])
                numpy.testing.assert_allclose(y_values_first_order[-1, :],
                                              expected_first_order_y_values,
                                              rtol=0,
                                              atol=step * 10,
                                              err_msg="Y values do not align with the expected y values.\n"
                                                      "The step size is: {step_size}\n"
                                                      "The model is index number: {model_number}".format(step_size=step,
                                                                                                         model_number=1))
                numpy.testing.assert_allclose(y_values_second_order[-1, :],
                                              expected_second_order_y_values,
                                              rtol=0,
                                              atol=step * 100,
                                              err_msg="Y values do not align with the expected y values.\n"
                                                      "The step size is: {step_size}\n"
                                                      "The model is index number: {model_number}".format(step_size=step,
                                                                                                         model_number=1))
                for alpha in numpy.linspace(0.1, 3.1, num=30, endpoint=False):
                    print("Alpha: {alpha_value}".format(alpha_value=alpha))
                    initial_conditions = numpy.ones(math.ceil(alpha))
                    x_values, y_values = model.fractional_order_solution(alpha=alpha,
                                                                         integration_step=step,
                                                                         ending_point=self.number_of_points * step,
                                                                         initial_conditions=initial_conditions)
                    plt.plot(x_values, y_values[-1, :])
                plt.plot(x_values, expected_first_order_y_values, linestyle='dashed', markersize=24)
                plt.plot(x_values, expected_second_order_y_values, linestyle='dashed', markersize=24)
                plt.savefig('Plot Of All Alpha Values 0.1-3.0 In 0.1 Iterations With Step {step_values}.png'
                            .format(step_values=step), format='png', dpi=2400)
                plt.show()

    def test_accelerated_fractional_order_solution_to_first_order(self):
        models = [GLMethod(equation=self.differential_equation, variable=self.variable),
                  GLMethod(equation=self.verification_dynamics)]

        for model_count, model in enumerate(models):
            for step in numpy.divide(numpy.random.randint(low=1, high=100000, size=self.number_of_verifications),
                                     100000000):
                print("Step size: {step_size}".format(step_size=step))

                x_values, y_values = model.accelerated_fractional_order_solution_to_first_order(alpha=1,
                                                                                                step=step,
                                                                                                number_of_iterations=self.number_of_points,
                                                                                                initial_conditions=[1.0])

                expected_x_values = numpy.multiply(
                    numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                    step)
                expected_first_order_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))

                numpy.testing.assert_allclose(y_values[-1, :],
                                              expected_first_order_y_values,
                                              rtol=0,
                                              atol=step * 10,
                                              err_msg="Y values do not align with the expected y values.\n"
                                                      "The step size is: {step_size}\n"
                                                      "The model is index number: {model_number}".format(step_size=step,
                                                                                                         model_number=model_count))

    def test_accelerated_fractional_order_solution_from_first_to_nth_order(self):
        models = [GLMethod(equation=self.differential_equation, variable=self.variable),
                  GLMethod(equation=self.verification_dynamics)]

        for model_count, model in enumerate(models):
            for step in numpy.divide(numpy.random.randint(low=1, high=10000, size=self.number_of_verifications),
                                     100000000):
                print("Step size: {step_size}\nTolerance: {tolerance_val}".format(step_size=step,
                                                                                  tolerance_val=step * 100))

                x_values, y_values = model.accelerated_fractional_order_solution_from_first_to_nth_order(alpha=2.0,
                                                                                                         step=step,
                                                                                                         number_of_iterations=self.number_of_points,
                                                                                                         initial_conditions=[
                                                                                                             1.0, 1.0])

                expected_x_values = numpy.multiply(
                    numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                    step)
                expected_second_order_y_values = self.verification_second_order_solution(numpy.array(expected_x_values))

            numpy.testing.assert_allclose(y_values[-1, :],
                                          expected_second_order_y_values,
                                          rtol=0,
                                          atol=step * 100,
                                          err_msg="Y values do not align with the expected y values.\n"
                                                  "The step size is: {step_size}\n"
                                                  "The model is index number: {model_number}".format(step_size=step,
                                                                                                     model_number=1))

    def test_accelerated_fractional_order_solution(self):
        models = [GLMethod(equation=self.differential_equation, variable=self.variable),
                  GLMethod(equation=self.verification_dynamics)]

        for model_count, model in enumerate(models):
            for step in numpy.divide(numpy.random.randint(low=1, high=100000, size=self.number_of_verifications),
                                     100000000):
                step = 0.001
                print("Model: {model_number}\nStep size: {step_size}".format(step_size=step, model_number=model_count))
                print("Validate Fixed 1st Order and 2nd Order Solutions".format(step_size=step))
                expected_x_values = numpy.multiply(
                    numpy.linspace(0, self.number_of_points - 1, num=self.number_of_points),
                    step)
                expected_first_order_y_values = self.verification_first_order_solution(numpy.array(expected_x_values))
                expected_second_order_y_values = self.verification_second_order_solution(numpy.array(expected_x_values))
                x_values_first_order, y_values_first_order = model.accelerated_fractional_order_solution(alpha=1.0,
                                                                                             integration_step=step,
                                                                                             ending_point=self.number_of_points * step,
                                                                                             initial_conditions=[1.0])
                x_values_second_order, y_values_second_order = model.accelerated_fractional_order_solution(alpha=2.0,
                                                                                               integration_step=step,
                                                                                               ending_point=self.number_of_points * step,
                                                                                               initial_conditions=[1.0,
                                                                                                                   1.0])
                numpy.testing.assert_allclose(y_values_first_order[-1, :],
                                              expected_first_order_y_values,
                                              rtol=0,
                                              atol=step * 10,
                                              err_msg="Y values do not align with the expected y values.\n"
                                                      "The step size is: {step_size}\n"
                                                      "The model is index number: {model_number}".format(step_size=step,
                                                                                                         model_number=1))
                numpy.testing.assert_allclose(y_values_second_order[-1, :],
                                              expected_second_order_y_values,
                                              rtol=0,
                                              atol=step * 100,
                                              err_msg="Y values do not align with the expected y values.\n"
                                                      "The step size is: {step_size}\n"
                                                      "The model is index number: {model_number}".format(step_size=step,
                                                                                                         model_number=1))
                print("Validate Non-Fixed 0.1st Order to 3.0rd Order Solutions".format(step_size=step))
                for alpha in numpy.linspace(0.1, 3.1, num=30, endpoint=False):
                    print("Alpha: {alpha_value}".format(alpha_value=alpha))
                    initial_conditions = numpy.ones(math.ceil(alpha))
                    x_values, y_values = model.accelerated_fractional_order_solution(alpha=alpha,
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
                    'Accelerated Plot Of All Alpha Values 0.1-3.0 In 0.1 Iterations With Step {step_values} - {identify}.png'
                        .format(step_values=step, identify=int(time.time())), format='png', dpi=2400)
                plt.show()
