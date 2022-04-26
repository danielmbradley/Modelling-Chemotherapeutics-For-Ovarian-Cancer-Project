import numpy
import math
from tqdm import tqdm


class EulersMethod:

    def __init__(self, equation: None, variable: str = None):
        self.system_dynamics = equation
        self.dynamics_variable = variable

    def first_order_point_solution(self, y: float, gradient: float, step: float):
        return y + (step * gradient)

    def euler_first_order_solution_string_dynamics(self,
                                                   number_of_iterations: int,
                                                   integration_step: float,
                                                   initial_condition: float,
                                                   starting_point: float = 0,):
        x_values = numpy.zeros(number_of_iterations)
        y_values = numpy.zeros(number_of_iterations)

        x_values[0] = starting_point
        y_values[0] = initial_condition

        for x in tqdm(range(1, number_of_iterations)):
            dynamics_equation = self.system_dynamics

            x_values[x] = x_values[x - 1] + integration_step

            y_values[x] = self.first_order_point_solution(y_values[x - 1],
                                                          eval(dynamics_equation.replace(self.dynamics_variable,
                                                                                         str(y_values[x - 1]))),
                                                          integration_step)
        return x_values, y_values

    def euler_first_order_solution_function_dynamics(self,
                                                     number_of_iterations: int,
                                                     integration_step: float,
                                                     initial_condition: float,
                                                     starting_point: float = 0):
        x_values = numpy.zeros(number_of_iterations)
        y_values = numpy.zeros(number_of_iterations)

        x_values[0] = starting_point
        y_values[0] = initial_condition

        for x in tqdm(range(1, number_of_iterations)):
            x_values[x] = x_values[x - 1] + integration_step

            y_values[x] = self.first_order_point_solution(y_values[x - 1],
                                                          self.system_dynamics(y_values[x - 1]),
                                                          integration_step)
        return x_values, y_values

    def euler_first_order_solution(self,
                                   integration_step: float,
                                   initial_condition: float,
                                   ending_point: int,
                                   starting_point: int = 0) -> ([float], [float]):

        number_of_iterations = math.ceil((ending_point - starting_point) / integration_step) + 1

        if callable(self.system_dynamics) and self.dynamics_variable is None:
            return self.euler_first_order_solution_function_dynamics(number_of_iterations=number_of_iterations,
                                                                     integration_step=integration_step,
                                                                     starting_point=starting_point,
                                                                     initial_condition=initial_condition)
        elif type(self.system_dynamics) == str and type(self.dynamics_variable):
            return self.euler_first_order_solution_string_dynamics(number_of_iterations=number_of_iterations,
                                                                   integration_step=integration_step,
                                                                   starting_point=starting_point,
                                                                   initial_condition=initial_condition)
        else:
            print("system_dynamics has been incorrectly defined - please pass a function that returns a single float "
                  "or a string and variable to be substituted and evaluated")
            return None

    def euler_nth_order_solution_string_dynamics(self,
                                                 number_of_iterations: int,
                                                 integration_step: float,
                                                 order: float,
                                                 initial_conditions: [float],
                                                 starting_point: int = 0):

        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((len(initial_conditions), number_of_iterations))

        y_values[:, 0] = initial_conditions
        x_values[0] = starting_point

        for x in tqdm(range(1, number_of_iterations)):
            for current_order in range(0, order):

                dynamics_equation = self.system_dynamics

                x_values[x] = x_values[x - 1] + integration_step

                if current_order == 0:
                    y_values[current_order, x] = self.first_order_point_solution(y_values[current_order, x - 1],
                                                                                 eval(dynamics_equation.replace(
                                                                                     self.dynamics_variable,
                                                                                     str(y_values[-1, x - 1]))),
                                                                                 integration_step)
                else:
                    y_values[current_order, x] = self.first_order_point_solution(y_values[current_order, x - 1],
                                                                                 y_values[current_order - 1, x - 1],
                                                                                 integration_step)
        return x_values, y_values

    def euler_nth_order_solution_function_dynamics(self,
                                                   number_of_iterations: int,
                                                   integration_step: float,
                                                   order: float,
                                                   initial_conditions: [float],
                                                   starting_point: int = 0):

        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((len(initial_conditions), number_of_iterations))

        y_values[:, 0] = initial_conditions
        x_values[0] = starting_point

        for x in tqdm(range(1, number_of_iterations)):
            for current_order in range(0, order):

                x_values[x] = x_values[x - 1] + integration_step

                if current_order == 0:
                    y_values[current_order, x] = self.first_order_point_solution(y_values[current_order, x - 1],
                                                                                 self.system_dynamics(
                                                                                     y_values[-1, x - 1]),
                                                                                 integration_step)
                else:
                    y_values[current_order, x] = self.first_order_point_solution(y_values[current_order, x-1],
                                                                                 y_values[current_order - 1, x-1],
                                                                                 integration_step)

        return x_values, y_values

    def euler_nth_order_solution(self,
                                 integration_step: float,
                                 initial_conditions: [float],
                                 ending_point: int,
                                 order: int,
                                 starting_point: int = 0) -> ([float], [[float]]):

        assert len(initial_conditions) == order, "Please check the parameters: number of initial conditions and order "\
                                                 "should match but do not "

        number_of_iterations = math.ceil((ending_point - starting_point) / integration_step) + 1

        if callable(self.system_dynamics) and self.dynamics_variable is None:
            x_values, y_values = self.euler_first_order_solution_function_dynamics(number_of_iterations=number_of_iterations,
                                                                                   integration_step=integration_step,
                                                                                   order=order,
                                                                                   initial_condition=initial_conditions,
                                                                                   starting_point=starting_point)
        elif type(self.system_dynamics) == str and type(self.dynamics_variable):
            x_values, y_values = self.euler_first_order_solution_string_dynamics(number_of_iterations=number_of_iterations,
                                                                                 integration_step=integration_step,
                                                                                 order=order,
                                                                                 initial_condition=initial_conditions,
                                                                                 starting_point=starting_point)
        else:
            print("system_dynamics has been incorrectly defined - please pass a function that returns a single float "
                  "or a string and variable to be substituted and evaluated")
            return None

        return x_values, y_values[-1, :]