import math

import numpy
from tqdm import tqdm

from CJA import CJA


class GLMethod:

    def __init__(self, equation: None, variable: str = None):
        self.system_dynamics = equation
        self.dynamics_variable = variable

    def first_order_point_solution(self, y, gradient, step):
        return y + (step * gradient)

    def fractional_order_point_solution(self, alpha, step, iteration, y_cache, previous_solution):
        dynamics_equation = self.system_dynamics
        iteration = int(iteration)
        summation = 0

        coefficient_generator = CJA(alpha)
        coefficient = iter(coefficient_generator)
        _ = next(coefficient)

        for j in range(1, iteration + 1):
            summation += next(coefficient) * y_cache[-j]
        if callable(self.system_dynamics) and self.dynamics_variable is None:
            return (step ** alpha) * self.system_dynamics(previous_solution) - summation
        elif type(self.system_dynamics) == str and type(self.dynamics_variable):
            return (step ** alpha) * eval(
                dynamics_equation.replace(self.dynamics_variable, str(previous_solution))) - summation

    def provided_dynamics_fractional_order_point_solution(self, alpha, step, iteration, y_cache, dynamics):
        iteration = int(iteration)
        summation = 0

        coefficient_generator = CJA(alpha)
        coefficient = iter(coefficient_generator)
        _ = next(coefficient)

        for j in range(1, iteration + 1):
            summation += next(coefficient) * y_cache[-j]
        return (step ** alpha) * dynamics - summation

    def fractional_order_solution_to_first_order(self,
                                                 alpha: float,
                                                 step: float,
                                                 number_of_iterations: int,
                                                 initial_conditions: [float]):

        assert len(initial_conditions) == 1, "Please check the parameters: number of initial conditions" \
                                             " and math.ceil(order) should match but do not"

        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((1, number_of_iterations))

        y_values[0] = initial_conditions[0]

        for x in tqdm(range(1, number_of_iterations)):
            x_values[x] = x_values[x - 1] + step

            y_values[0, x] = self.fractional_order_point_solution(alpha,
                                                                  step,
                                                                  x,
                                                                  y_values[0, :x],
                                                                  y_values[0, x - 1])
        return x_values, y_values

    def fractional_order_solution_from_first_to_nth_order(self,
                                                          alpha: float,
                                                          step: float,
                                                          number_of_iterations: int,
                                                          initial_conditions: [float]):
        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((len(initial_conditions), number_of_iterations))

        y_values[:, 0] = initial_conditions

        for x in tqdm(range(1, number_of_iterations)):
            for current_order in range(0, math.ceil(alpha)):

                x_values[x] = x_values[x - 1] + step

                if current_order == 0:
                    y_values[current_order, x] = self.fractional_order_point_solution(alpha=1.0,
                                                                                      step=step,
                                                                                      iteration=x,
                                                                                      y_cache=y_values[
                                                                                              current_order,
                                                                                              :x],
                                                                                      previous_solution=y_values[
                                                                                          -1, x - 1])
                elif (alpha % 1 == 0 and current_order > 0) or (alpha % 1 != 0 and current_order < math.floor(alpha)):
                    y_values[current_order, x] = self.provided_dynamics_fractional_order_point_solution(alpha=1.0,
                                                                                                        step=step,
                                                                                                        dynamics=
                                                                                                        y_values[
                                                                                                            current_order - 1, x - 1],
                                                                                                        iteration=x,
                                                                                                        y_cache=y_values[
                                                                                                                current_order,
                                                                                                                :x])
                else:
                    current_system_alpha = alpha - math.floor(alpha)
                    y_values[current_order, x] = self.provided_dynamics_fractional_order_point_solution(
                        alpha=current_system_alpha,
                        step=step,
                        dynamics=y_values[current_order - 1, x - 1],
                        iteration=x,
                        y_cache=y_values[current_order, :x])

        return x_values, y_values

    def fractional_order_solution(self,
                                  alpha: float,
                                  integration_step: float,
                                  ending_point: float,
                                  initial_conditions: [float]) -> ([float], [float]):

        assert len(initial_conditions) == math.ceil(alpha), "Please check the parameters: number of initial conditions" \
                                                            " and math.ceil(order) should match but do not"
        assert alpha >= 0, "Please check the parameters: alpha must be a positive value"

        number_of_iterations = math.ceil(ending_point / integration_step)

        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((len(initial_conditions), number_of_iterations))

        if alpha < 1:
            x_values, y_values = self.fractional_order_solution_to_first_order(alpha=alpha,
                                                                               step=integration_step,
                                                                               number_of_iterations=number_of_iterations,
                                                                               initial_conditions=initial_conditions)
        elif alpha >= 1:
            x_values, y_values = self.fractional_order_solution_from_first_to_nth_order(alpha=alpha,
                                                                                        step=integration_step,
                                                                                        number_of_iterations=number_of_iterations,
                                                                                        initial_conditions=initial_conditions)
        return x_values, y_values

    def accelerated_fractional_order_solution_to_first_order(self,
                                                             alpha: float,
                                                             step: float,
                                                             number_of_iterations: int,
                                                             initial_conditions: [float]):

        assert len(initial_conditions) == 1, "Please check the parameters: number of initial conditions" \
                                             " and math.ceil(order) should match but do not"

        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((1, number_of_iterations))

        y_values[0, 0] = initial_conditions[0]

        if alpha % 1 != 0:
            for x in tqdm(range(1, number_of_iterations)):
                x_values[x] = x_values[x - 1] + step

                y_values[0, x] = self.fractional_order_point_solution(alpha,
                                                                      step,
                                                                      x,
                                                                      y_values[0, :x],
                                                                      y_values[0, x - 1])
        else:
            for x in tqdm(range(1, number_of_iterations)):
                x_values[x] = x_values[x - 1] + step

                if callable(self.system_dynamics) and self.dynamics_variable is None:
                    y_values[0, x] = self.first_order_point_solution(
                        y_values[0, x - 1],
                        self.system_dynamics(y_values[0, x - 1]),
                        step)
                elif type(self.system_dynamics) == str and type(self.dynamics_variable):
                    dynamics_equation = self.system_dynamics
                    y_values[0, x] = self.first_order_point_solution(
                        y_values[0, x - 1],
                        eval(dynamics_equation.replace(self.dynamics_variable, str(y_values[0, x - 1]))),
                        step)

        return x_values, y_values

    def accelerated_fractional_order_solution_from_first_to_nth_order(self,
                                                                      alpha: float,
                                                                      step: float,
                                                                      number_of_iterations: int,
                                                                      initial_conditions: [float]):
        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((len(initial_conditions), number_of_iterations))

        y_values[:, 0] = initial_conditions

        for x in tqdm(range(1, number_of_iterations)):
            for current_order in range(0, math.ceil(alpha)):

                x_values[x] = x_values[x - 1] + step

                if current_order == 0:
                    if callable(self.system_dynamics) and self.dynamics_variable is None:
                        y_values[current_order, x] = self.first_order_point_solution(
                            y_values[current_order, x - 1],
                            self.system_dynamics(y_values[-1, x - 1]),
                            step)
                    elif type(self.system_dynamics) == str and type(self.dynamics_variable):
                        dynamics_equation = self.system_dynamics
                        y_values[current_order, x] = self.first_order_point_solution(
                            y_values[current_order, x - 1],
                            eval(
                                dynamics_equation.replace(
                                    self.dynamics_variable,
                                    str(y_values[-1, x - 1]))),
                            step)
                elif (alpha % 1 == 0 and current_order > 0) or (alpha % 1 != 0 and current_order < math.floor(alpha)):
                    y_values[current_order, x] = self.first_order_point_solution(y_values[current_order, x - 1],
                                                                                 y_values[current_order - 1, x - 1],
                                                                                 step)
                else:
                    current_system_alpha = alpha - math.floor(alpha)
                    y_values[current_order, x] = self.provided_dynamics_fractional_order_point_solution(
                        alpha=current_system_alpha,
                        step=step,
                        dynamics=y_values[current_order - 1, x - 1],
                        iteration=x,
                        y_cache=y_values[current_order, :x])

        return x_values, y_values

    def accelerated_fractional_order_solution(self,
                                              alpha: float,
                                              integration_step: float,
                                              ending_point: float,
                                              initial_conditions: [float]) -> ([float], [float]):

        assert len(initial_conditions) <= math.ceil(alpha), "Please check the parameters: number of initial conditions" \
                                                            " and math.ceil(order) should match but do not"
        assert alpha > 0, "Please check the parameters: alpha must be a positive value"

        number_of_iterations = math.ceil(ending_point / integration_step)

        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((len(initial_conditions), number_of_iterations))

        if alpha < 1:
            x_values, y_values = self.accelerated_fractional_order_solution_to_first_order(alpha=alpha,
                                                                                           step=integration_step,
                                                                                           number_of_iterations=number_of_iterations,
                                                                                           initial_conditions=initial_conditions)
        elif alpha >= 1:
            x_values, y_values = self.accelerated_fractional_order_solution_from_first_to_nth_order(alpha=alpha,
                                                                                                    step=integration_step,
                                                                                                    number_of_iterations=number_of_iterations,
                                                                                                    initial_conditions=initial_conditions)
        return x_values, y_values

    def optimise_accelerated_fractional_order_solution_to_first_order(self,
                                                                      alpha: float,
                                                                      step: float,
                                                                      number_of_iterations: int,
                                                                      initial_conditions: [float]):

        assert len(initial_conditions) == 1, "Please check the parameters: number of initial conditions" \
                                             " and math.ceil(order) should match but do not"

        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((1, number_of_iterations))

        y_values[0, 0] = initial_conditions[0]

        if alpha % 1 != 0:
            for x in range(1, number_of_iterations):
                x_values[x] = x_values[x - 1] + step

                y_values[0, x] = self.fractional_order_point_solution(alpha,
                                                                      step,
                                                                      x,
                                                                      y_values[0, :x],
                                                                      y_values[0, x - 1])
        else:
            for x in range(1, number_of_iterations):
                x_values[x] = x_values[x - 1] + step

                if callable(self.system_dynamics) and self.dynamics_variable is None:
                    y_values[0, x] = self.first_order_point_solution(
                        y_values[0, x - 1],
                        self.system_dynamics(y_values[0, x - 1]),
                        step)
                elif type(self.system_dynamics) == str and type(self.dynamics_variable):
                    dynamics_equation = self.system_dynamics
                    y_values[0, x] = self.first_order_point_solution(
                        y_values[0, x - 1],
                        eval(dynamics_equation.replace(self.dynamics_variable, str(y_values[0, x - 1]))),
                        step)

        return x_values, y_values

    def optimise_accelerated_fractional_order_solution_from_first_to_nth_order(self,
                                                                               alpha: float,
                                                                               step: float,
                                                                               number_of_iterations: int,
                                                                               initial_conditions: [float]):
        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((len(initial_conditions), number_of_iterations))

        y_values[:, 0] = initial_conditions

        for x in range(1, number_of_iterations):
            for current_order in range(0, math.ceil(alpha)):

                x_values[x] = x_values[x - 1] + step

                if current_order == 0:
                    if callable(self.system_dynamics) and self.dynamics_variable is None:
                        y_values[current_order, x] = self.first_order_point_solution(
                            y_values[current_order, x - 1],
                            self.system_dynamics(y_values[-1, x - 1]),
                            step)
                    elif type(self.system_dynamics) == str and type(self.dynamics_variable):
                        dynamics_equation = self.system_dynamics
                        y_values[current_order, x] = self.first_order_point_solution(
                            y_values[current_order, x - 1],
                            eval(
                                dynamics_equation.replace(
                                    self.dynamics_variable,
                                    str(y_values[-1, x - 1]))),
                            step)
                elif (alpha % 1 == 0 and current_order > 0) or (alpha % 1 != 0 and current_order < math.floor(alpha)):
                    y_values[current_order, x] = self.first_order_point_solution(y_values[current_order, x - 1],
                                                                                 y_values[current_order - 1, x - 1],
                                                                                 step)
                else:
                    current_system_alpha = alpha - math.floor(alpha)
                    y_values[current_order, x] = self.provided_dynamics_fractional_order_point_solution(
                        alpha=current_system_alpha,
                        step=step,
                        dynamics=y_values[current_order - 1, x - 1],
                        iteration=x,
                        y_cache=y_values[current_order, :x])

        return x_values, y_values

    def optimise_accelerated_fractional_order_solution(self,
                                                       parameters: [float],
                                                       integration_step: float,
                                                       ending_point: float) -> ([float], [float]):

        alpha = parameters[0]

        parameters = parameters[1:int(numpy.ceil(alpha)) + 1]

        number_of_iterations = math.ceil(ending_point / integration_step)

        x_values = numpy.zeros(number_of_iterations)

        y_values = numpy.zeros((len(parameters), number_of_iterations))

        assert len(parameters) <= math.ceil(alpha), "Please check the parameters: number of initial conditions" \
                                                    " and math.ceil(order) should match but do not"

        assert alpha > 0, "Please check the parameters: alpha must be a positive value"

        if alpha < 1:
            x_values, y_values = self.optimise_accelerated_fractional_order_solution_to_first_order(alpha=alpha,
                                                                                                    step=integration_step,
                                                                                                    number_of_iterations=number_of_iterations,
                                                                                                    initial_conditions=parameters)
        elif alpha >= 1:
            x_values, y_values = self.optimise_accelerated_fractional_order_solution_from_first_to_nth_order(
                alpha=alpha,
                step=integration_step,
                number_of_iterations=number_of_iterations,
                initial_conditions=parameters)
        return x_values, y_values
