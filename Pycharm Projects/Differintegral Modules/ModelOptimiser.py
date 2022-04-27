import random

import numpy
import pathos.multiprocessing as pmp
from scipy.optimize import minimize
from tqdm import tqdm

from GLMethod import GLMethod
from GLMethodAccelerated import GLMethodAccelerated


class ModelOptimiser:

    def __init__(self, model: GLMethodAccelerated, target_dataset: numpy.ndarray):
        self.model = model
        self.dataset = target_dataset
        self.past_alphas = numpy.array([])
        self.past_w_t = numpy.array([])

    def scipy_minimise(self,
                       integration_step,
                       ending_point,
                       min_order=0,
                       max_order=pmp.cpu_count()) -> (float, [float]):

        if isinstance(self.model, GLMethodAccelerated):
            if min_order > max_order:
                x = min_order
                min_order = max_order
                max_order = x

            if max_order < 1:
                max_order = 1

            if min_order < 0:
                min_order = 0

            if max_order - min_order > pmp.cpu_count():
                print("Not enough cores available - using max cores and limiting max alpha to search to {y}".format(
                    y=pmp.cpu_count()))
                max_order = pmp.cpu_count() + min_order

            initial_parameters = list()
            function_bounds = list()

            optimisable_function = lambda initial_params: numpy.sum(numpy.square(numpy.subtract(
                self.model.fractional_order_solution(initial_params[0], initial_params[1:], integration_step,
                                                     ending_point)[1][-1], self.dataset)))

            for alpha in range(min_order, max_order):
                initial_parameters.append([random.uniform(0, 1) + alpha])

            for order in range(min_order, max_order + 1):
                bounds = ((numpy.nextafter(order, +numpy.inf), order + 1), (0, None),)
                for x in range(order):
                    if order != min_order:
                        bounds = bounds + ((0, None),)
                    initial_parameters[order - min_order - 1] = numpy.append(initial_parameters[order - min_order - 1],
                                                                             random.random() * numpy.max(
                                                                                 self.dataset[-1][:]))
                if order != max_order:
                    function_bounds.append(bounds)

            parallelizable_function = lambda func: minimize(optimisable_function, initial_parameters[func],
                                                            method="L-BFGS-B",
                                                            bounds=function_bounds[func])
            results = []
            for bounds in list(range(0, len(function_bounds))):
                results.append(parallelizable_function(bounds))

            results = sorted(results, key=lambda x: x['fun'])

            return results[0].x
        else:
            if min_order > max_order:
                x = min_order
                min_order = max_order
                max_order = x

            if max_order < 1:
                max_order = 1

            if min_order < 0:
                min_order = 0

            if max_order - min_order > pmp.cpu_count():
                print("Not enough cores available - using max cores and limiting max alpha to search to {y}".format(
                    y=pmp.cpu_count()))
                max_order = pmp.cpu_count() + min_order

            pool = pmp.Pool(max_order)
            initial_parameters = list()
            function_bounds = list()

            optimisable_function = lambda initial_params: numpy.sum(numpy.square(numpy.subtract(
                self.model.optimise_accelerated_fractional_order_solution(initial_params, integration_step,
                                                                          ending_point)[
                    1][-1], self.dataset)))

            for alpha in range(min_order, max_order):
                initial_parameters.append([random.uniform(0, 1) + alpha])

            for order in range(min_order, max_order + 1):
                bounds = ((numpy.nextafter(order, +numpy.inf), order + 1), (0, None),)
                for x in range(order):
                    if order != min_order:
                        bounds = bounds + ((0, None),)
                    initial_parameters[order - min_order - 1] = numpy.append(initial_parameters[order - min_order - 1],
                                                                             random.random() * numpy.max(
                                                                                 self.dataset[-1][:]))
                if order != max_order:
                    function_bounds.append(bounds)

            parallelizable_function = lambda func: minimize(optimisable_function, initial_parameters[func],
                                                            method="L-BFGS-B",
                                                            bounds=function_bounds[func])

            results = pool.map(parallelizable_function, list(range(0, len(function_bounds))))
            results = sorted(results, key=lambda x: x['fun'])

            return results[0].x

    def noise_characterised_minimise(self,
                                     integration_step,
                                     ending_point,
                                     y_dataset=None,
                                     x_dataset=None) -> ([float]):

        if y_dataset == None:
            y_dataset = self.dataset[0]

        optimisable_function = lambda initial_params: self.model.cost_with_fractional_order_solution(initial_params,
                                                                                                     y_dataset,
                                                                                                     integration_step,
                                                                                                     ending_point,
                                                                                                     x_dataset=x_dataset)

        initial_parameters = [random.uniform(0.001, 0.999),
                              random.uniform(0, numpy.max(y_dataset))]

        for w_t in range(int(numpy.ceil(ending_point / integration_step))):
            initial_parameters.append(random.uniform(-0.5, 0.5))

        for w_t_alpha in range(int(numpy.ceil(ending_point / integration_step))):
            initial_parameters.append(random.uniform(-0.5, 0.5))

        for v_t in range(int(numpy.ceil(ending_point / integration_step))):
            initial_parameters.append(random.uniform(-0.5, 0.5))

        function_bounds = ((0.001, 0.999), (0, 5))

        for w_t in range(int(numpy.ceil(ending_point / integration_step))):
            function_bounds = function_bounds + ((-1, 1),)

        for w_t_alpha in range(int(numpy.ceil(ending_point / integration_step))):
            function_bounds = function_bounds + ((-1, 1),)

        for v_t in range(int(numpy.ceil(ending_point / integration_step))):
            function_bounds = function_bounds + ((-1, 1),)

        result = minimize(optimisable_function, initial_parameters, method="L-BFGS-B", bounds=function_bounds)

        alpha = (result.x[0])
        x_0 = (result.x[1])
        w_t = (result.x[2:int(numpy.ceil(ending_point / integration_step)) + 2])
        w_t_alpha = (result.x[int(numpy.ceil(ending_point / integration_step)) + 2:-int(
            numpy.ceil(ending_point / integration_step))])
        v_t = (result.x[-int(numpy.ceil(ending_point / integration_step)):])
        return alpha, x_0, w_t, w_t_alpha, v_t

    def monte_carlo_least_squares(self,
                                  integration_step,
                                  ending_point,
                                  min_order=0,
                                  max_order=pmp.cpu_count()) -> (float, [float]):
        estimated_alpha = []
        estimated_x_0 = []

        for data in tqdm(self.dataset):
            if isinstance(self.model, GLMethodAccelerated):
                if min_order > max_order:
                    x = min_order
                    min_order = max_order
                    max_order = x

                if max_order < 1:
                    max_order = 1

                if min_order < 0:
                    min_order = 0

                if max_order - min_order > pmp.cpu_count():
                    print("Not enough cores available - using max cores and limiting max alpha to search to {y}".format(
                        y=pmp.cpu_count()))
                    max_order = pmp.cpu_count() + min_order

                initial_parameters = list()
                function_bounds = list()

                optimisable_function = lambda initial_params: numpy.sum(numpy.square(numpy.subtract(
                    self.model.fractional_order_solution(initial_params[0], initial_params[1:], integration_step,
                                                         ending_point)[1][-1], data)))

                for alpha in range(min_order, max_order):
                    initial_parameters.append([random.uniform(0, 1) + alpha])

                for order in range(min_order, max_order + 1):
                    bounds = ((numpy.nextafter(order, +numpy.inf), order + 1), (0, None),)
                    for x in range(order):
                        if order != min_order:
                            bounds = bounds + ((0, None),)
                        initial_parameters[order - min_order - 1] = numpy.append(
                            initial_parameters[order - min_order - 1],
                            random.random() * numpy.max(data[-1][:]))
                    if order != max_order:
                        function_bounds.append(bounds)

                parallelizable_function = lambda func: minimize(optimisable_function, initial_parameters[func],
                                                                method="L-BFGS-B",
                                                                bounds=function_bounds[func])
                results = []
                for bounds in list(range(0, len(function_bounds))):
                    results.append(parallelizable_function(bounds))

                results = sorted(results, key=lambda x: x['fun'])

            else:
                if min_order > max_order:
                    x = min_order
                    min_order = max_order
                    max_order = x

                if max_order < 1:
                    max_order = 1

                if min_order < 0:
                    min_order = 0

                if max_order - min_order > pmp.cpu_count():
                    print("Not enough cores available - using max cores and limiting max alpha to search to {y}".format(
                        y=pmp.cpu_count()))
                    max_order = pmp.cpu_count() + min_order

                pool = pmp.Pool(max_order)
                initial_parameters = list()
                function_bounds = list()

                optimisable_function = lambda initial_params: numpy.sum(numpy.square(numpy.subtract(
                    self.model.optimise_accelerated_fractional_order_solution(initial_params, integration_step,
                                                                              ending_point)[
                        1][-1], data)))

                for alpha in range(min_order, max_order):
                    initial_parameters.append([random.uniform(0, 1) + alpha])

                for order in range(min_order, max_order + 1):
                    bounds = ((numpy.nextafter(order, +numpy.inf), order + 1), (0, None),)
                    for x in range(order):
                        if order != min_order:
                            bounds = bounds + ((0, None),)
                        initial_parameters[order - min_order - 1] = numpy.append(
                            initial_parameters[order - min_order - 1],
                            random.random() * numpy.max(data[-1][:]))
                    if order != max_order:
                        function_bounds.append(bounds)

                parallelizable_function = lambda func: minimize(optimisable_function, initial_parameters[func],
                                                                method="L-BFGS-B",
                                                                bounds=function_bounds[func])

                results = pool.map(parallelizable_function, list(range(0, len(function_bounds))))
                results = sorted(results, key=lambda x: x['fun'])

            estimated_alpha.append(results[0].x[0])
            estimated_x_0.append(results[0].x[1])
        return (estimated_alpha, estimated_x_0)


    def monte_carlo_noise_characterised_minimise(self,
                                                integration_step,
                                                ending_point,
                                                y_dataset,
                                                x_dataset) -> (float, [float]):
        estimated_alpha = []
        estimated_x_0 = []
        estimated_w_t = []
        estimated_w_t_alpha = []
        estimated_v_t = []

        iteration = 0

        for data in tqdm(y_dataset):
            if isinstance(self.model, GLMethodAccelerated):
                optimisable_function = lambda initial_params: self.model.cost_with_fractional_order_solution(
                    initial_params,
                    data,
                    integration_step,
                    ending_point,
                    x_dataset=x_dataset[iteration])

                initial_parameters = [random.uniform(0.001, 0.999),
                                      random.uniform(0, numpy.max(y_dataset))]

                for w_t in range(int(numpy.ceil(ending_point / integration_step))):
                    initial_parameters.append(random.uniform(-0.5, 0.5))

                for w_t_alpha in range(int(numpy.ceil(ending_point / integration_step))):
                    initial_parameters.append(random.uniform(-0.5, 0.5))

                for v_t in range(int(numpy.ceil(ending_point / integration_step))):
                    initial_parameters.append(random.uniform(-0.5, 0.5))

                function_bounds = ((0.001, 0.999), (0, 5))

                for w_t in range(int(numpy.ceil(ending_point / integration_step))):
                    function_bounds = function_bounds + ((-1, 1),)

                for w_t_alpha in range(int(numpy.ceil(ending_point / integration_step))):
                    function_bounds = function_bounds + ((-1, 1),)

                for v_t in range(int(numpy.ceil(ending_point / integration_step))):
                    function_bounds = function_bounds + ((-1, 1),)

                result = minimize(optimisable_function, initial_parameters, method="L-BFGS-B", bounds=function_bounds)

                alpha = (result.x[0])
                x_0 = (result.x[1])
                w_t = (result.x[2:int(numpy.ceil(ending_point / integration_step)) + 2])
                w_t_alpha = (result.x[int(numpy.ceil(ending_point / integration_step)) + 2:-int(
                    numpy.ceil(ending_point / integration_step))])
                v_t = (result.x[-int(numpy.ceil(ending_point / integration_step)):])

            estimated_alpha.append(alpha)
            estimated_x_0.append(x_0)
            estimated_w_t.append(w_t)
            estimated_w_t_alpha.append(w_t_alpha)
            estimated_v_t.append(v_t)

        return (estimated_alpha, estimated_x_0, estimated_w_t, estimated_w_t_alpha, estimated_v_t)
