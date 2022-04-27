import os
import random
import time
import unittest

import matplotlib.pyplot as plt
import numpy

from GLMethod import GLMethod
from GLMethodAccelerated import GLMethodAccelerated
from ModelOptimiser import ModelOptimiser


class TestGLMethod(unittest.TestCase):
    measurement_noise_std = 0.1

    step = 0.01
    alpha = 0.7
    ending_point = 7
    initial_parameters = numpy.repeat(1.0, numpy.ceil(alpha))

    accelerated_model = GLMethodAccelerated()
    standard_model = GLMethod("-5 * math.pow(numpy.e, x)", "x")

    data = accelerated_model.fractional_order_solution(alpha, initial_parameters, step, ending_point)
    noise = numpy.random.normal(0, measurement_noise_std, data[1][-1].shape)
    noise_and_data = [data[1][-1] + noise]

    accelerated_optimisation = ModelOptimiser(accelerated_model, noise_and_data)
    standard_optimisation = ModelOptimiser(standard_model, noise_and_data)

    def test_scipy_minimise_accelerated(self):
        if os.path.exists("GLMethodAccelerated.dylib"):
            os.remove("GLMethodAccelerated.dylib")
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        else:
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        estimated_parameters = self.accelerated_optimisation.scipy_minimise(self.step, self.ending_point, max_order=2)
        self.assertAlmostEqual(estimated_parameters[0], self.alpha, places=2)
        self.assertAlmostEqual(estimated_parameters[1], self.initial_parameters[0], places=2)

    def test_scipy_minimise_standard(self):
        if os.path.exists("GLMethodAccelerated.dylib"):
            os.remove("GLMethodAccelerated.dylib")
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        else:
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        estimated_parameters = self.standard_optimisation.scipy_minimise(self.step, self.ending_point, max_order=2)
        self.assertAlmostEqual(estimated_parameters[0], self.alpha, places=2)
        self.assertAlmostEqual(estimated_parameters[1], self.initial_parameters[0], places=2)

    def test_noise_characterised_minimise(self):
        if os.path.exists("GLMethodAccelerated.dylib"):
            os.remove("GLMethodAccelerated.dylib")
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        else:
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        alpha, x_0, w_t, w_t_alpha, v_t = self.accelerated_optimisation.noise_characterised_minimise(self.step,
                                                                                                     self.ending_point)
        self.assertAlmostEqual(alpha / 10, self.alpha / 10, places=2)
        self.assertAlmostEqual(x_0 / 10, self.initial_parameters[0] / 10, places=2)
        numpy.testing.assert_allclose(v_t, self.noise,
                                      rtol=0,
                                      atol=self.step * 1000,
                                      err_msg="Estimated Noise Does Not Equal Measurement Noise\n")

    def test_noise_characterised_minimise_sparse(self):
        if os.path.exists("GLMethodAccelerated.dylib"):
            os.remove("GLMethodAccelerated.dylib")
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')
        else:
            os.system('g++ -dynamiclib -o GLMethodAccelerated.dylib GLMethodAccelerated.cpp -std=c++17')

        for x in range(5):
            sparse_noise_and_y_data = [self.data[1][-1] + self.noise]
            sparse_x_data = self.data[0]

            for i in range(0, 680):
                index = random.randint(5, len(sparse_x_data) - 1)
                sparse_x_data = numpy.delete(sparse_x_data, index)
                sparse_noise_and_y_data = numpy.delete(sparse_noise_and_y_data, index)

            alpha, x_0, w_t, w_t_alpha, v_t = self.accelerated_optimisation.noise_characterised_minimise(self.step,
                                                                                                         self.ending_point)

            plt.plot(sparse_x_data, sparse_noise_and_y_data, 'bo')

            modelled = self.accelerated_model.fractional_order_solution(alpha, [x_0], self.step, self.ending_point)

            plt.plot(modelled[0], modelled[1][0])
            plt.title("Estimate Of Parameters For A Sparse Dataset")
            plt.xlabel("t")
            plt.ylabel("x(t)")
            plt.savefig('Estimated System - {identify}.png'.format(identify=int(time.time())), format='png', dpi=2400)
            plt.show()

            plt.hist(w_t, density=True, bins=10)
            plt.title("Estimate Noise In Model")
            plt.xlabel("w_t")
            plt.savefig('Estimated alpha values - {identify}.png'.format(identify=int(time.time())), format='png',
                        dpi=2400)
            plt.show()

            plt.hist(w_t_alpha, density=True, bins=10)
            plt.title("Estimate Of Order Alpha Noise")
            plt.xlabel("Alpha")
            plt.savefig('Estimated alpha values - {identify}.png'.format(identify=int(time.time())), format='png',
                        dpi=2400)
            plt.show()

            plt.hist(v_t, density=True, bins=10)
            plt.title("Estimate Of Measurement Noise")
            plt.xlabel("v_t")
            plt.savefig('Estimated alpha values - {identify}.png'.format(identify=int(time.time())), format='png',
                        dpi=2400)
            plt.show()

            # self.assertTrue(((alpha-self.alpha)<self.alpha*0.15), "Alpha Is Not Correct")
            # self.assertTrue(((x_0-self.initial_parameters[0])<self.initial_parameters[0]*0.15), "Initial Conditions Is Not Correct")
            # numpy.testing.assert_allclose(v_t, self.noise,
            #                               rtol=0,
            #                               atol=self.step * 1000,
            #                               err_msg="Estimated Noise Does Not Equal Measurement Noise\n")

    def test_monte_carlo_least_squares(self):
        noise = []

        for iteration in range(12000):
            noise.append(
                [self.data[1][-1] + numpy.random.normal(0, self.measurement_noise_std, self.data[1][-1].shape)])

        self.accelerated_optimisation.dataset = noise

        estimated_alpha, estimated_x_0 = self.accelerated_optimisation.monte_carlo_least_squares(self.step,
                                                                                                 self.ending_point,
                                                                                                 max_order=2)
        self.assertAlmostEqual(numpy.mean(estimated_alpha), self.alpha, places=2)
        self.assertAlmostEqual(numpy.mean(estimated_x_0), self.initial_parameters[0], places=2)

        plt.hist(estimated_alpha, density=True, bins=100)
        plt.title("Estimate Of Order Alpha In Population With Different Noise")
        plt.xlabel("Alpha")
        plt.savefig('Estimated alpha values - {identify}.png'.format(identify=int(time.time())), format='png', dpi=2400)
        plt.show()

        plt.hist(estimated_x_0, density=True, bins=100)
        plt.title("Estimate Of Initial Conditions In Population With Different Noise")
        plt.xlabel("Initial Conditions")
        plt.savefig('Estimated Initial Condition values - {identify}.png'.format(identify=int(time.time())),
                    format='png', dpi=2400)
        plt.show()

    def test_monte_carlo_noise_characterised_minimise_sparse(self):
        sparse_noise_and_y_data = []
        sparse_x_data = []

        for iteration in range(350):
            sparse_noise_and_y_data.append(
                [self.data[1][-1] + numpy.random.normal(0, self.measurement_noise_std, self.data[1][-1].shape)])
            sparse_x_data.append(self.data[0])

            for i in range(0, 680):
                index = random.randint(5, len(sparse_x_data[iteration]) - 1)
                sparse_x_data[iteration] = numpy.delete(sparse_x_data[iteration], index)
                sparse_noise_and_y_data[iteration] = numpy.delete(sparse_noise_and_y_data[iteration], index)

        estimated_alpha, estimated_x_0, estimated_w_t, estimated_w_t_alpha, estimated_v_t = self.accelerated_optimisation.monte_carlo_noise_characterised_minimise(
            self.step, self.ending_point, sparse_noise_and_y_data, sparse_x_data)

        plt.hist(estimated_alpha, density=True, bins=10)
        plt.title("Estimate Of Order Alpha In Population With Different Noise")
        plt.xlabel("Alpha")
        plt.savefig('Estimated alpha values - {identify}.png'.format(identify=int(time.time())), format='png', dpi=2400)
        plt.show()

        plt.hist(estimated_x_0, density=True, bins=10)
        plt.title("Estimate Of Initial Conditions In Population With Different Noise")
        plt.xlabel("Initial Conditions")
        plt.savefig('Estimated Initial Condition values - {identify}.png'.format(identify=int(time.time())),
                    format='png', dpi=2400)
        plt.show()

        plt.hist(estimated_w_t, density=True, bins=100)
        plt.title("Estimate Of Estimation Noise In Population With Different Noise")
        plt.xlabel("Noise Magnitude")
        plt.savefig('Estimated Of Estimation Noise Values - {identify}.png'.format(identify=int(time.time())),
                    format='png', dpi=2400)
        plt.show()

        plt.hist(estimated_w_t_alpha, density=True, bins=100)
        plt.title("Estimate Of Alpha Noise In Population With Different Noise")
        plt.xlabel("Alpha Variation")
        plt.savefig('Estimated Alpha Noise Values - {identify}.png'.format(identify=int(time.time())), format='png',
                    dpi=2400)
        plt.show()

        plt.hist(estimated_v_t, density=True, bins=100)
        plt.title("Estimate Of Measurement Noise In Population With Different Noise")
        plt.xlabel("Measurement Noise Magnitude")
        plt.savefig('Estimated Measurement Noise Values - {identify}.png'.format(identify=int(time.time())),
                    format='png', dpi=2400)
        plt.show()
