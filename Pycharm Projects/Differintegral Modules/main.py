import numpy
import os
import random
import math
import time
import matplotlib.pyplot as plt
from GLMethod import GLMethod
from tqdm import tqdm
from GLMethodAccelerated import GLMethodAccelerated
from ModelOptimiser import ModelOptimiser

def signaltonoise(a, axis=0, ddof=0):
    a = numpy.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return numpy.where(sd == 0, 0, m/sd)

if __name__ == '__main__':

    plt.plot(alphas)

    plt.hist(alphas, density=True, bins=10)
    plt.savefig('Estimated alpha values - {identify}.png'.format(identify=int(time.time())), format='png', dpi=2400)
    plt.show()
    plt.hist(alphas, density=True, bins=10)
    plt.savefig('Estimated Initial Condition values - {identify}.png'.format(identify=int(time.time())), format='png', dpi=2400)
    plt.show()

    print(alphas)
