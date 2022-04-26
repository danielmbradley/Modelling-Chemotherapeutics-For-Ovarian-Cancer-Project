
class CJA:

    def __init__(self, alpha):
        self.alpha = alpha

    def __iter__(self):
        self.binomial_coefficient = 1
        self.i = 0
        return self

    def __next__(self):
        x = self.i
        self.i += 1
        if x == 0:
            return ((-1) ** x) * self.binomial_coefficient
        else:
            self.binomial_coefficient = self.binomial_coefficient * ((self.alpha - (x-1))/((x-1) + 1))
            return ((-1) ** x) * self.binomial_coefficient

    def instant_coefficient_calculation(self, alpha, j):
        binomial_coefficient = 1
        for i in range(j):
            binomial_coefficient = binomial_coefficient * ((alpha - i) / (i + 1))
        return ((-1) ** j) * binomial_coefficient