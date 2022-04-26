import unittest
from CJA import CJA


class TestCJA(unittest.TestCase):

    alpha_is_one_solutions = [1, -1]
    alpha_is_two_solutions = [1, -2, 1]
    alpha_is_three_solutions = [1, -3, 3, -1]
    solutions = [alpha_is_one_solutions, alpha_is_two_solutions, alpha_is_three_solutions]

    def test_CJA_init(self):
        for alpha in [1, 2, 3]:
            cja_generator = CJA(alpha)
            cja = iter(cja_generator)
            for j in range(alpha+1):
                value_returned = next(cja)
                self.assertEqual(value_returned, self.solutions[alpha-1][j],
                                 "CJA output is incorrect. Output is {output} instead of correct solution {solution}"
                                 .format(output=value_returned, solution=self.solutions[alpha-1][j]))

    def test_instant_coefficient_calculation(self):
        cja_generator = CJA(alpha=0)

        for alpha in [1, 2, 3]:
            for j in range(alpha+1):
                value_returned = cja_generator.instant_coefficient_calculation(alpha, j)
                self.assertEqual(value_returned, self.solutions[alpha-1][j],
                                 "CJA output is incorrect. Output is {output} instead of correct solution {solution}"
                                 .format(output=value_returned, solution=self.solutions[alpha-1][j]))