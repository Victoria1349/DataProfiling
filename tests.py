# Python 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import unittest
from DataProfiling import DataProfiling as dp



class maxValueTests(unittest.TestCase):

    def test_correct(self):
        DP = dp()

        ser = pd.Series([-200, 0, 24, np.nan, 150, 62, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        d = {"price": [1, 2, 0, 4, 5], "count": [0, 4, 0, 3, 1], "percent": [24, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        result = dp.maxValue(DP.ser)
        self.assertEqual(result, 150)


if __name__ == '__main__':
    unittest.main()
