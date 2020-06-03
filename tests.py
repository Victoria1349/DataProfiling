# Python 3
import pandas as pd
import numpy as np
from io import StringIO
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

        data = 'price,count,percent\n1,10,\n1,10,\n3,20,51\n4,,26.3\n4,50,26.3'
        df = pd.read_csv(StringIO(data))
        DP.__setDF__(df)

        result = dp.maxValue(DP.ser)
        self.assertEqual(result, 150)


if __name__ == '__main__':
    unittest.main()
