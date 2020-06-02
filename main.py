# Python 3
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import math
import unittest
import DataProfiling



class maxValueTests(unittest.TestCase):
    #col = pd.Series()

    def test_correct(self):
        ser = pd.Series([-200, 0, 24, np.nan, 150, 62, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])

        #DP = DataProfiling()
        #DP.__setSeries__(ser)

        result = DataProfiling.maxValue(ser)
        self.assertEqual(result, 150)

if __name__ == '__main__':
    unittest.main()