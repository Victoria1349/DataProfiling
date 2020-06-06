# Python 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import unittest
from DataProfiling import DataProfiling



class maxValueTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 62, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.maxValue()
        self.assertEqual(result, 150)

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 24, np.nan, 150, 0, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.maxValue()
        self.assertEqual(result, 150)

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([150, 150, 150, 150, 150, 150, 150], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.maxValue()
        self.assertEqual(result, 150)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([], [])
        DP.__setSeries__(ser)

        res = DP.maxValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.maxValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.maxValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)


if __name__ == '__main__':
    unittest.main()


''' example
DP = DataProfiling(
ser = pd.Series([-200, 0, 24, np.nan, 150, 62, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
DP.__setSeries__(ser
d = {"price": [1, 2, 0, 4, 5], "count": [0, 4, 0, 3, 1], "percent": [24, 51, 0, 0, 4]}
df = pd.DataFrame(d)
DP.__setDF__(df
result = DP.maxValue()
self.assertEqual(result, 150)'''
