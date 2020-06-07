# Python 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import unittest
from DataProfiling import DataProfiling



# class Cleaning

class findSkipsDFTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 0, 4, 5], "count": [0, 4, 0, 3, 1], "percent": [24, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findSkipsDF()
        expRes = pd.Series([])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSkips(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 0, np.nan, 5], "count": [0, 4, 0, 3, 1], "percent": [np.nan, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findSkipsDF()
        expRes = pd.Series([{'price': 3}, {'percent': 0}])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findSkipsDF()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findSkipsDF()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findSkipsDF()
        result = res.empty
        self.assertEqual(result, True)


class cleanSkipsDFTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 0, 4, 5], "count": [0, 4, 0, 3, 1], "percent": [24, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanSkipsDF()
        expRes = pd.DataFrame(d)
        if DataProfiling.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSkips(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 0, 5, np.nan], "count": [0, 4, 0, 1, np.nan], "percent": [8, 51, 0, 4, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanSkipsDF()
        d = {"price": [1.0, 2.0, 0.0, 5.0], "count": [0.0, 4.0, 0.0, 1.0], "percent": [8.0, 51.0, 0.0, 4.0]}
        expRes = pd.DataFrame(d)

        print()
        print(res)
        print(expRes)

        if DataProfiling.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanSkipsDF()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanSkipsDF()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanSkipsDF()
        result = res.empty
        self.assertEqual(result, True)


class findSkipsSerTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, 0, 150, 62, 24])
        DP.__setSeries__(ser)

        res = DP.findSkipsSer()
        expRes = pd.Series([])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSkips(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, np.nan, 24])
        DP.__setSeries__(ser)

        res = DP.findSkipsSer()
        expRes = list([3,5])
        print(res)
        print(expRes)
        if res == expRes:
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.findSkipsSer()
        result = len(res) == 0
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.findSkipsSer()
        result = len(res) == 0
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan, np.nan, np.nan, np.nan])
        DP.__setSeries__(ser)

        res = DP.findSkipsSer()
        result = len(res) == 0
        self.assertEqual(result, True)


class cleanSkipsSerTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, 0, 150, 62, 24])
        DP.__setSeries__(ser)

        res = DP.cleanSkipsSer()
        expRes = pd.Series([-200, 0, 2, 0, 150, 62, 24])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSkips(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, np.nan, 24])
        DP.__setSeries__(ser)

        res = DP.cleanSkipsSer()
        expRes = pd.Series([-200, 0, 2, 150, np.nan, 24])

        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.cleanSkipsSer()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.cleanSkipsSer()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan, np.nan, np.nan, np.nan])
        DP.__setSeries__(ser)

        res = DP.cleanSkipsSer()
        result = res.empty
        self.assertEqual(result, True)


class findEjectionsTests(unittest.TestCase):

    def test_noEj(self):
        DP = DataProfiling()
        ser = pd.Series([-20, 0, 20, np.nan, 15, 6, '42', -20, 12, 4, 10, 10, 0, 22])
        DP.__setSeries__(ser)

        res = DP.findEjections()
        expRes = pd.Series([],[])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someEj(self):
        DP = DataProfiling()
        ser = pd.Series([-20, 0, 20, np.nan, 15, 6, '42', -200, 12, 450, 10, 10, 0, 22])
        DP.__setSeries__(ser)

        res = DP.findEjections()
        expRes = pd.Series([-200, 450],[7, 9])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.findEjections()
        result = str(type(res)) == "<class 'NoneType'>"
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.findEjections()
        result = str(type(res)) == "<class 'NoneType'>"
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan, np.nan, np.nan, np.nan])
        DP.__setSeries__(ser)

        res = DP.findEjections()
        result = str(type(res)) == "<class 'NoneType'>"
        self.assertEqual(result, True)


class cleanEjectionsTests(unittest.TestCase):

    def test_noEj(self):
        DP = DataProfiling()
        ser = pd.Series([-20, 0, 20, 0, 15, 6, 42, -20, 12, 4, 10, 10, 0, 22])
        DP.__setSeries__(ser)

        res = DP.cleanEjections()
        expRes = pd.Series([-20, 0, 20, 0, 15, 6, 42, -20, 12, 4, 10, 10, 0, 22])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someEj(self):
        DP = DataProfiling()
        ser = pd.Series([-20, 0, 20, 0, 15, 6, 42, -200, 12, 4, 10, 10, 0, 22])
        DP.__setSeries__(ser)

        res = DP.cleanEjections()
        expRes = pd.Series([-20, 0, 20, 0, 15, 6, 42, 12, 4, 10, 10, 0, 22], [0,1,2,3,4,5,6,8,9,10,11,12,13])

        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.cleanEjections()
        result = str(type(res)) == "<class 'NoneType'>"
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.cleanEjections()
        result = str(type(res)) == "<class 'NoneType'>"
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan, np.nan, np.nan, np.nan])
        DP.__setSeries__(ser)

        res = DP.cleanEjections()
        result = str(type(res)) == "<class 'NoneType'>"
        self.assertEqual(result, True)






# class Statistics

class distributionFuncTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 62, 24])
        DP.__setSeries__(ser)

        res = DP.distributionFunc()
        expRes = pd.Series([1,1,1,1,1,1], [-200,0,2,42,62,150])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 24, np.nan, 150, 0, 24])
        DP.__setSeries__(ser)

        res = DP.distributionFunc()
        expRes = pd.Series([1, 2, 2, 1], [-200, 0, 24, 150])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([150, 150, 150, 150, 150, 150, 150])
        DP.__setSeries__(ser)

        res = DP.distributionFunc()
        expRes = pd.Series([7], [150])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([], [])
        DP.__setSeries__(ser)

        res = DP.distributionFunc()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.distributionFunc()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.distributionFunc()
        result = res.empty
        self.assertEqual(result, True)


class frequencyFuncTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, 0, 2, 2])
        DP.__setSeries__(ser)

        res = DP.frequencyFunc()
        expRes = pd.Series([1,2,3], [-200,0,2])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, -200, 0, 2, 2])
        DP.__setSeries__(ser)

        res = DP.frequencyFunc()
        expRes = pd.Series([2,2,2], [-200,0,2])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([150, 150, 150, 150])
        DP.__setSeries__(ser)

        res = DP.frequencyFunc()
        expRes = pd.Series([4], [150])
        if DataProfiling.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([], [])
        DP.__setSeries__(ser)

        res = DP.frequencyFunc()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.frequencyFunc()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.frequencyFunc()
        result = res.empty
        self.assertEqual(result, True)


class modaTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 62, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = list(DP.moda())
        self.assertEqual(result, [-200, 0, 2, 24, 62, 150])

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 24, np.nan, 150, 0, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = list(DP.moda())
        self.assertEqual(result, [0, 24])

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([150, 150, 150, 150, 150, 150, 150], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = list(DP.moda())
        self.assertEqual(result, [150])

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([], [])
        DP.__setSeries__(ser)

        result = list(DP.moda())
        self.assertEqual(result, [])

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        result = list(DP.moda())
        self.assertEqual(result, [])

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        result = list(DP.moda())
        self.assertEqual(result, [])


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


class minValueTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 62, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.minValue()
        self.assertEqual(result, -200)

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 24, np.nan, 150, 0, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.minValue()
        self.assertEqual(result, -200)

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, -200, -200, -200, -200, -200, -200], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.minValue()
        self.assertEqual(result, -200)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([], [])
        DP.__setSeries__(ser)

        res = DP.minValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.minValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.minValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)


class meanValueTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 62, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.meanValue()
        self.assertEqual(result, 6.333333333333333)

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 24, np.nan, 150, 0, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.meanValue()
        self.assertEqual(result, -0.3333333333333333)

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, -200, -200, -200, -200, -200, -200], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.meanValue()
        self.assertEqual(result, -200)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([], [])
        DP.__setSeries__(ser)

        res = DP.meanValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.meanValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.meanValue()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)


class medianTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 62, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.median()
        self.assertEqual(result, 13)

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 24, np.nan, 150, 0, 24], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.median()
        self.assertEqual(result, 12)

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, -200, -200, -200, -200, -200, -200], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
        DP.__setSeries__(ser)

        result = DP.median()
        self.assertEqual(result, -200)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([], [])
        DP.__setSeries__(ser)

        res = DP.median()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.median()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.median()
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
DP.__setDF__(df)
result = DP.maxValue()
self.assertEqual(result, 150)'''
