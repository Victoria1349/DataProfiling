# Python 3
import pandas as pd
import numpy as np
import unittest
from DataProfiling import DataProfiling
from DataProfiling import PairsInRelations
from DataProfiling import SecondaryFuncs


# class Profiling

class dataTypeTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, '150', 62, '24'])
        DP.__setSeries__(ser)

        result = str(DP.dataType())
        self.assertEqual(result, 'object')

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 12, 0, -200, -5, -12, 3])
        DP.__setSeries__(ser)

        result = DP.dataType()
        self.assertEqual(result, 'int64')

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.dataType()
        print()
        print(res)
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.dataType()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.dataType()
        if str(res) == str(np.nan): # hack of checking empty data
            result = True
        else:
            result = False
        self.assertEqual(result, True)


class findMistakesTests(unittest.TestCase):

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 12, 0, -200, -5, -12, 3])
        DP.__setSeries__(ser)

        res = DP.findMistakes()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someMist(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, '150', 62, '24'])
        DP.__setSeries__(ser)

        res = DP.findMistakes()
        expRes = pd.Series(['150', '24'], [4, 6])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_twoDiff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, 'np.nan', '150', '24'])
        DP.__setSeries__(ser)

        res = DP.findMistakes()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.findMistakes()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.findMistakes()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.findMistakes()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)


class cntOfOneValueInColumnTests(unittest.TestCase):

    def test_allDiff(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 3, 4, 5], "count": [6, 7, 8, 9, 10], "percent": [11, 12, 13, 14, 15]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cntOfOneValueInColumn()
        expRes = pd.Series([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someEq(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 3, 3, 5], "count": [6, 6, 8, 9, 10], "percent": [10, 12, 13, 13, 15]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cntOfOneValueInColumn()
        expRes = pd.Series([2,2,2,2,1,1,1,1,1,1,1], [13, 10, 6, 3, 1, 2, 5, 8, 9, 12, 15])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_allEq(self):
        DP = DataProfiling()
        d = {"price": [2, 2, 2, 2, 2], "count": [2, 2, 2, 2, 2], "percent": [2, 2, 2, 2, 2]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cntOfOneValueInColumn()
        expRes = pd.Series([15], [2])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cntOfOneValueInColumn()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cntOfOneValueInColumn()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cntOfOneValueInColumn()
        result = res.empty
        self.assertEqual(result, True)


class dataStandardizationTests(unittest.TestCase):

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([5,5,5,5,5])
        DP.__setSeries__(ser)

        res = DP.dataStandardization()
        expRes = pd.Series([np.nan,np.nan,np.nan,np.nan,np.nan])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_allDiff(self):
        DP = DataProfiling()
        ser = pd.Series([-10, 0, 10, 5])
        DP.__setSeries__(ser)

        res = DP.dataStandardization()
        expRes = pd.Series([-1.52, -0.17, 1.18, 0.51])

        print()
        print()
        print(res)
        print(expRes)
        print(SecondaryFuncs.isEqSer(res, expRes))

        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-10, 0, 10, 0])
        DP.__setSeries__(ser)

        res = DP.dataStandardization()
        expRes = pd.Series([-1.41, 0.0, 1.41, 0.0])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.dataStandardization()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.dataStandardization()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.dataStandardization()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)


class dataNormalizationTests(unittest.TestCase):

    def test_allSame(self):
        DP = DataProfiling()
        ser = pd.Series([5,5,5,5,5])
        DP.__setSeries__(ser)

        res = DP.dataNormalization()
        expRes = pd.Series([np.nan,np.nan,np.nan,np.nan,np.nan])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_allDiff(self):
        DP = DataProfiling()
        ser = pd.Series([-10, 0, 10, 5])
        DP.__setSeries__(ser)

        res = DP.dataNormalization()
        expRes = pd.Series([0, 0.5, 1, 0.75])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSame(self):
        DP = DataProfiling()
        ser = pd.Series([-10, 0, 10, 0])
        DP.__setSeries__(ser)

        res = DP.dataNormalization()
        expRes = pd.Series([0, 0.5, 1, 0.5])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.dataNormalization()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.dataNormalization()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan,np.nan,np.nan,np.nan])
        DP.__setSeries__(ser)

        res = DP.dataNormalization()
        expRes = pd.Series()
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)


# class Cleaning

class findSkipsDFTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 0, 4, 5], "count": [0, 4, 0, 3, 1], "percent": [24, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findSkipsDF()
        expRes = pd.Series([])
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqDF(res, expRes):
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

        if SecondaryFuncs.isEqDF(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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

        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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

        if SecondaryFuncs.isEqSer(res, expRes):
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


class findNullsDFTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 10, 4, 5], "count": [10, 4, 10, 3, 1], "percent": [24, 51, 10, 10, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findNullsDF()
        expRes = pd.Series([])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someNulls(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 0, np.nan, 5], "count": [0, 4, 0, 3, 1], "percent": [np.nan, 51, 0, 10, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findNullsDF()
        expRes = pd.Series([{'price': 3}, {'count': 0}, {'count': 2}, {'percent': 2}])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findNullsDF()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findNullsDF()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.findNullsDF()
        result = res.empty
        self.assertEqual(result, True)


class cleanNullsDFTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 10, 4, 5], "count": [10, 4, 10, 3, 1], "percent": [24, 51, 10, 10, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanNullsDF()
        expRes = pd.DataFrame(d)
        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someNulls(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 10, 5, 0], "count": [0, 4, 0, 1, 0], "percent": [8, 51, 0, 4, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanNullsDF()
        d = {"price": [1, 2, 10, 5], "count": [0, 4, 0, 1], "percent": [8, 51, 0, 4]}
        expRes = pd.DataFrame(d)

        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanNullsDF()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanNullsDF()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.cleanNullsDF()
        result = res.empty
        self.assertEqual(result, True)


class findNullsSerTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 10, 2, 10, 150, 62, 24])
        DP.__setSeries__(ser)

        res = DP.findNullsSer()
        expRes = pd.Series([])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someNulls(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 0, 24])
        DP.__setSeries__(ser)

        res = DP.findNullsSer()
        expRes = list([1,5])
        if res == expRes:
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.findNullsSer()
        result = len(res) == 0
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.findNullsSer()
        result = len(res) == 0
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan, np.nan, np.nan, np.nan])
        DP.__setSeries__(ser)

        res = DP.findNullsSer()
        result = len(res) == 0
        self.assertEqual(result, True)


class cleanNullsSerTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 10, 2, 10, 150, 62, 24])
        DP.__setSeries__(ser)

        res = DP.cleanNullsSer()
        expRes = pd.Series([-200, 10, 2, 10, 150, 62, 24])
        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someNulls(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 0, 24])
        DP.__setSeries__(ser)

        res = DP.cleanNullsSer()
        expRes = pd.Series([-200, 2, np.nan, 150, 24], [0,2,3,4,6])

        if SecondaryFuncs.isEqSer(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        res = DP.cleanNullsSer()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        res = DP.cleanNullsSer()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan, np.nan, np.nan, np.nan])
        DP.__setSeries__(ser)

        res = DP.cleanNullsSer()
        result = res.empty
        self.assertEqual(result, True)


class cntOfSkipDataInDFTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 0, 4, 5], "count": [0, 4, 0, 3, 1], "percent": [24, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        result = DP.cntOfSkipDataInDF()
        expRes = 0
        self.assertEqual(result, expRes)

    def test_someSkips(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 0, np.nan, 5], "count": [0, 4, 0, 3, 1], "percent": [np.nan, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        result = DP.cntOfSkipDataInDF()
        expRes = 2
        self.assertEqual(result, expRes)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        result = DP.cntOfSkipDataInDF()
        self.assertEqual(result, 0)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        result = DP.cntOfSkipDataInDF()
        self.assertEqual(result, 0)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        result = DP.cntOfSkipDataInDF()
        self.assertEqual(result, 0)


class cntOfSkipDataInColumnTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, 0, 150, 62, 24])
        DP.__setSeries__(ser)

        result = DP.cntOfSkipDataInColumn()
        self.assertEqual(result, 0)

    def test_someSkips(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, np.nan, 24])
        DP.__setSeries__(ser)

        result = DP.cntOfSkipDataInColumn()
        self.assertEqual(result, 2)

    def test_empty(self):
        DP = DataProfiling()
        ser = pd.Series([])
        DP.__setSeries__(ser)

        result = DP.cntOfSkipDataInColumn()
        self.assertEqual(result, 0)

    def test_nulls(self):
        DP = DataProfiling()
        ser = pd.Series([0,0,0,0])
        DP.__setSeries__(ser)

        result = DP.cntOfSkipDataInColumn()
        self.assertEqual(result, 0)

    def test_nans(self):
        DP = DataProfiling()
        ser = pd.Series([np.nan, np.nan, np.nan, np.nan])
        DP.__setSeries__(ser)

        result = DP.cntOfSkipDataInColumn()
        self.assertEqual(result, 0)


class fillMissingDataTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 10, 4, 5], "count": [10, 4, 10, 3, 1], "percent": [24, 51, 10, 10, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.fillMissingData()
        expRes = pd.DataFrame(d)
        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someNulls(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 10, 5, np.nan], "count": [0, 4, 0, 1, np.nan], "percent": [np.nan, 20, 0, 4, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.fillMissingData()
        d = {"price": [1.0, 2.0, 10.0, 5.0, 4.5], "count": [0.0, 4.0, 0.0, 1.0, 1.25], "percent": [8.0, 20.0, 0.0, 4.0, 8.0]}
        expRes = pd.DataFrame(d)

        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.fillMissingData()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.fillMissingData()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.fillMissingData()
        result = res.empty
        self.assertEqual(result, True)


class replacementMissingsTests(unittest.TestCase):

    def test_full(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 10, 4, 5], "count": [10, 4, 10, 3, 1], "percent": [24, 51, 10, 10, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.replacementMissings(1)
        expRes = pd.DataFrame(d)
        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someNulls(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 10, 5, np.nan], "count": [0, 4, 0, 1, np.nan], "percent": [np.nan, 20, 0, 4, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.replacementMissings(1)
        d = {"price": [1.0, 2.0, 10.0, 5.0, 1.0], "count": [0.0, 4.0, 0.0, 1.0, 1.0], "percent": [1.0, 20.0, 0.0, 4.0, 1.0]}
        expRes = pd.DataFrame(d)

        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.replacementMissings(1)
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.replacementMissings(1)
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.replacementMissings(1)
        result = res.empty
        self.assertEqual(result, True)


class delDuplicatesTests(unittest.TestCase):

    def test_allDiff(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 10, 4, 5], "count": [10, 4, 10, 3, 1], "percent": [24, 51, 10, 10, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.delDuplicates()
        expRes = pd.DataFrame(d)
        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_someSame(self):
        DP = DataProfiling()
        d = {"price":[1, 2, 0, 4, 1], "count": [0, np.nan, 0, 3, 0], "percent": [24, 51, 0, 0, 24]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.delDuplicates()
        d = {"price":[1, 2, 0, 4], "count": [0, np.nan, 0, 3], "percent": [24, 51, 0, 0]}
        expRes = pd.DataFrame(d)

        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_allSame(self):
        DP = DataProfiling()
        d = {"price":[1, 1, 1, 1, 1], "count": [0, 0, 0, 0, 0], "percent": [24, 24, 24, 24, 24]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.delDuplicates()
        d = {"price": [1], "count": [0], "percent": [24]}
        expRes = pd.DataFrame(d)

        if SecondaryFuncs.isEqDF(res, expRes):
            result = True
        else:
            result = False
        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.delDuplicates()
        result = res.empty
        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.delDuplicates()
        result = res.empty
        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.delDuplicates()
        result = res.empty
        self.assertEqual(result, True)





# class Statistics

class distributionFuncTests(unittest.TestCase):

    def test_diff(self):
        DP = DataProfiling()
        ser = pd.Series([-200, 0, 2, np.nan, 150, 62, 24])
        DP.__setSeries__(ser)

        res = DP.distributionFunc()
        expRes = pd.Series([1,1,1,1,1,1], [-200,0,2,42,62,150])
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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
        if SecondaryFuncs.isEqSer(res, expRes):
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


# class Structures

class relationsDetectionTests(unittest.TestCase):

    def test_noRels(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 4, 4, 5], "count": [1, 4, 3, 3, 1], "percent": [24, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.relationsDetection()
        expRes = list()

        result = True
        for i in range(len(res)):
            if not PairsInRelations.isEq(res[i], expRes[i]):
                result = False

        self.assertEqual(result, True)

    def test_someRels(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 3, 4, 5], "count": [1, 4, 3, 3, 1], "percent": [24, 51, 0, 0, 4]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.relationsDetection()
        tmp = PairsInRelations()
        tmp.setter('price', 'count')
        expRes = list()
        expRes.append(tmp)

        result = True
        for i in range(len(res)):
            if not PairsInRelations.isEq(res[i], expRes[i]):
                result = False

        self.assertEqual(result, True)

    def test_allRels(self):
        DP = DataProfiling()
        d = {"price": [1, 2, 3, 4, 5], "count": [1, 4, 3, 3, 1], "percent": [3, 4, 5, 1, 2]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.relationsDetection()
        tmp = PairsInRelations()
        tmp.setter('percent', 'price')
        expRes = list()
        expRes.append(tmp)
        tmp1 = PairsInRelations()
        tmp1.setter('price', 'count')
        expRes.append(tmp1)
        tmp2 = PairsInRelations()
        tmp2.setter('percent', 'count')
        expRes.append(tmp2)
        tmp3 = PairsInRelations()
        tmp3.setter('price', 'percent')
        expRes.append(tmp3)

        result = True
        for i in range(len(res)):
            if not PairsInRelations.isEq(res[i], expRes[i]):
                result = False

        self.assertEqual(result, True)

    def test_empty(self):
        DP = DataProfiling()
        d = {}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.relationsDetection()
        expRes = list()

        result = True
        for i in range(len(res)):
            if not PairsInRelations.isEq(res[i], expRes[i]):
                result = False

        self.assertEqual(result, True)

    def test_nulls(self):
        DP = DataProfiling()
        d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.relationsDetection()
        expRes = list()

        result = True
        for i in range(len(res)):
            if not PairsInRelations.isEq(res[i], expRes[i]):
                result = False

        self.assertEqual(result, True)

    def test_nans(self):
        DP = DataProfiling()
        d = {"price": [np.nan, np.nan, np.nan], "count": [np.nan, np.nan, np.nan], "percent": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(d)
        DP.__setDF__(df)

        res = DP.relationsDetection()
        expRes = list()

        result = True
        for i in range(len(res)):
            if not PairsInRelations.isEq(res[i], expRes[i]):
                result = False

        self.assertEqual(result, True)


if __name__ == '__main__':
    unittest.main()
