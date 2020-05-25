# Python 3
import pandas as pd
import numpy as np

class DataProfiling(object):
    data = pd.DataFrame()

    def __init__(self, dt):
        """Constructor"""
        data = dt


class Profiling(object):

    def dataType(col):
        print(col.__class__())
        return col.dtypes

    def funcType(func):
        str = "Hello world"
        return str

    def findMistakes(data):
        resData = pd.DataFrame()
        return resData

    def numbOfRepetitionsOfOneValueInColumn(data):
        numbs = [0,0,0]
        return numbs

    def dataStandardization(col):
        resCol = pd.Series()
        return resCol


class Cleaning(object):

    def findSkips(data):
        ids = [0, 0, 0]
        return ids

    def cleanSkips(data):
        resData = pd.DataFrame()
        return resData

    def findEjections(data):
        ids = [0, 0, 0]
        return ids

    def cleanEjections(data):
        resData = pd.DataFrame()
        return resData

    def findNulls(data):
        ids = [0,0,0]
        return ids

    def numbOfSkips(data):
        numb = 0
        return numb

    def findMissingData(data):
        ids = [0,0,0]
        return ids

    def numbOfSkipDataInColumn(col):
        numb = 0
        return numb

    def delDuplicates(data):
        resData = pd.DataFrame()
        return resData



class Statistics(object):

    def distributionFunc(data):
        str = "Hello world"
        return str

    def frequencyFunc(data):
        ids = [0.0, 0.0, 0.0]
        return ids

    def moda(col):
        moda = 0.0
        return moda

    def maxValue(col):
        maxValue = 0.0
        return maxValue

    def minValue(col):
        minValue = 0.0
        return minValue

    def meanValue(col):
        meanValue = 0.0
        return meanValue

    def median(col):
        median = 0.0
        return median



class Structures(object):

    def structureDetection(data):
        str = "Hello world"

    def relationsDetection(data):
        str = "Hello world"



class Vizual(object):

    def datasetVisualization(data):
        str = "Hello world"


class Report(object):

    def metadataReport(data):
        str = "Hello world"


# ------------------------------------------------------------------------------------------------

d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']), 'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
#df = pd.DataFrame(np.random.randn(3,3),index='A B C'.split(),columns='1 2 3'.split())
df = pd.DataFrame(d)
#print(df)
#print(df.dtypes)
#print("df.class = ", df.__class__())

ser = pd.Series([10, 20, 30], ['a', 'b', 'c'])
#print(ser.dtype)
#print("ser.class = ", ser.__class__())


print("--")
print(Profiling.dataType(ser))