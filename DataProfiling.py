# Python 3
import pandas as pd
import numpy as np
#import xray
from io import StringIO

class DataProfiling(object):
    data = pd.DataFrame()

    def __init__(self, dt):
        """Constructor"""
        data = dt


class Profiling(object):

    def dataType(col):
        #print(col.__class__())
        return col.dtypes

    def funcType(func):
        str = "Hello world"
        return str

    def findMistakes(data):
        resData = pd.DataFrame()
        return resData

    def numbOfRepetitionsOfOneValueInColumn(data):
        return data.stack().value_counts()

    def dataStandardization(col):
        resCol = pd.Series()
        return resCol


class Cleaning(object):

    def findSkips(data):
        rez = pd.Series()
        id = 0

        for col in data:
            index = df[col].index[df[col].apply(np.isnan)]
            df_index = df.index.values.tolist()
            for i in index:
                ind = df_index.index(i)
                # добавить данные в Series
                d = {col: ind}
                rez[id.__str__()] = d
                id = id + 1

        return rez

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

#d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': pd.Series([1., 2., 4., 4.], index=['a', 'b', 'c', 'd'])}
#df = pd.DataFrame(np.random.randn(3,3),index='A B C'.split(),columns='1 2 3'.split())
#df = pd.DataFrame(d)

data = 'price,count,percent\n1,10,\n2,20,51\n3,30,'
df = pd.read_csv(StringIO(data))
df.loc[3] = {'price': 4, 'count': None, 'percent': 26.3}
print(df)

ser = pd.Series([10, 20, 30, 20, 40, 10], ['a', 'b', 'c', 'a', 'b', 'c'])
#print(ser)


print("--")
print(Cleaning.findSkips(df))