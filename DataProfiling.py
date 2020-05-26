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

    def isNull(cnt):
        return cnt == 0

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
        resData = data

        for col in data:
            resData = resData.dropna(axis='index', how='any', subset=[col])

        return resData

    def findEjections(data):
        ids = [0, 0, 0]
        return ids

    def cleanEjections(data):
        resData = pd.DataFrame()
        return resData

    def findNulls(data):
        rez = pd.Series()
        id = 0
        str = 0

        for col in data:
            for el in data[col]:
                if Cleaning.isNull(el):
                    d = {col: str}
                    rez[id.__str__()] = d
                str = str + 1
                if str > 3:
                    str = str - 4
            id = id + 1

        return rez

    def cntOfSkips(data):
        df = Cleaning.findSkips(data)
        numb = len(df)
        return numb

    def findMissingData(data):
        ids = [0,0,0]
        return ids

    def cntOfSkipDataInColumn(col):
        cnt = 0
        for el in col:
            if el != el:
                cnt = cnt + 1
        return cnt

    def delDuplicates(data):
        tmp = pd.Series()
        indexes = []

        for i in range(len(data)):
            tmp[i.__str__()] = data.loc[i]
            for j in range(i):
                if  pd.Series.equals(tmp[i.__str__()], tmp[j.__str__()]):
                    indexes.append(i)

        df = data.drop(data.index[indexes])
        return df



class Statistics(object):

    def distributionFunc(data):
        str = "Hello world"
        return str

    def frequencyFunc(data):
        ids = [0.0, 0.0, 0.0]
        return ids

    def moda(col):
        return col.mode()

    def maxValue(col):
        return col.max()

    def minValue(col):
        return col.min()

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

data = 'price,count,percent\n1,10,\n1,10,\n0,20,51'
df = pd.read_csv(StringIO(data))
df.loc[3] = {'price': 4, 'count': None, 'percent': 26.3}
df.loc[4] = {'price': 4, 'count': None, 'percent': 26.3}
#print(df)

ser = pd.Series([np.nan, 20, 10, np.nan, 40, 10], ['a', 'b', 'c', 'a', 'b', 'c'])
print(ser)


print("--")
print(Statistics.minValue(ser))