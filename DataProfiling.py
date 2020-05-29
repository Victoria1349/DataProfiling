# Python 3
import pandas as pd
import numpy as np
#import xray
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import math

class DataProfiling(object):
    data = pd.DataFrame()

    def __init__(self, dt):
        """Constructor"""
        data = dt


class Profiling(object):

    def dataType(col):
        return col.dtypes

    def funcType(func):                 # !
        str = "Hello world"
        return str

    def findMistakes(data):                 # !
        resData = pd.DataFrame()
        return resData

    def numbOfRepetitionsOfOneValueInColumn(data):
        return data.stack().value_counts()

    def dataStandardization(col):                 # ?
        resCol = Cleaning.cleanSkipsSer(col)

        mean = Statistics.meanValue(resCol)
        sumX = 0
        for i in range (len(resCol)):
            sumX = sumX + ((resCol[i] - mean) * (resCol[i] - mean))
        standard_deviation = np.math.sqrt(sumX / len(resCol))

        for i in range (len(resCol)):
            resCol[i] = (resCol[i] - mean) / standard_deviation

        return resCol

    def dataNormalization(col):
        resCol = col
        min = Statistics.minValue(resCol)
        max = Statistics.maxValue(resCol)
        i = 0

        for el in col:
            if el == el:
                tmp = (el - min) / (max - min)
                resCol[resCol.index[i]] = tmp
            i = i + 1

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

    def cleanSkipsDF(data):
        resData = data
        for col in data:
            resData = resData.dropna(axis='index', how='any', subset=[col])

        return resData

    def cleanSkipsSer(col):
        resCol = col
        ind1 = 0
        ind2 = 0
        for el in col:
            if el != el:
                resCol = resCol.drop(labels=[col.index[ind1]])
                ind2 = ind2 + 1
            ind1 = ind1 + 1

        return resCol

    def findEjections(col):
        ej = pd.Series()
        col2 = col.sort_values(ignore_index=True)

        n = len(col2)
        indQ1 = (n+1)/4
        indQ3 = (n+1)*3/4
        #indIqr = indQ3 - indQ1

        if(indQ1 % 1 != 0):
            #print(indQ1)
            ind1 = math.floor(indQ1)
            ind2 = math.ceil(indQ1)
            #print(ind1-1, ind2-1)
            #print()
            q1 = (col2[ind1-1] + col2[ind2-1]) / 2

        else:
            q1 = col2[indQ1 - 1]

        if (indQ3 % 1 != 0):
            #print(indQ3)
            ind1 = math.floor(indQ3)
            ind2 = math.ceil(indQ3)
            #print(ind1-1, ind2-1)
            #print()
            q3 = (col2[ind1-1] + col2[ind2-1]) / 2

        else:
            q3 = col2[indQ3 - 1]

        #iqr = col2[indIqr - 1]
        iqr = q3 - q1
        #print(q1, q3, iqr)

        left = q1 - iqr*1.5
        right = q3 + iqr*1.5
        #print(left, right)
        #print()

        for i in range (n):
            if(col[i] < left or col[i] > right):
                ej[i.__str__()] = col[i]

        return ej

    def cleanEjections(col):                 # !
        delEl = Cleaning.findEjections(col)
        resCol = col

        for i in range (len(delEl)):
            resCol = resCol.drop(labels=[int(delEl.index[i])])

        return resCol

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

    def findMissingData(data):                 # !
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

    def replacementMissings(data, cnt):
        return data.fillna(cnt)



class Statistics(object):

    def sumSer(col):
        sum = 0
        resCol = Cleaning.cleanSkipsSer(col)

        for i in range (len(resCol)):
            sum = sum + resCol[i]

        return sum

    def distributionFunc(col, cnt):
        return col.groupby(col).size().nlargest(cnt)

    def frequencyFunc(data):                 # ?
        #ids = [0.0, 0.0, 0.0]
        plt.hist(data['percent'])
        plt.show()
        return sns.kdeplot(data['percent'])
        #return ids

    def moda(col):
        return col.mode()

    def maxValue(col):
        return col.max()

    def minValue(col):
        return col.min()

    def meanValue(col):
        return col.mean()

    def median(col):
        return col.median()



class Structures(object):

    def structureDetection(data):                 # !
        str = "Hello world"

    def relationsDetection(data):                 # !
        str = "Hello world"



class Vizual(object):

    def datasetVisualization(data):                 # !!!!!
        #str = "Hello world"
        pd.plotting.scatter_matrix(data, alpha = 0.7, figsize = (14,8))
        plt.show()


class Report(object):

    def metadataReport(data):                 # !!!!!
        str = "Hello world"


# ------------------------------------------------------------------------------------------------

data = 'price,count,percent\n1,10,\n1,10,\n3,20,51'
df = pd.read_csv(StringIO(data))
df.loc[3] = {'price': 4, 'count': None, 'percent': 26.3}
df.loc[4] = {'price': 4, 'count': 50, 'percent': 26.3}
#print(df)

#ser = pd.Series([np.nan, 20, 10, 'np.nan', 40, 10], ['a', 'b', 'c', 'd', 'e', 'f'])
ser = pd.Series([22, 24, -60, 32, -200, 34, 200, 34, 24, 43, 44, 43, 57, 88, 150, 62, 67, 81])
#ser = pd.Series([7,8,9,10,10,10,11,12,13,14])
print(ser)


print("--")
print(Cleaning.cleanEjections(ser))