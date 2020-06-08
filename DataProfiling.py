# Python 3
import pandas as pd
import numpy as np
#from io import StringIO
import matplotlib.pyplot as plt
#import seaborn as sns
import math
from pandas import MultiIndex


class DataProfiling(object):
    data = pd.DataFrame()
    col = pd.Series()

    def __init__(self):
        """Constructor"""

    def __setDF__(self, dt):
        """setter"""

        # size of matrix
        cnt = dt.shape[0] * dt.shape[1]
        if cnt == 0:
            print("Data is empty!")
            return

        # count of nulls
        cntNulls = len(Cleaning.findNullsDF(dt))
        if cnt == cntNulls:
            print("All data is nulls!")
            return

        # count of nans
        cntNans = len(Cleaning.findSkipsDF(dt))
        if cnt == cntNans:
            print("All data is nans!")
            return

        self.data = dt
        return self.data


    def __setSeries__(self, col):
        """setter"""

        # size of array
        cnt = len(col)
        if cnt == 0:
            print("Column is empty!")
            return

        # count of nulls
        cntNulls = len(Cleaning.findNullsSer(col))
        if cnt == cntNulls:
            print("All column is nulls!")
            return

        # count of nans
        cntNans = Cleaning.cntOfSkipDataInColumn(col)
        if cnt == cntNans:
            print("All column is nans!")
            return

        self.col = col
        return self.col



    def dataType(self):
        return Profiling.dataType(self.col)

    def findMistakes(self):
        resCol = self.cleanSkipsSer()
        return Profiling.findMistakes(resCol)

    def cntOfOneValueInColumn(self):
        if len(self.data) == 0:
            return pd.DataFrame()
        return Profiling.cntOfOneValueInColumn(self.data)

    def dataStandardization(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        if len(resCol) == 0:
            return pd.Series()
        return Profiling.dataStandardization(resCol)

    def dataNormalization(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        if len(resCol) == 0:
            return pd.Series()
        return Profiling.dataNormalization(resCol)



    def findSkipsDF(self):
        return Cleaning.findSkipsDF(self.data)

    def findSkipsSer(self):
        return Cleaning.findSkipsSer(self.col)

    def cleanSkipsDF(self):
        return Cleaning.cleanSkipsDF(self.data)

    def cleanSkipsSer(self):
        return Cleaning.cleanSkipsSer(self.col)

    def findEjections(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        if len(resCol) == 0:
            return
        return Cleaning.findEjections(resCol)

    def cleanEjections(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        if len(resCol) == 0:
            return
        return Cleaning.cleanEjections(resCol)

    def findNullsDF(self):
        return Cleaning.findNullsDF(self.data)

    def findNullsSer(self):
        return Cleaning.findNullsSer(self.col)

    def cleanNullsDF(self):
        return Cleaning.cleanNullsDF(self.data)

    def cleanNullsSer(self):
        return Cleaning.cleanNullsSer(self.col)

    def cntOfSkipDataInDF(self):
        return Cleaning.cntOfSkipDataInDF(self.data)

    def cntOfSkipDataInColumn(self):
        return Cleaning.cntOfSkipDataInColumn(self.col)

    def fillMissingData(self):
        return Cleaning.fillMissingData(self.data)

    def replacementMissings(self, cnt):
        return Cleaning.replacementMissings(self.data, cnt)

    def delDuplicates(self):
        return Cleaning.delDuplicates(self.data)



    def distributionFunc(self):
        resCol = self.col
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        return Statistics.distributionFunc(resCol)

    def frequencyFunc(self):
        delEls = Profiling.findMistakes(self.col)
        col2 = SecondaryFuncs.cleanElsFromSer(delEls, self.col)
        return Statistics.frequencyFunc(col2)

    def moda(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        return Statistics.moda(resCol)

    def maxValue(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        return Statistics.maxValue(resCol)

    def minValue(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        return Statistics.minValue(resCol)

    def meanValue(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        return Statistics.meanValue(resCol)

    def median(self):
        resCol = self.cleanSkipsSer()
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        return Statistics.median(resCol)



    def relationsDetection(self):
        return Structures.relationsDetection(self.data)



    def datasetVisualizationDF(self):
        Vizual.datasetVisualizationDF(self.data)

    def datasetVisualizationSer(self):
        resCol = self.col
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)
        Vizual.datasetVisualizationSer(resCol)



    def metadataReportDF(self, filename):
        Report.metadataReportDF(self.data, filename)

    def metadataReportSer(self, filename):
        Report.metadataReportSer(self.col, filename)


# -----------------------------------------------------------------------------------------------


class Profiling(object):

    def dataType(col):
        if len(col) == 0:
            return np.nan
        return col.dtypes

    def findMistakes(col):
        resCol = pd.Series()
        type = Profiling.dataType(col)

        if type == object:
            types = pd.Series()

            # counts of each type in column:
            for el in col:
                tmpType = SecondaryFuncs.elType(el)

                if len(types) == 0:
                    types[str(tmpType)] = 1

                elif SecondaryFuncs.isIndInCol(types, tmpType) == True:
                    types[str(tmpType)] = types[str(tmpType)] + 1

                else:
                    types[str(tmpType)] = 1

            max = Statistics.maxValue(types)

            # find index of our max type and max type itself
            indMax = 0
            cntMax = 0
            for i in range (len(types)):
                if types.values[i] == max:
                    indMax = i
                    cntMax = cntMax + 1

            if cntMax > 1:
                print("There are", cntMax, "peer values")
                return resCol

            maxType = types.index[indMax]

            # find indexes of mistakes values
            i = 0
            for el in col:
                tmpType = SecondaryFuncs.elType(el)

                if str(tmpType) != str(maxType):    # if that element not good
                    resCol[str(col.index[i])] = el
                i = i + 1

        return resCol

    def cntOfOneValueInColumn(data):
        return data.stack().value_counts()

    def dataStandardization(col):
        resCol = col.astype('float')
        mean = Statistics.meanValue(resCol)
        sumX = 0
        for i in range (len(resCol)):
            if SecondaryFuncs.isElInCol(list(resCol.index), i):
                sumX = sumX + ((resCol[i] - mean) * (resCol[i] - mean))
        standard_deviation = np.math.sqrt(sumX / len(resCol))

        for i in range (len(resCol)):
            if SecondaryFuncs.isElInCol(list(resCol.index), i):
                resCol[i] = round((resCol[i] - mean) / standard_deviation, 2)

        return resCol

    def dataNormalization(col):
        resCol = col.astype('float')
        min = Statistics.minValue(resCol)
        max = Statistics.maxValue(resCol)
        i = 0

        for el in resCol:
            tmp = (el - min) / (max - min)
            resCol[resCol.index[i]] = round(tmp, 2)
            i = i + 1

        return resCol


class Cleaning(object):

    def findSkipsDF(data):
        rez = pd.Series()
        id = 0

        for col in data:
            index = data[col].index[data[col].apply(np.isnan)]
            df_index = data.index.values.tolist()
            for i in index:
                ind = df_index.index(i)
                # добавить данные в Series
                d = {col: ind}
                rez[id.__str__()] = d
                id = id + 1

        return rez

    def findSkipsSer(col):
        ids = []
        cnt = len(col)

        for i in range(cnt):
            if col[i] != col[i]:
                ids.append(i)

        return ids

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
        #print(q1, q3, iqr, "!!!")

        left = q1 - iqr*1.5
        right = q3 + iqr*1.5
        #print(left, right)
        #print()

        for i in range (n):
            if(col2[i] < left or col2[i] > right):
                ej[col[col == col2[i]].index[0].__str__()] = col2[i]

        return ej

    def cleanEjections(col):
        delEl = Cleaning.findEjections(col)
        resCol = col

        resCol = SecondaryFuncs.cleanElsFromSer(delEl, resCol)

        '''for i in range (len(delEl)):
            resCol = resCol.drop(labels=delEl.index[i])'''

        return resCol

    def findNullsDF(data):
        rez = pd.Series()
        ind = 0
        row = 0

        for col in data:
            for el in data[col]:
                if SecondaryFuncs.isNull(el):
                    d = {col: row}
                    rez[ind.__str__()] = d
                    ind = ind + 1
                row = row + 1
                if row > len(data.index)-1:
                    row = 0

        return rez

    def findNullsSer(col):
        ids = []
        cnt = len(col)

        for i in range (cnt):
            if SecondaryFuncs.isNull(col[i]):
                ids.append(col.index[i])

        return ids

    def cleanNullsDF(data):
        resData = data
        delEl = Cleaning.findNullsDF(resData)
        delEl = list(delEl)
        rows = pd.Series()

        # colculate count of nulls in each row
        for i in range (len(delEl)):
            row = list(delEl[i].values())[0]

            if len(rows) == 0:
                rows[str(row)] = 1

            elif SecondaryFuncs.isIndInCol(rows, str(row)) == True:
                rows[str(row)] = rows[str(row)] + 1

            else:
                rows[str(row)] = 1

        cntCols = resData.shape[1]
        rowsList = list(rows)
        delRows = list()

        # check if any rows are totally nulls
        for i in range (len(rowsList)):
            if rowsList[i] == cntCols:
                delRows.append(i)

        for i in range (len(delRows)):
            delRows[i] = int(rows.index[delRows[i]])

        resData = resData.drop(delRows)

        return resData

    def cleanNullsSer(col):
        delEl = Cleaning.findNullsSer(col)
        resCol = col

        resCol = resCol.drop(labels=delEl)

        return resCol

    def cntOfSkipDataInDF(data):
        df = Cleaning.findSkipsDF(data)
        numb = len(df)
        return numb

    def cntOfSkipDataInColumn(col):
        cnt = 0
        for el in col:
            if el != el:
                cnt = cnt + 1
        return cnt

    def fillMissingData(data):
        resData = data
        nans = Cleaning.findSkipsDF(resData)

        for i in range (len(nans)):
            col = list(nans[i].keys())[0]
            row = list(nans[i].values())[0]

            ser = resData[col]
            ser.values[row] = Statistics.meanValue(ser)

        return resData

    def delDuplicates(data):
        tmp = pd.Series()
        indexes = []
        for i in range(len(data)):
            tmp[i.__str__()] = data.loc[i]
            for j in range(i):
                if pd.Series.equals(tmp[i.__str__()], tmp[j.__str__()]):
                    indexes.append(i)

        df = data.drop(data.index[indexes])
        return df

    def replacementMissings(data, cnt):
        return data.fillna(cnt)



class Statistics(object):

    def distributionFunc(col):
        return col.groupby(col).size().nlargest(len(col))

    def frequencyFunc(col):
        nums = pd.Series()

        # counts of each number in column:
        for el in col:
            if len(nums) == 0:
                nums[str(el)] = 1

            elif SecondaryFuncs.isIndInCol(nums, str(el)) == True:
                nums[str(el)] = nums[str(el)] + 1

            else:
                nums[str(el)] = 1

        plt.hist(col)
        #plt.show()                 tests

        return nums

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

    def relationsDetection(data):
        cntCol = data.shape[1]
        cols = list(data)
        indUnics = list()

        for i in range (cntCol):
            values = list(data[cols[i]])
            isUnic = True

            for j in range (len(values)):
                for k in range (len(values)):
                    if values[j] == values[k] and j != k:
                        isUnic = False

            if isUnic == True:
                indUnics.append(i)         # find indexes of unic columns (keys)

        rels = list()
        for i in range (cntCol):    # find pairs of {key include in column}
            for j in range (len(indUnics)):
                if i != indUnics[j]:    # same columns
                    isIncl = SecondaryFuncs.isColIncludedInCol(data[cols[i]], data[cols[indUnics[j]]])
                    if isIncl == True:
                        rel = PairsInRelations()
                        rel.setter(cols[indUnics[j]], cols[i])
                        rels.append(rel)

        for i in range(len(rels)):
            print(rels[i].key, rels[i].col)

        return rels



class Vizual(object):

    def datasetVisualizationDF(data):
        data.plot()
        plt.show()


    def datasetVisualizationSer(col):
        col.plot()
        plt.show()


class Report(object):

    def metadataReportDF(data, filename):
        data.to_excel(filename, sheet_name='report', na_rep='', header=True, index=True, merge_cells=MultiIndex, encoding='utf8', inf_rep='inf', verbose=True)

    def metadataReportSer(ser, filename):
        ser.to_excel(filename, sheet_name='report', na_rep='', header=True, index=True, merge_cells=MultiIndex, encoding='utf8', inf_rep='inf', verbose=True)


# ----------

class PairsInRelations(object):
    key = 0
    col = 0

    def __init__(self):
        """Constructor"""

    def setter(self, key, col):
        """Setter"""
        self.key = key
        self.col = col

    def isEq(p1, p2):
        if str(p1.key) == str(p2.key) and str(p1.col) == str(p2.col):
            return True
        return False


class SecondaryFuncs(object):

    def isColIncludedInCol(inputCol, col):
        isInCol = True

        for i in range(len(inputCol)):
            if SecondaryFuncs.isElInCol(col, inputCol[i]) == False:
                isInCol = False

        return isInCol

    def isElInCol(col, el):
        isInCol = False

        for i in range(len(col)):
            if col[i] == el:
                isInCol = True

        return isInCol

    def elType(el):
        return type(el)

    def isIndInCol(col, ind):
        cnt = len(col)
        for i in range(cnt):
            if str(col.index[i]) == str(ind):
                return True
        return False

    def cleanElsFromSer(els, col):
        resCol = col
        indsDel = els.index

        for i in range(len(indsDel)):
            resCol = resCol.drop(labels=[int(indsDel[i])])  # int ???????????????

        return resCol

    def isNull(cnt):
        return cnt == 0

    def sumSer(col):
        resCol = Cleaning.cleanSkipsSer(col)
        delEls = Profiling.findMistakes(resCol)
        resCol = SecondaryFuncs.cleanElsFromSer(delEls, resCol)

        sum = 0

        for el in resCol:
            sum = sum + el

        return sum

    def isEqDF(df1, df2):

        if df1.shape[0] * df1.shape[1] != df2.shape[0] * df2.shape[1] or df1.shape[0] != df2.shape[0] or df1.shape[1] != \
                df2.shape[1]:
            return False

        cols1 = list(df1)
        cols2 = list(df2)

        if cols1 != cols2:
            return False

        for col in cols1:
            for i in range(df1.shape[0]):
                if str(df1[col][i]) != str(df2[col][i]):
                    return False

        return True

    def isEqSer(ser1, ser2):
        col1 = Cleaning.cleanSkipsSer(ser1)
        col2 = Cleaning.cleanSkipsSer(ser2)

        if len(col1) != len(col2):
            return False

        for i in range(len(col1)):
            if SecondaryFuncs.isElInCol(list(col1.index), i) and SecondaryFuncs.isElInCol(list(col2.index), i) and col1[
                i] != col2[i]:
                return False

        return True

# ------------------------------------------------------------------------------------------------

'''data = 'price,count,percent\n1,10,\n1,30,\n3,20,51'
df = pd.read_csv(StringIO(data))
df.loc[3] = {'price': 4, 'count': 40, 'percent': 26.3}
df.loc[4] = {'price': 4, 'count': 50, 'percent': 26.3}'''

'''data = 'price,count,percent\n1,2,2\n2,3,4\n3,1,5'
df = pd.read_csv(StringIO(data))
df.loc[3] = {'price': 4, 'count': 5, 'percent': 5}
df.loc[4] = {'price': 5, 'count': 4, 'percent': 2}'''

#d = {"price":[1, 2, 3, 4, 5], "count": [2, 4, 4, 3, 1], "percent": [2, 4, 4, 3, 1]}
d = {"price": [1, 2, 3, 4, 5], "count": [1, 4, 3, 3, 1], "percent": [3, 4, 5, 1, 2]}
#d = {"price":[1, 2, 0, 4, 1], "count": [0, np.nan, 0, 3, 0], "percent": [24, 51, 0, 0, 24]}
#d = {"price": [1, 2, 10, 5, np.nan], "count": [0, 4, 0, 1, np.nan], "percent": [np.nan, 20, 0, 4, np.nan]}
#d = {"price": [1, 2, 0, 5, np.nan], "count": [0, 4, 0, 1, np.nan], "percent": [np.nan, 51, 0, 4, np.nan]}
#d = {"price": [0, 0, 0], "count": [0, 0, 0], "percent": [0, 0, 0]}
df = pd.DataFrame(d)
print(df)

#ser = pd.Series([np.nan, 20, 10, 0, 40, 0], ['a', 'b', 'c', 'd', 'e', 'f'])
#ser = pd.Series([22, 24, -60, 32, -200, 34, 200, 0, 24.0, 43, 44, 43, 57, 88, 150, '62', 67, 81], ['a', 'b', 'c', 'd', 'e', 'f', 'j', 'h', 'i', 'g', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r'])
#ser = pd.Series([-200, 0, '24.0', 'np.nan', 150, 62, 24.0], ['a', 'b', 'c', 'd', 'e', 'f', 'j'])
#ser = pd.Series([7,8,9,12,np.nan,14], ['a', 'd', 'e', 'j', 'i', 'g'])
#ser1 = pd.Series([7,8,9,12,np.nan,14], ['a', 'd', 'e', 'j', 'i', 'g'])
#ser = pd.Series([7,7,7,8,9,12,12,13,14], ['a', 'b', 'c', 'd', 'e', 'f', 'j', 'h', 'i'])
#ser = pd.Series([-20, 0, 20, 0, 15, np.nan, '42', -200, 12, 45, 10, 10, 0, 22])
ser = pd.Series([-10, 0, 10, 5])
#ser = pd.Series()
#print(ser)

print()

DP = DataProfiling()
DP.__setDF__(df)
DP.__setSeries__(ser)
print("--")

#'D:\\I\\Studies\\8_semester\\_Diploma\\DataProfiling\\report.xls'
print(DP.datasetVisualizationDF())
#print(DP.dataStandardization())
#print(SecondaryFuncs.isEqSer(ser,ser1))
