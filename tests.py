# Python 3
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import math
import unittest
import DataProfiling

class setDFTests(unittest.TestCase):
    df = pd.DataFrame()

    def test_correct(self):
        data = 'price,count,percent\n1,10,\n1,10,\n3,20,51'
        df = pd.read_csv(StringIO(data))
        df.loc[3] = {'price': 4, 'count': None, 'percent': 26.3}
        df.loc[4] = {'price': 4, 'count': 50, 'percent': 26.3}

        result = DataProfiling.__setDF__(df)
        self.assertEqual(result, df)