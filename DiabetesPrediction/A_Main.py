''' Pima Indian Diabetes Prediction '''
# https://github.com/JerryKurata/MachineLearningWithPython

from B_DataPreparation import DataPreparation
from C_NaiveBayes import NaiveBayes
from C_DecisionTree import DecisionTree
from C_RandomForest import RandomForest
from C_LogisticRegression import LogisticRegression
from D_Metrics import Metrics

data = DataPreparation()
'''
nbc = NaiveBayes(data)
print('-----------------NaiveBayes-----------------')
Metrics(data, nbc.classifier)

dtc = DecisionTree(data)
print('-----------------DecisionTree-----------------')
Metrics(data, dtc.classifier)

rfc = RandomForest(data)
print('-----------------RandomForest-----------------')
Metrics(data, rfc.classifier)
'''
lrc = LogisticRegression(data)
lrc.findBestC()
print('-----------------LogisticRegression-----------------')
Metrics(data, lrc.classifier)