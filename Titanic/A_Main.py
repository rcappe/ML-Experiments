from B_DataExploration import DataExploration
from C_DataPreparation import DataPreparation
from D_Metrics import Metrics
from E_Prediction import Prediction
import pandas as pd
        
trainData = pd.read_csv("./Titanic/data/train.csv")

de = DataExploration(trainData)
#de.head()
#de.describe()
#de.charts()
#de.sexAnalysis()
#de.pclass()
#de.plot_correlation_map()

data = DataPreparation(trainData)

data.removeUnusedColumn()
data.moleData()
data.split()

from sklearn.linear_model import LogisticRegression
my_classifier =LogisticRegression(C=0.8, class_weight="balanced", random_state=42)
my_classifier.fit(data.X_train, data.y_train.ravel())

Metrics(data, my_classifier)

#Prediction(my_classifier)


