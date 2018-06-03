class Prediction:
    
    def __init__(self, classifier):
        from C_DataPreparation import DataPreparation
        import pandas as pd
                
        testData = pd.read_csv("./Titanic/data/test.csv")

        data = DataPreparation(testData.copy())
        data.removeUnusedColumn()
        data.moleData()

        test_Y = classifier.predict( data.df )

        test = pd.DataFrame( { 'PassengerId': testData.PassengerId , 'Survived': test_Y } )
        print(test.head())
        test.to_csv( '.\\Titanic\\titanic_pred.csv' , index = False )