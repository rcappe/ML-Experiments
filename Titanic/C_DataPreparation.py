class DataPreparation:
    
    def __init__(self, data):
        self.df = data

    def removeUnusedColumn(self ):
        # Delete unusfull columns
        del self.df['Name']
        del self.df['Ticket']
        del self.df['Embarked'] #Port of Embarkation
        del self.df['Fare']

        del self.df['SibSp']
        del self.df['Parch']
        del self.df['PassengerId']

    def moleData(self ):
        # Mold Data
        ## --- Sex ---
        sex_map = {'male' : 1, 'female' : 0}
        self.df['Sex'] = self.df['Sex'].map(sex_map)

        ## --- Cabin ---
        #Fill NaN with 0
        self.df['Cabin'].fillna(0, inplace=True)
        
        def cleanCabin( cabin ):
            if cabin == 0:
                return 0
            else: 
                return 1

        #Fill != 0 with 1
        self.df['Cabin'] = self.df['Cabin'].map(cleanCabin)

        ## --- Age ---
        self.df.Age.fillna( self.df.Age.mean() , inplace=True)


    def split(self ):
        ''' Split '''
        from sklearn.model_selection import train_test_split 

        feature_col_names = ['Pclass', 'Sex', 'Age', 'Cabin']
        predicted_class_names = ['Survived']

        X = self.df[feature_col_names].values     
        y = self.df[predicted_class_names].values 
        split_test_size = 0.3

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size) 

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
