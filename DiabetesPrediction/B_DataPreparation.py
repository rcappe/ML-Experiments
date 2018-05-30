class DataPreparation:
    
    def __init__(self):
        import pandas as pd    
        
        df = pd.read_csv("./DiabetesPrediction/data/pima-data.csv")

        ''' Explore data '''
        # Structure of file 768 rows e 10 colums
        #print( df.shape ) 

        # First 5 rows
        #print( df.head(5) ) 

        # Last 5 rows
        #print( df.tail(5) ) 

        ''' Check for null values '''
        #print(df.isnull().values.any())

        ''' Check the correlation '''
        #from utilities.plotu import plot_corr
        # Print chart
        #plot_corr(df)
        # Print table
        #print (df.corr())

        ''' Delete Column '''
        del df['skin']
        #print( df.head(5) ) 

        ''' Mold Data '''
        diabetes_map = {True : 1, False : 0}
        df['diabetes'] = df['diabetes'].map(diabetes_map)

        ''' Check class distribution '''
        #num_obs = len(df)
        #num_true = len(df.loc[df['diabetes'] == 1])
        #num_false = len(df.loc[df['diabetes'] == 0])
        #print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, (num_true/num_obs) * 100))
        #print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/num_obs) * 100))

        #Good distribution of true and false cases. No special work needed.

        ''' Spliting the data '''

        from sklearn.model_selection import train_test_split #sklearn.cross_validation is deprecaded

        feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
        predicted_class_names = ['diabetes']

        X = df[feature_col_names].values     # predictor feature columns (8 X m)
        y = df[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)
        split_test_size = 0.30

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 

        # Check percentage
        #print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
        #print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))


        # Check lable distribution
        #print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 1]), (len(df.loc[df['diabetes'] == 1])/len(df.index)) * 100.0))
        #print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 0]), (len(df.loc[df['diabetes'] == 0])/len(df.index)) * 100.0))
        #print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
        #print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
        #print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
        #print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))

        ''' Unexpected 0 values '''

        #print("# rows in dataframe {0}".format(len(df)))
        #print("# rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
        #print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
        #print("# rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))
        #print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
        #print("# rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
        #print("# rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
        #print("# rows missing age: {0}".format(len(df.loc[df['age'] == 0])))

        # Impute with the mean
        from sklearn.preprocessing import Imputer
        #Impute with mean all 0 readings
        fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

        X_train = fill_0.fit_transform(X_train)
        X_test = fill_0.fit_transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test