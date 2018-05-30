class RandomForest:
    
    def __init__(self, data):
        ''' Train and predict '''

        from sklearn.ensemble import RandomForestClassifier
        my_classifier = RandomForestClassifier(random_state=42)      # Create random forest object
        my_classifier.fit(data.X_train, data.y_train.ravel()) 

        self.classifier = my_classifier