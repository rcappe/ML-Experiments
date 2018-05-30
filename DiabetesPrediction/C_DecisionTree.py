class DecisionTree:
    
    def __init__(self, data):

        ''' Train and predict '''

        from sklearn import tree
        my_classifier = tree.DecisionTreeClassifier()

        my_classifier.fit(data.X_train, data.y_train.ravel())

        self.classifier = my_classifier