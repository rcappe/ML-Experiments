class NaiveBayes:
    
    def __init__(self, data):
        ''' Train and predict '''

        from sklearn.naive_bayes import GaussianNB

        my_classifier = GaussianNB()
        my_classifier.fit(data.X_train, data.y_train.ravel())

        self.classifier = my_classifier
        