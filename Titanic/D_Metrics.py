        
class Metrics:
    
    def __init__(self, data, classifier):
        ''' Performance on Training Data '''

        # predict values using the training data
        nb_predict_train = classifier.predict(data.X_train)

        # import the performance metrics library
        from sklearn import metrics

        # Accuracy
        print()
        print("Accuracy Train: {0:.4f}".format(metrics.accuracy_score(data.y_train, nb_predict_train)))
        

        ''' Performance on Testing Data '''

        # predict values using the testing data
        nb_predict_test = classifier.predict(data.X_test)

        # training metrics
        print("Accuracy Test: {0:.4f}".format(metrics.accuracy_score(data.y_test, nb_predict_test)))
        print("")
        
        ''' Metrics '''

        print("Confusion Matrix")
        print("{0}".format(metrics.confusion_matrix(data.y_test, nb_predict_test)))
        print("")

        print("Classification Report")
        print(metrics.classification_report(data.y_test, nb_predict_test))