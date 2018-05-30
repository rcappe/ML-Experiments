class LogisticRegression:
    
    def __init__(self, data):
        self.data = data
        ''' Train and predict '''

        from sklearn.linear_model import LogisticRegression

        my_classifier =LogisticRegression(C=0.3, class_weight="balanced", random_state=42)
        my_classifier.fit(data.X_train, data.y_train.ravel())

        self.classifier = my_classifier

    def findBestC(self):
        from sklearn.linear_model import LogisticRegression
        import matplotlib.pyplot as plt 
        from sklearn import metrics

        C_start = 0.1
        C_end = 5
        C_inc = 0.1

        C_values, recall_scores = [], []

        C_val = C_start
        best_recall_score = 0
        while (C_val < C_end):
            C_values.append(C_val)
            lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", random_state=42)
            lr_model_loop.fit(self.data.X_train, self.data.y_train.ravel())
            lr_predict_loop_test = lr_model_loop.predict(self.data.X_test)
            recall_score = metrics.recall_score(self.data.y_test, lr_predict_loop_test)
            recall_scores.append(recall_score)
            if (recall_score > best_recall_score):
                best_recall_score = recall_score
                
            C_val = C_val + C_inc

        best_score_C_val = C_values[recall_scores.index(best_recall_score)]
        print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

        plt.plot(C_values, recall_scores, "-")
        plt.xlabel("C value")
        plt.ylabel("recall score")
        plt.show()