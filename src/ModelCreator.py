import logging
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import \
    GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import model_selection
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
import time
import os
import pickle


class ModelCreator:
    def __init__(self, config):
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        self._logger.info('Model Creator Module Init..')
        self._label = config['Model']['label']

    @staticmethod
    def getBestModelResults(modelBasePath, modelName, X_test):
        # Load model from file
        with open(os.path.join(modelBasePath, (modelName + '.pkl')), 'rb') as file:
            best_model = pickle.load(file)
        preds = best_model.predict(X_test)
        return preds

    def getTunedModels(self):
        # get models with tuned parameters
        tunedModels = list()
        tunedModels.append(('BernoulliNB', BernoulliNB(alpha=0.5)))
        tunedModels.append(('LogisticReg', LogisticRegression(max_iter=1000, solver='lbfgs')))
        tunedModels.append(('RandomForest', RandomForestClassifier(criterion='gini', max_depth=50, n_estimators=10)))
        tunedModels.append(('XGBoost', xgb.XGBClassifier(gamma=0.2, learning_rate=0.1, max_depth=5, min_child_weight=3,
                                                         n_estimators=300)))
        tunedModels.append(('MLP', MLPClassifier(batch_size=64, hidden_layer_sizes=(4, 3), max_iter=300)))
        tunedModels.append(('Adaboost', AdaBoostClassifier(learning_rate=1, n_estimators=50)))
        tunedModels.append(('Bagging', BaggingClassifier(max_samples=1.0, n_estimators=10)))
        tunedModels.append(('GB', GradientBoostingClassifier(max_depth=3, n_estimators=100)))
        tunedModels.append(('DecisionTree', DecisionTreeClassifier(criterion='gini', max_depth=None)))
        return tunedModels

    @staticmethod
    def getModels():
        # get models for grid cv
        models = list()
        models.append(('XGBoost', xgb.XGBClassifier()))
        models.append(('MLP', MLPClassifier()))
        models.append(('Adaboost', AdaBoostClassifier()))
        models.append(('Bagging', BaggingClassifier()))
        models.append(('ExtraTrees', ExtraTreesClassifier()))
        models.append(('GB', GradientBoostingClassifier()))
        models.append(('RandomForest', RandomForestClassifier()))
        models.append(('LogisticReg', LogisticRegression()))
        models.append(('BernoulliNB', BernoulliNB()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('DecisionTree', DecisionTreeClassifier()))
        return models

    def tuneModels(self, modelsToGridCV, X_train, y_train):
        grid_n_estimator = [50, 100]
        grid_ratio = [.5, 1.0]
        grid_learn = [.1, .3, .5, 1.0]
        grid_max_depth = [2, 3, 4, None]
        grid_criterion = ['gini', 'entropy']

        grid_param = [
            {
                # XGBClassifier
                'learning_rate': grid_learn,  # default: .3
                'max_depth': [3, 4, 5],  # default 2
                'n_estimators': [200, 300],
                'gamma': [0.2],
                'min_child_weight': [3],
            },
            {
                # MLPClassifier
                'hidden_layer_sizes': [(7, 2), (4, 3)],
                'batch_size': [64, 200],
                'max_iter': [300, 500]
            },
            {
                # AdaBoostClassifier
                'n_estimators': grid_n_estimator,  # default=50
                'learning_rate': grid_learn  # default=1
            },
            {
                # BaggingClassifier
                'n_estimators': grid_n_estimator,  # default=10
                'max_samples': grid_ratio  # default=1.0
            },
            {
                # ExtraTreesClassifier
                'n_estimators': grid_n_estimator,  # default=10
                'criterion': grid_criterion,  # default=”gini”
                'max_depth': grid_max_depth  # default=None
            },
            {
                # GradientBoostingClassifier
                'learning_rate': grid_learn,  # default=0.1
                'n_estimators': [100, 300],  # default=100
                'max_depth': grid_max_depth  # default=3
            },
            {
                # RandomForestClassifier
                'n_estimators': [10, 20],  # default=10
                'criterion': ['gini'],  # default=”gini”
                'max_depth': [10, 50, None],  # default=None
            },
            {
                # LogisticRegression
                'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],  # default: lbfgs
                'max_iter': [1000]
            },
            {
                # BernoulliNB
                'alpha': grid_ratio  # default: 1.0
            },
            {
                # KNeighborsClassifier
                'n_neighbors': [3, 5, 7, 9, 11, 13],  # default: 5
                'weights': ['uniform', 'distance'],  # default = ‘uniform’
            },
            {
                # DecisionTree
                'criterion': grid_criterion,  # default: gini
                'max_depth': grid_max_depth # default None
            }
        ]

        models = self.getModels()
        tunedModels = list()
        for model, param in zip(models, grid_param):
            if model[0] in modelsToGridCV:
                start = time.perf_counter()
                logging.info("Start time: %s." % datetime.now().strftime('%H:%M:%S'))
                logging.info("\n Model: %s \n Params: %s " % (model[0], param))
                grid_search = model_selection.GridSearchCV(estimator=model[1], scoring='roc_auc', param_grid=param, cv=10)\
                    .fit(X_train, y_train)
                run = time.perf_counter() - start
                logging.info("Finish time: %s." % datetime.now().strftime('%H:%M:%S'))
                logging.info("Run time: %.2f seconds." % run)
                logging.info("Best cross-validation accuracy: {:.4f}".format(grid_search.best_score_))
                logging.info("Best parameters: {}".format(grid_search.best_params_))
                model[1].set_params(**grid_search.best_params_)
                tunedModels.append(model)
        return tunedModels

    def ensembleModel(self, X, y):
        best3classifiers=self.getModels()[2:5]
        from sklearn.ensemble import VotingClassifier
        voting_est = VotingClassifier(estimators=best3classifiers, voting='hard')
        cross = model_selection.cross_val_score(voting_est, X, y, cv=10, scoring="accuracy")
        print('The cross validated score is', cross.mean())
        return voting_est

    def calculateError(self, modelName, y_preds, y_test):
        score = metrics.accuracy_score(y_test, y_preds)
        roc_score = metrics.roc_auc_score(y_test, y_preds)
        f1_score = metrics.f1_score(y_test, y_preds, average='weighted')
        fpr, tpr, thresholds = metrics.roc_curve(y_preds, y_test)
        roc_auc = metrics.auc(fpr, tpr)

        logging.info("%s Accuracy of model %s: " % (modelName, score))
        logging.info("%s ROC Accuracy of model %s: " % (modelName, roc_score))
        logging.info("%s F1 Score of model %s: " % (modelName, f1_score))

        conf_matrix = metrics.confusion_matrix(y_test, y_preds)
        title = "%s Confusion Matrix" % modelName
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="YlGnBu").set_title(title)  # font size
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(fpr, tpr, color='g', label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%s - ROC Curve' % modelName)
        plt.tight_layout()
        plt.show()
        plt.close()

        print("%s - Classification Report \n" % modelName)
        print(metrics.classification_report(y_test, y_preds))
        return score

    def crossValidateModels(self, models, X_train, y_train):
        # evaluate each model
        results = []
        for name, model in models:
            y_pred = model_selection.cross_val_predict(model, X_train, y_train, cv=10)
            accuracy = metrics.accuracy_score(y_pred, y_train)
            results.append([name, accuracy, model])
            self.calculateError(name, y_pred, y_train)

        df = pd.DataFrame(results, columns=['Name', 'Accuracy', 'Model'])
        df = df.set_index(df.Name).sort_values('Accuracy', ascending=False)
        df.drop(['Name', 'Model'], axis=1, inplace=True)

        df['Accuracy'].plot.barh(width=0.8)
        plt.title('CV Mean Accuracy Comparison of Algorithms')
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        plt.show()
        plt.close()

        plt.subplots(figsize=(12, 6))
        df.T.boxplot(column=list(df.T.columns))
        plt.show()
        plt.close()

        def getKey(item): return item[1]
        results = sorted(results, key=getKey, reverse=True)
        logging.info("Best Model Based on CV: %s with accuracy: %.3f" % (results[0][0], results[0][1]))
        return results

    @staticmethod
    def saveModels(models, modelBasePath, X_train, y_train):
        for model_ in models:
            modelName, model = model_
            model.fit(X_train, y_train)
            pkl_filename = os.path.join(modelBasePath, "%s.pkl" % modelName)   # Save model to Models directory
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)
