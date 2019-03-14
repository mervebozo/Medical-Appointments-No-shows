from src.DataPreprocess import DataPreprocessor
from src.FeatureExtractor import FeatureExtractor
from src.ModelCreator import ModelCreator
from sklearn.model_selection import train_test_split
import logging
import configparser
import os
import pandas as pd


class medicalAppForecastor:
    def __init__(self, config):
        logging.basicConfig(level=logging.DEBUG)
        self._logger = logging.getLogger(__name__)
        self._logger.info('Init..')
        self._dataRootPath = config['Path']['dataRootPath']
        self.__modelBasePath = config['Path']['modelRootPath']
        self._fileName = config['Path']['fileName']
        self._label = config['Model']['label']
        self._filePath = os.path.join(self._dataRootPath, self._fileName)
        self._dp = DataPreprocessor()
        self._fe = FeatureExtractor()
        self._mc = ModelCreator(config)

    def readData(self):
        df = pd.read_csv(self._filePath, index_col='PatientId')
        return df

    def splitData(self, df, y):
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=2)
        self._logger.debug("Train Shape: %s", X_train.shape)
        self._logger.debug("Test Shape: %s", X_test.shape)
        self._logger.info('Training Data Distribution: \n %s', y_train.value_counts(normalize=True).to_frame())
        self._logger.info('Validation Data Distribution: \n %s \n', y_test.value_counts(normalize=True).to_frame())
        return X_train, X_test, y_train, y_test

    def prepareData(self, df):
        df = self._dp.rename_columns(
            df, {'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'AppointmentID': 'AppointmentId'})
        df.drop(['Hypertension', 'Diabetes', 'Handicap'], axis=1, inplace=True)
        df = self._dp.remove_outliers(df)
        df = self._fe.getDateBasedFeatures(df)
        df = self._fe.getFeatureBins(df)
        df = self._dp.encode_columns(df)
        y = df.pop(self._label)
        X = self._dp.one_hot_encoder(df)
        return X, y

    def getModelResults(self, modelName, X_test, y_test):
        logging.info("Getting %s model results on test data: " % modelName)
        y_preds = self._mc.getBestModelResults(self.__modelBasePath, modelName, X_test)  # get model
        self._mc.calculateError(modelName, y_preds, y_test)  # calculate error

    def main(self):
        df = self.readData()
        X, y = self.prepareData(df)
        X_train, X_test, y_train, y_test = self.splitData(X, y)
        X_train, y_train = self._fe.oversamplingData(X_train, y_train)
        X_train, X_test = self._dp.normalizeData(X_train.astype(float), X_test.astype(float))
        # X_train, X_test = self._dp.dimentionality_reduction(X_train, X_test)
        tunedModels = self._mc.getTunedModels()
        self._mc.saveModels(tunedModels, self.__modelBasePath, X_train, y_train)
        cv_results = self._mc.crossValidateModels(tunedModels, X_train, y_train)

        best_model = cv_results[0]
        self.getModelResults(best_model[0][0], X_test, y_test)


if __name__ == '__main__':
    try:
        config = configparser.ConfigParser()
        config.read('config.cfg')
        mf = medicalAppForecastor(config)
        mf.main()
    except:
        logging.exception("Exception")
