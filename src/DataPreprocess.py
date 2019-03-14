import pandas as pd
import numpy as np
import logging
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA


class DataPreprocessor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        self._logger.info('Data Preprocess Module Init..')

    @staticmethod
    def remove_dublicates(df):
        logging.info("Dropping dublicate entries..")
        df = df.drop_duplicates()
        return df

    @staticmethod
    def remove_outliers(df):
        logging.info("Removing outlier data..")
        # df = df[df.AppointmentDay > df.ScheduledDay]
        df = df[(df.Age > 0) & (df.Age < 100)]
        return df

    @staticmethod
    def rename_columns(df, nameDict):
        logging.info("Renaming columns..")
        df = df.rename(columns=nameDict)
        return df

    @staticmethod
    def encode_columns(df):
        logging.info("Encoding columns..")
        encoder = LabelEncoder()
        df['Gender'] = encoder.fit_transform(df['Gender'])
        df['Neighbourhood'] = encoder.fit_transform(df['Neighbourhood'])
        df['No-show'] = encoder.fit_transform(df['No-show'])
        df['ScheduleHourBins'] = encoder.fit_transform(df['ScheduleHourBins'])
        df['AppointmentHourBins'] = encoder.fit_transform(df['AppointmentHourBins'])
        return df

    # Find categorical columns and encode.
    def one_hot_encoder(self, df, columnList=None):
        if columnList is None:
            columnList = self.is_categorical(df)
        logging.debug('Encoded columns: %s', columnList)
        dummies = pd.get_dummies(df, columns=columnList)
        return dummies

    # If less than or equal to 10 percent of the column is unique then say it is categorical.
    @staticmethod
    def is_categorical(data):
        columns = data.columns.values
        n_rows, _ = data.shape
        isCategoricalList = []
        for col in columns:
            if round(len(data[col].unique()) * 100.0) / n_rows <= 10:
                isCategoricalList.append(col)
                logging.debug("Categorical column: %s", col)
        return isCategoricalList

    @staticmethod
    def choosePCAComponentNum(X_train):
        pca = PCA().fit(X_train)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig('../Plots/PCANumComponents.png')
        plt.show()

    @staticmethod
    def dimentionality_reduction(X_train, X_test):
        pca = PCA(40)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        logging.info("Shape of data after PCA: (%s, %s)" % X_train.shape)
        return X_train, X_test

    @staticmethod
    def normalizeData(X_train, X_test):
        scaler = MinMaxScaler()
        X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
        X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
        return X_train, X_test