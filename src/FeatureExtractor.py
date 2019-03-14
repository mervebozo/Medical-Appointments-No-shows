import pandas as pd
import logging


class FeatureExtractor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        self._logger.info('Feature Extractor Module Init..')

    @staticmethod
    def getDateBasedFeatures(df):
        logging.info('Add date based features...')
        df['WaitingTime'] = (pd.to_datetime(df['AppointmentDay']) - pd.to_datetime(df['ScheduledDay']))
        df['WaitingTime'] = df['WaitingTime'].apply(lambda x: x.total_seconds() / (3600 * 24))
        df['AppointmentDayOfWeek'] = pd.to_datetime(df['AppointmentDay']).dt.dayofweek
        df['AppointmentHour'] = pd.to_datetime(df['AppointmentDay']).dt.hour
        df['ScheduledDayOfWeek'] = pd.to_datetime(df['ScheduledDay']).dt.dayofweek
        df['ScheduledHour'] = pd.to_datetime(df['ScheduledDay']).dt.hour
        logging.info('Drop unused / non-informative features...')
        df.drop(['AppointmentDay', 'ScheduledDay', 'AppointmentId'], axis=1, inplace=True)
        return df

    @staticmethod
    def getFeatureBins(df):
        logging.info('Change Age to categories...')
        labels = list(range(8))
        age_bins = pd.cut(df.Age, 8, labels=labels)
        df['AgeBins'] = age_bins

        logging.info('Change Schedule Hour to categories...')
        df['ScheduleHourBins'] = pd.cut(df.ScheduledHour, [0, 8, 12, 18, 24],
                                        labels=['Night', 'Morning', 'Afternoon', 'Evening'])

        logging.info('Change Schedule Hour to categories...')
        df['AppointmentHourBins'] = pd.cut(df.AppointmentHour, [0, 8, 12, 18, 24],
                                           labels=['Night', 'Morning', 'Afternoon', 'Evening'])

        logging.info('Drop unused features...')
        df.drop(['Age', 'ScheduledHour', 'AppointmentHour'], axis=1, inplace=True)
        return df

    def oversamplingData(self, X_train, y_train):
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=12, ratio=1.0)
        columns = X_train.columns
        logging.debug("Before oversampling: %s", y_train.value_counts(normalize=True).to_frame())
        os_data_X, os_data_y = smote.fit_sample(X_train, y_train)
        os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
        os_data_y = pd.DataFrame(data=os_data_y, columns=['No-show'])
        logging.debug("After oversampling: %s", os_data_y['No-show'].value_counts(normalize=True).to_frame())
        return os_data_X, os_data_y.values.ravel()
