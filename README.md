# Project Description

The dataset consists of 100k medical appointments in Brazil. This project aims to predict whether patients who have an appointment actually showed up or not based on various parameters such as age, gender, neighbourhood etc.

# Details

I have analyzed the data and made data cleaning first. This step includes renaming and correcting the feature names, removing the outliers and feature encoding. I have continued with the feature extraction from the existing data and I have eliminated some columns according to the correlation matrix and feature importance graphs. Then I have normalized the data and continued with modelling part. In this part I runned grid search for several algorithms and set the parameters based on that. Then I made cross validation and select the best model based on cv accuracy score. 

### Usage
```
$ git clone https://github.com/mervebozo/Medical-Appointments-No-shows.git
$ cd Medical-Appointments-No-shows
$ pip install -r requirements.txt
$ python ./src/medicalAppForecastor.py
```
