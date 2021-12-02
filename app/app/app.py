from flask import Flask, request
from flask_apscheduler import APScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import create_engine
import pandas as pd
import json
import os
import logging
import psycopg2
import requests
import io

from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.impute import SimpleImputer

import sklearn

app = Flask(__name__)


@app.route('/health-check')
def health_check():
    return 'OK'

@app.route('/api_data', methods = ['GET'])
def api_data():

    response = requests.get(url = r'https://api.covidactnow.org/v2/state/WA.timeseries.csv?apiKey=6e156d960a9f45a886905e3008c2c35b')
    
    df = pd.read_csv(io.StringIO(response.text))
    
    engine = create_engine('postgresql+psycopg2://postgres:aserverfortheages!@database-2.cc9fdxmr2mkl.us-west-2.rds.amazonaws.com:5432/StewartR')

    df.head(0).to_sql('country', engine, if_exists='replace',index=False) #drops old table and creates new empty table

    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, 'country', null="") # null values become ''
    conn.commit()
    conn.close()
    return 'success'
#@app.route('/pull_data', methods = ['GET'])
#def get_all_data():


#    return country.to_json()

@app.route('/make_model', methods = ['GET'])
def make_model():
    
    response = requests.get(url = r'https://api.covidactnow.org/v2/state/WA.timeseries.csv?apiKey=6e156d960a9f45a886905e3008c2c35b')
    
    df = pd.read_csv(io.StringIO(response.text))

    data = pd.DataFrame(df)
    #get rid on unnecessary columns
    data_drop = data.drop(columns= ['country', 'state', 'metrics.testPositivityRatioDetails','county', 'lat', 'long', 'locationId', 'unused1', 'unused2', 'unused3', 'unused4', 'metrics.icuCapacityRatio', 'metrics.vaccinationsInitiatedRatio', 'metrics.vaccinationsCompletedRatio'])

    objects = data_drop.select_dtypes(include = ['object']).columns.to_list()
    # Use apply function to get unique values for every column
    print(data_drop[objects].apply(lambda col: col.unique()))

    X = data_drop[['fips', 'actuals.deaths', 'actuals.positiveTests', 'actuals.negativeTests', 'actuals.contactTracers', 'actuals.hospitalBeds.capacity', 'actuals.hospitalBeds.currentUsageTotal', 'actuals.hospitalBeds.currentUsageCovid', 'actuals.newCases', 'actuals.icuBeds.currentUsageTotal', 'actuals.icuBeds.currentUsageCovid', 'actuals.newCases', 'actuals.vaccinesDistributed', 'actuals.vaccinationsInitiated', 'actuals.vaccinationsCompleted', 'metrics.testPositivityRatio', 'metrics.caseDensity', 'metrics.contactTracerCapacityRatio', 'metrics.infectionRate', 'metrics.infectionRateCI90', 'riskLevels.overall', 'actuals.newDeaths', 'actuals.vaccinesAdministered', 'riskLevels.caseDensity', 'cdcTransmissionLevel']]
    y = data_drop[['actuals.cases']]
    y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=801)

    train = []
    train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    #train = train.reset_index()
    train = train.reset_index()
    train.drop('index', axis=1, inplace=True)
    train

    #Impute the values using scikit-learn SimpleImpute Class
    imp_mean = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'
    imp_mean.fit(train)
    imputed_train_df = imp_mean.transform(train)

    # turn back into DF and add column names
    train = pd.DataFrame(imputed_train_df)
    train.rename({0: 'fips', 1: 'actuals.deaths', 2: 'actuals.positiveTests', 3: 'actuals.negativeTests', 4: 'actuals.contactTracers', 5: 'actuals.hospitalBeds.capacity', 6: 'actuals.hospitalBeds.currentUsageTotal', 7: 'actuals.hospitalBeds.currentUsageCovid', 8: 'actuals.newCases', 9: 'actuals.icuBeds.currentUsageTotal', 10: 'actuals.icuBeds.currentUsageCovid', 11: 'actuals.newCases', 12: 'actuals.vaccinesDistributed', 13: 'actuals.vaccinationsInitiated', 14: 'actuals.vaccinationsCompleted', 15: 'metrics.testPositivityRatio', 16: 'metrics.caseDensity', 17: 'metrics.contactTracerCapacityRatio', 18: 'metrics.infectionRate', 19: 'metrics.infectionRateCI90', 20: 'riskLevels.overall', 21: 'actuals.newDeaths', 22: 'actuals.vaccinesAdministered', 23: 'riskLevels.caseDensity', 24: 'cdcTransmissionLevel', 25: 'actuals.cases'}, axis=1, inplace=True)

    X = train[['fips', 'actuals.deaths', 'actuals.positiveTests', 'actuals.negativeTests', 'actuals.contactTracers', 'actuals.hospitalBeds.capacity', 'actuals.hospitalBeds.currentUsageTotal', 'actuals.hospitalBeds.currentUsageCovid', 'actuals.newCases', 'actuals.icuBeds.currentUsageTotal', 'actuals.icuBeds.currentUsageCovid', 'actuals.newCases', 'actuals.vaccinesDistributed', 'actuals.vaccinationsInitiated', 'actuals.vaccinationsCompleted', 'metrics.testPositivityRatio', 'metrics.caseDensity', 'metrics.contactTracerCapacityRatio', 'metrics.infectionRate', 'metrics.infectionRateCI90', 'riskLevels.overall', 'actuals.newDeaths', 'actuals.vaccinesAdministered', 'riskLevels.caseDensity', 'cdcTransmissionLevel']]
    y = train[['actuals.cases']]
    y = pd.DataFrame(y)

    ##### Impute data with means of training data set #####

    #Impute the values using scikit-learn SimpleImpute Class
    imp_mean = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'
    imp_mean.fit(y_train)
    y_train = imp_mean.transform(y_train)

    # turn back into DF and add column names
    y_train = pd.DataFrame(y_train)
    y_train.rename({ 0: 'actuals.cases'}, axis=1, inplace=True)

    #Impute the values using scikit-learn SimpleImpute Class
    imp_mean = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'
    imp_mean.fit(X_train)
    X_train = imp_mean.transform(X_train)

    # turn back into DF and add column names
    X_train = pd.DataFrame(X_train)
    X_train.rename({0: 'fips', 1: 'actuals.deaths', 2: 'actuals.positiveTests', 3: 'actuals.negativeTests', 4: 'actuals.contactTracers', 5: 'actuals.hospitalBeds.capacity', 6: 'actuals.hospitalBeds.currentUsageTotal', 7: 'actuals.hospitalBeds.currentUsageCovid', 8: 'actuals.newCases', 9: 'actuals.icuBeds.currentUsageTotal', 10: 'actuals.icuBeds.currentUsageCovid', 11: 'actuals.newCases', 12: 'actuals.vaccinesDistributed', 13: 'actuals.vaccinationsInitiated', 14: 'actuals.vaccinationsCompleted', 15: 'metrics.testPositivityRatio', 16: 'metrics.caseDensity', 17: 'metrics.contactTracerCapacityRatio', 18: 'metrics.infectionRate', 19: 'metrics.infectionRateCI90', 20: 'riskLevels.overall', 21: 'actuals.newDeaths', 22: 'actuals.vaccinesAdministered', 23: 'riskLevels.caseDensity', 24: 'cdcTransmissionLevel'}, axis=1, inplace=True)


    #### Test Df
    #Impute the values using scikit-learn SimpleImpute Class
    imp_mean = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'
    imp_mean.fit(y_train)
    y_test = imp_mean.transform(y_test)

    # turn back into DF and add column names
    y_test = pd.DataFrame(y_test)
    y_test.rename({ 0: 'actuals.cases'}, axis=1, inplace=True)

    #Impute the values using scikit-learn SimpleImpute Class
    imp_mean = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'
    imp_mean.fit(X_train)
    X_test = imp_mean.transform(X_test)

    # turn back into DF and add column names
    X_test = pd.DataFrame(X_test)
    X_test.rename({0: 'fips', 1: 'actuals.deaths', 2: 'actuals.positiveTests', 3: 'actuals.negativeTests', 4: 'actuals.contactTracers', 5: 'actuals.hospitalBeds.capacity', 6: 'actuals.hospitalBeds.currentUsageTotal', 7: 'actuals.hospitalBeds.currentUsageCovid', 8: 'actuals.newCases', 9: 'actuals.icuBeds.currentUsageTotal', 10: 'actuals.icuBeds.currentUsageCovid', 11: 'actuals.newCases', 12: 'actuals.vaccinesDistributed', 13: 'actuals.vaccinationsInitiated', 14: 'actuals.vaccinationsCompleted', 15: 'metrics.testPositivityRatio', 16: 'metrics.caseDensity', 17: 'metrics.contactTracerCapacityRatio', 18: 'metrics.infectionRate', 19: 'metrics.infectionRateCI90', 20: 'riskLevels.overall', 21: 'actuals.newDeaths', 22: 'actuals.vaccinesAdministered', 23: 'riskLevels.caseDensity', 24: 'cdcTransmissionLevel'}, axis=1, inplace=True)


    # KNN REGRESSION
    knn = KNeighborsRegressor(n_neighbors=150)
    knn.fit(X_train, y_train)

    score = str(knn.score(X_train, y_train))
    
    return score

if __name__ == '__main__':

    app.run(host= '0.0.0.0', port=8000, debug = True)


