import requests
import pandas as pd
from flask import Flask 
import financialanalysis as fa
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.model_selection import train_test_split
import pickle

app =Flask(__name__)
def train():
    url="https://edonations.000webhostapp.com/api-last-day-donation.php"
    headers = {'Content-Type': 'application/json' }
    response = requests.get(url=url,headers=headers)
    df = pd.read_json(response.text)
    if df.empty:
        pass
    else:
        df.drop([u'name',u'donationId',u'name'],inplace=True,axis=1)
        X = df["date"].to_list() # convert Series to list
        X = fa.datetimeToFloatyear(X) # for example, 2020-07-01 becomes 2020.49589041
        # [2020.0054794520547, 2020.0082191780823, 2020.01643835616, ...]
        df["date"] = X 
        # separate the independent and target variable 
        train_X = df.drop(columns=['quantity'])
        train_Y = df['quantity']
        # randomly split the data
        train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y,test_size=0.25,random_state=0)
        # create an object of the LinearRegression Model
        model1 = pickle.load(open('lassoModel.pkl', 'rb'))
        model2 = pickle.load(open('linearModel.pkl', 'rb'))
        model3 = pickle.load(open('ridgeModel.pkl', 'rb'))
        # fit the model with the training data
        model1.fit(train_x, train_y)
        model2.fit(train_x, train_y)
        model3.fit(train_x, train_y)
        #save traning
        pickle.dump(model1,open('lassoModel.pkl', 'rb'))
        pickle.dump(model2, open('linearModel.pkl', 'rb'))
        pickle.dump(model3,open('ridgeModel.pkl', 'rb'))

if __name__ == '__main__':
    scheduler = BackgroundScheduler({'apscheduler.timezone': 'UTC'}, daemon=True)
    scheduler.add_job(func=train, trigger='interval',hour = 24 )
    scheduler.start()
#Exit on app exit
    atexit.register(lambda: scheduler.shutdown())
    app.run(debug = False)   