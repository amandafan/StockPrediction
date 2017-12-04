#!/usr/bin/python3.6
'''
This Module generate the forcasting probabilty of stock increase x, and decrease y which x + y = 1
use import HMMlearning as hmmlearning
hmmlearning.Hmmlearning(name, start_date, end_date) to start training 
format: name--stock name，  start_date： mm/dd/yy  end_date： mm/dd/yy
use test.get_prob to return increase prob, and decrease prob correspondingly
    
'''
import pandas_datareader as pdr
import datetime
from hmmlearn import hmm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class HMMlearning:
    def __init__(self, stock, start, end):
        self.stock = stock
        self.start = start
        self.end = end

    def get_price_all(self, start, end):
        '''
        get historical data of one specific stock from yahoo Finance
        parameter: start--the start date  end--the end date
        '''
        try:
            data = pdr.get_data_yahoo(self.stock, start, end)
            dates = []
            close_v = []
            volume = []
            for date, price in data['Close'].items():
                dates.append(date)
                close_v.append(round(price,2))
            for date, v in data['Volume'].items():
                volume.append(v)
        except Exception as e:
            print ('There is exception when getting the stock price. Exception: %s' % e)

        return dates, close_v, volume

    def get_prob(self):
        '''
        calculate the probabliliy of increasing and decreasing
        param: stock--stock name, start_date and end_date
        '''
        #training process through hmm learning model
        mm, dd, yy = self.start.split("/")
        mm2, dd2, yy2 = self.end.split("/")
        start_date = datetime.datetime(int(yy), int(mm), int(dd))
        end_date = datetime.datetime(int(yy2), int(mm2), int(dd2))

        dates, close_v, volume = self.get_price_all(start_date, end_date)
        dates = np.array(dates)
        close_v = np.array(close_v)
        volume = np.array(volume)
        diff = np.diff(close_v)
        dates = dates[1:]
        close_v = close_v
        volume = volume
        X = np.column_stack([close_v])
        model = hmm.GaussianHMM(n_components=2, covariance_type="full",n_iter=10)
        model.fit(X)
        startprob = np.array([0.5, 0.5])

        mx = np.matrix(startprob)
        my = np.matrix(model.transmat_)

        prob = mx*my

        return round(prob.item(0), 5), round(prob.item(1), 5)

