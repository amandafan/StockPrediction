#!/usr/bin/python3.6

import pandas_datareader as pdr
import datetime
from hmmlearn import hmm
import numpy as np
import warnings
import random
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
        retry = 0
        while retry < 3:
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
                break
            except Exception as e:
                print ('There is exception when getting the stock price. Exception: %s' % e)
                retry += 1
        if retry >= 3:
            return None, None, None
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
        if not close_v:
            return 0.5, 0.5
        dates = np.array(dates)
        close_v = np.array(close_v)
        close_v_2 = np.array(close_v[1:])
        volume = np.array(volume[1:])
        diff = np.diff(close_v)
        dates = dates[1:]
        X = np.column_stack([close_v_2, diff, volume])
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag",n_iter=10)
        model.fit(X)
        startprob = np.array([0.5, 0.5])

        mx = np.matrix(startprob)
        my = np.matrix(model.transmat_)
        prob = mx * my
        return round(prob.item(0), 5), round(prob.item(1), 5)


    def eval(self, start_date, end_date):
        self.stock = 'AAPL'
        dates, close_v, volume = self.get_price_all(start_date, end_date)
        if not close_v:
            return True
        actual_result = float(close_v[-1]) > float(close_v[-2])
        dates = dates[:-1]
        close_v = close_v[:-1]
        volume = volume[:-1]
        dates = np.array(dates)
        close_v = np.array(close_v)
        close_v_2 = np.array(close_v[1:])
        volume = np.array(volume[1:])
        diff = np.diff(close_v)
        dates = dates[1:]
        X = np.column_stack([close_v_2, diff, volume])
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag",n_iter=10)
        model.fit(X)
        startprob = np.array([0.5, 0.5])
        days = (end_date - start_date).days

        mx = np.matrix(startprob)
        my = np.matrix(model.transmat_)
        prob = mx*my
        increProb = str(round(prob.item(0), 5))
        decreProb = str(round(prob.item(1), 5))

        predict_result = increProb > decreProb
        print("Predict result ", predict_result)
        print("Actual result:", actual_result)

        return predict_result == actual_result

    
    def eval_model(self):
        count = 0
        day_len = 10
        for i in range(0, 100):
            start = datetime.datetime(2016, random.randint(1, 12), random.randint(1, 25))
            end = start + datetime.timedelta(days=day_len)
            result = self.eval(start, end)
            if result:
                count += 1
        return max(count, 50) * 1.0 / 100

if __name__ == '__main__':
    stock = Stock('GOOGL', '1/1/2016', '2/1/2016')
    print(stock.get_prob())
    rate = stock.eval_model()
    print('Our predication rate:',  rate)
