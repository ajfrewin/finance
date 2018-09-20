import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import datetime
import alpha_vantage
import requests
from math import log, exp, sqrt

''' Stock object with various home-brewed functions for basic analytics'''

today = datetime.datetime.today() # today's date
URL = 'https://www.alphavantage.co/query' #alpha vantage url
API_KEY = '0AQPGOXILXLKKF2W'# key to use alpha vantage api
R = 0.019 # risk-free interest rate

def to_panda(dictionary):
    df = pd.DataFrame(index=list(dictionary.keys()),columns=['open','high','low','close','volume'])
    # create a dataframe with index of dates/times, columns corresponding to dict
    i = 0 # for indexing
    for key in dictionary:
        df.iloc[i] = list(dictionary[key].values())
        # sets the appropriate row of the dataframe to the
        # appropriate values
        i = i + 1

    return df.astype(float)[::-1] # convert values to floats, reverse index order
                                # to be in line with what datareader gave

class Stock:
    # This stock can do stuff

    def __init__(self, symbol):
        # constructor based on only ticker symbol, creates a structure where
        # you can access historical data, todays data, and get a quote from
        # the most recent minute.
        # can also price options and compute technical indicators

        self.symbol = symbol

        hist_params = {"function" : "TIME_SERIES_DAILY",
                     "symbol" : symbol,
                     "apikey" : API_KEY} # params for daily historical data

        req = requests.get(URL, hist_params)
        data = req.json()

        self.historical = to_panda(data['Time Series (Daily)'])
        # historical data from past 100 days
        # -> consider a way to select date ranges,
        # as of now av can only pull past 100 days, or ALL available

        today_params = {"function" : "TIME_SERIES_INTRADAY",
                     "symbol" : symbol,
                     "interval" : "1min",
                     "outputsize": "full",
                     "apikey" : API_KEY} # params for today's tick data in 1min intervals

        req = requests.get(URL,today_params)
        data = req.json() # convert to dict
        self.today = to_panda(data['Time Series (1min)']) # puts today's tick data into a dataframe
        self.latest = self.today['close'][-1] # latest price, "current quote" more or less
        self.historical['Return'] = np.log(self.historical['close'] / self.historical['close'].shift(1))
        # log returns
        self.historical['42d'] = self.historical['close'].rolling(window=42).mean() # 42 day average
        self.historical['252d'] = self.historical['close'].rolling(window=252).mean() # 252 day average
        self.historical['Mov_Vol'] = self.historical['Return'].rolling(window=42).std() # 42 day moving vol
        self.hist_vol = np.std(self.historical['Return'])

    def get_historical(self):
        # returns dataframe of historical data, useful
        # if you want a variable thats just the historical data
        return self.historical.copy()

    def quote(self):
        # returns latest price
        print('Current quote for ' + self.symbol + ':', '$', self.latest, )
        return self.latest

    def tail(self):
        # equivalent to tail from pandas
        print(self.historical.tail())

    def head(self):
        # equivalent to head from pandas
        print(self.historical.head())

    def day_volume(self):
        # volume traded so far today
        return self.today['volume'].cumsum()[-1]

    def eu_eval(self, strike, time, type='call', output=False):
        # uses Black-Scholes formula and historical volatility
        # to appropriately price a EU call at the given strike
        # for exercise on the given date
        t = time/365
        s0 = self.latest
        sigma = self.hist_vol # historical volatility

        s0 = 100.
        R = 0.05
        sigma = 0.25

        # Black-Scholes formula
        d1 = (log(s0/strike) + (R + (sigma**2)/2)*t)/(sigma*sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        C0 = (s0 * st.norm.cdf(d1) - strike * exp(-R * t) * st.norm.cdf(d2))
        if not(type=='call'):
            C0 = C0 - s0 + strike*np.exp(-R*t)
        if output:
            print('Appropriate ' + type + ' premium for ' + self.symbol + ' with strike ' + str(strike) + ' expiring in ' +
              str(time) + ' days: \n$', round(C0, 2))
        return C0

    def gbm_mcs_dyna(self, K, ttm, type='call'):
        ''' Valuation of EU option in BSM by Monte Carlo sim
        Params
        =======
        K: float
            (positive) strike price
        ttx: float
            time to maturity (in years)
        type: string
            'call' for call option, computes for put for all other entries
        '''

        I = 50000
        M = 50
        dt = ttm / M
        sn = np.random.standard_normal((M + 1, I))
        sigma = self.hist_vol
        s0 = 100.
        R = 0.05
        sigma = 0.25
        S = np.zeros((M+1, I))
        S[0] = s0
        # simulate index level at maturity
        for t in range(1, M + 1):
            S[t] = S[t-1] * np.exp((R - 0.5 * sigma ** 2) * dt
                                   + sigma * np.sqrt(dt) * sn[t])
        # calculating payoff for each case
        if(type=='call'):
            hT = np.maximum(S[-1]-K, 0)
        else:
            hT = np.maximum(K - S[-1], 0)

        # calculate estimator
        C0 = np.exp(-R * ttm) * 1 / I * np.sum(hT)
        return C0

    def bsm(self,time):
        S0 = self.latest
        M = time # number of days = steps
        dt = 1./365 # one day intervals, expressed in years, MUST BE SMALL
        I = 5000 # number of simulations
        sigma = self.hist_vol

        S = np.zeros((M+1, I))
        S[0] = S0
        for t in range(1,M+1):
            S[t] = S[t-1] * np.exp((R - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*np.random.standard_normal(I))

        return S

    def option_payoff(self, strike, prem, ttx):
        S = self.bsm(ttx) # run stochastic simulation
        breakeven = strike + prem # breakeven price
        S = S.T # flip to check each simulation
        count = 0
        for sim in S:
            where_breakeven = np.where(sim>=breakeven)[0] # check if underlying price
            # exceeds breakeven price in the given simulation
            if not(where_breakeven.size == 0): #
                count = count+1

        numSims = len(S[0]) # total number of simulations
        percent = float(count/numSims) # percentage of cases where profit is realized
        return percent

    def plot_hist(self):
        self.historical[['close','Return','Mov_Vol']].plot(subplots=True,figsize=(8,6), grid=True)

# driver below
MU = Stock('DAX')
print(MU.gbm_mcs_dyna(95., 1.,type='put'))
print(MU.eu_eval(95, 365., type='put'))