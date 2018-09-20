import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import datetime
import alpha_vantage
import requests
import statsmodels.api as sm
import statistics as sc


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

# Picking a few highly traded assets
symbols = ['DAX', '^GSPC', 'GOOG', 'MSFT']
data = pd.DataFrame()
for sym in symbols:
    params = {"function" : "TIME_SERIES_DAILY",
                     "symbol" : sym,
                     "outputsize" : 'full',
                     "apikey" : API_KEY} # params for daily historical data
    req = requests.get(URL, params)
    dat = req.json()
    data[sym] = to_panda(dat['Time Series (Daily)'])['close']
    data = data.dropna()

# plotting the normalized percentage gains/losses
(data / data.ix[0] * 100).plot(figsize=(8,6))
plt.grid()

# calculate log returns
log_returns = np.log(data / data.shift(1))
log_returns.hist(bins=50, figsize=(9,6))

# statistics
for sym in symbols:
    print("\nResults for symbol %s" % sym)
    print(30*"-")
    log_data = np.array(log_returns[sym].dropna())
    sc.print_statistics(log_data)

#qq plot, demonstrates that most assets have "fat tails", or outliers occur
# much more frequently than in normally distributed sets
sm.qqplot(log_returns['MSFT'].dropna(), line='s')
plt.grid()
plt.xlabel('theoretical sampling')
plt.ylabel('sample quantiles')


# finally, normality tests to demonstrate that the data is NOT normal
for sym in symbols:
    print("\nResults for symbol %s" % sym)
    print(30*"-")
    log_data = np.array(log_returns[sym].dropna())
    sc.normality_test(log_data)

plt.show()