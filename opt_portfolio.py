import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import pandas_datareader.data as web

'''
This script demonstrates how expected return and voltatility is distributed
amongst a portfolio of several securities, as well as demonstrating the concept
of the efficient frontier, the line along which all optimal portfolios lie
'''

# Picking a few highly traded traded
symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "DB"]
noa = len(symbols) # number of assets

''' Following lines used for initial import
data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='yahoo', end='2014-09-12')['Adj Close']
data.to_csv('opt_portfolio_example.csv', ',')
'''
# Loading data from a saved CSV because data_reader me up too much
data = pd.read_csv('opt_portfolio_example.csv',',', index_col=0)

rets = np.log(data / data.shift(1)) # logarithmic returns

# Use Monte-carlo simulation to generate random portfolio weights on a large scale
prets = []
pvols = []
for p in range(2500):
    weights = np.random.random(noa) # random weights on stock purchases
    weights /= np.sum(weights) # normalizing
    prets.append(np.sum(rets.mean() * weights) * 252) # expected returns
    pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))) # expected variance
prets = np.array(prets)
pvols = np.array(pvols)

def statistics(weights):
    '''Returns portfolio statistics

    :param weights: array-like
                weights for different securities in portfoliol
    :return:
    pret : float
        expected return
    pvol : float
        expected portfolio volatility
    pret / pvol : float
        Sharpe ratio for rf (risk-free short rate) = 0
    '''
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret/pvol])

def min_func_sharpe(weights):
    # Function to minimize sharpe value
    return -statistics(weights)[2]

def min_func_variance(weights):
    # function to minimize absolute variance
    return statistics(weights)[1]**2

def min_func_port(weights):
    # function to minimize variate, going to be iterated thru
    return statistics(weights)[1]

# constraining that parameters add up to 1
cons = ({'type' : 'eq', 'fun' : lambda x: np.sum(x) - 1})
# bound the weights between 0 and 1
bnds = tuple((0,1) for x in range(noa))

# highest sharp ratio portfolio
opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
# absolute minimum variance portfolio
optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
# note: using noa * [1. / noa,] as initial guess, even distribution of securities


# to compute the efficient frontier, we evaluate the minimum variance portfolio
# a ta given target return
trets = np.linspace(0., 0.25, 50)
tvols = []
for tret in trets:
    cons = ({'type' : 'eq', 'fun' : lambda x: statistics(x)[0] - tret},
            {'type' : 'eq', 'fun' : lambda x: np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(8,6))
plt.scatter(pvols, prets, c=prets / pvols, marker='o') # random compositions
plt.scatter(tvols, trets, c=trets/tvols, marker='o') # efficient frontier
plt.plot(tvols, trets, 'r')
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0], 'r*', markersize=15.0) # highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'k*', markersize=15.0) # minimum variance
plt.grid()
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.legend(['Efficient frontier','Max Sharpe Ratio','Minimum Variance'], loc=3)

plt.show()
