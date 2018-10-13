import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci

# Asset price data library
from quantopian.research import prices, symbols

'''
This script demonstrates how expected return and voltatility is distributed
amongst a portfolio of several securities, as well as demonstrating the concept
of the efficient frontier, the line along which all optimal portfolios lie
'''

# Pick a few Highly traded assets
syms = ['AAPL', 'MSFT','YHOO','DB','GLD']
noa = len(syms)
data = prices(assets=symbols(syms), start='2010-04-01', end='2014-09-12')
data.columns = syms
(data / data.ix[0]).plot()
plt.title('Normalized Return')

rets = np.log(data / data.shift(1)) # logarithmic returns

# Use Monte-carlo simulation to generate random portfolio weights on a large scale
prets = []
pvols = []
for p in range(2500):
    weights = np.random.random(noa) # random weights on stock purchases
    weights /= np.sum(weights)
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
        Sharpe ratio for rf = 0
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
    # function to minimize absolute variance
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
bnds = tuple((0,1) for x in weights)
trets = np.linspace(0., np.max(prets), 50)
tvols = []
for tret in trets:
    cons = ({'type' : 'eq', 'fun' : lambda x: statistics(x)[0] - tret},
            {'type' : 'eq', 'fun' : lambda x: np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(8,6))
plt.scatter(pvols, prets, c=prets / pvols, marker='o',cmap='Greens') # random compositions
plt.scatter(tvols, trets, c=trets/tvols, marker='x',cmap='Greens') # efficient frontier
plt.plot(tvols, trets, 'r')
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0], 'r*', markersize=15.0) # highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0], 'k*', markersize=15.0) # minimum variance
plt.grid()
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.legend(['Efficient frontier','Max Sharpe Ratio','Minimum Variance'], loc=3)
plt.title('MPT Demonstration: $E_v$ vs $\sigma$')

# now a demonstration of the capital market line
# using only portfolios on the efficient frontier, starting from that of the minimum variance
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]

# use these for a spline interpolation
tck = sci.splrep(evols, erets)
def f(x):
    '''Efficient frontier function (spline approx)'''
    return sci.splev(x, tck, der=0)
def df(x):
    '''First Deriv of efficient frontier func'''
    return sci.splev(x, tck, der=1)

def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

# use fsolve equation to determine solution to equation set,
# using well-chosen initial guesses
opt = sco.fsolve(equations, [0.01, 0.5, .15])
print(opt)
print(np.round(equations(opt),6))

cons = ({'type' : 'eq', 'fun' : lambda x: statistics(x)[0] - f(opt[2])},
        {'type' : 'eq', 'fun' : lambda x: np.sum(x) - 1})
res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP', bounds=bnds, constraints=cons)
opt_weights = res['x'].round(3)
print(opt_weights)
print(statistics(opt_weights))

plt.figure(figsize=(8,6))
plt.scatter(pvols, prets, c=(prets - 0.01)/pvols, marker='o', cmap='Greens') # random compositions
plt.plot(evols, erets, 'g', lw=4.)
cx = np.linspace(0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
plt.plot(opt[2],f(opt[2]), 'r*', markersize=15.)
plt.grid()
plt.axhline(0, color='k', ls='--', lw=2.)
plt.axvline(0, color='k', ls='--', lw=2.)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

plt.show()
