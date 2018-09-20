import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib as mpl

# Benchmark case, GBM

np.random.seed(1000)


def gen_paths(S0, r, sigma, T, M, I):
    '''
    Generates Monte Carlo paths for GBM

    :param S0: initial index value
    :param r: constant short rate
    :param sigma: constant volatility
    :param T: final time horizon
    :param M: number of time steps
    :param I: number of paths to simulate
    :return: simulated pathes given the parameters
    '''

    dt = float(T) / M
    paths = np.zeros((M+1, I), np.float64)
    paths[0] = S0
    for t in range(1, M+1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt)*rand)

    return paths

def print_statistics(array):
    '''
    Prints selected statistics
    :param array: object to generate statistics on
    '''
    sta = scs.describe(array)
    print("%14s %15s" % ('statistic', 'value'))
    print(30*"-")
    print("%14s %15.5f" % ('size', sta[0]))
    print("%14s %15.5f" % ('min', sta[1][0]))
    print("%14s %15.5f" % ('max', sta[1][1]))
    print("%14s %15.5f" % ('mean', sta[2]))
    print("%14s %15.5f" % ('std', np.sqrt(sta[3])))
    print("%14s %15.5f" % ('skew', sta[4]))
    print("%14s %15.5f" % ('kurtosis', sta[5]))

def normality_test(arr):
    '''
    Robust normality test based on skewness, kurtosis, and normality

    :param arr: obj to generate statistics on
    '''

    print("Skew of data set  %14.3f" % scs.skew(arr))
    print("Skew test p-value %14.3f" % scs.skewtest(arr)[1])
    print("Kurt of sata set  %14.3f" % scs.kurtosis(arr))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(arr)[1])

'''Driver code, commented out to use these mehtods in other scripts
S0 = 100.
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000

paths = gen_paths(S0, r, sigma, T, M, I)

# plot of first 10 paths
plt.figure()
plt.plot(paths[:, :10])
plt.grid()
plt.xlabel('time steps')
plt.ylabel('index level')

log_returns = np.log(paths[1:] / paths[0:-1])
print_statistics(log_returns.flatten())

# histogram graphically comparing normality with theoretical,
# very basic normality test
plt.figure()
plt.hist(log_returns.flatten(), bins=70, normed=True, label='frequency')
plt.grid()
plt.xlabel('log-return')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M)), 'r', lw=2.0, label='pdf')
#probability density function
plt.legend()

# QQ plot, compares theoretical values with actual
sm.qqplot(log_returns.flatten()[::500],line='s')
plt.grid()
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
ax1.hist(paths[-1], bins=30)
ax1.grid()
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.set_title('regular data')
ax2.hist(np.log(paths[-1]), bins=30)
ax2.grid()
ax2.set_xlabel('log index level')
ax2.set_title('log data')


print()
# showing the normal distribution of the data
print(normality_test(log_returns.flatten()))
print_statistics(paths[-1])
print_statistics(np.log(paths[-1]))


plt.show()'''