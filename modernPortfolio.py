from pandas_datareader import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests_cache
from mlfinlab.portfolio_optimization.modern_portfolio_theory import CriticalLineAlgorithm

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

expire_after = datetime.timedelta(days=3)
session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)


DATE = 'Date'
HIGH = 'High'
LOW = 'Low'
OPEN = 'Open'
CLOSE = 'Close'
VOLUMN = 'Volume'
ADJ_CLOSE = 'Adj Close'
CODE = 'Code'

NUM_PORTFOLIOS = 5000
resoure = 'yahoo'
codes = ['ANSS','ATVI','AMD']
start_date = '2019/01/01'
end_date = '2019/12/31'

frames = []
for code in codes:
    r = data.DataReader(code, resoure, start=start_date, end=end_date, session=session)
    r[CODE] = code
    r = r[[ADJ_CLOSE, CODE]]
    frames.append(r)
    print(code)

total_prices = pd.concat(frames)
table = total_prices.pivot(columns = CODE, values= ADJ_CLOSE)

# Sampling

port_returns = []
port_volatility = []
stock_weights = []
sharpe_ratio = []

num_assets = len(codes)
num_portfolios = NUM_PORTFOLIOS

returns_daily_all = table.pct_change().dropna(how="all")
returns_daily = returns_daily_all.mean()
# returns_annual = returns_daily.mean()
print(returns_daily_all.std())
cov_daily = returns_daily_all.cov()
# cov_annual = cov_daily
# print(returns_annual)

avg_returns = expected_returns.mean_historical_return(table)
cov_mat = risk_models.sample_cov(table)
ef = EfficientFrontier(avg_returns, cov_mat)

# CLA
# cla = CriticalLineAlgorithm()
# cla.allocate(expected_asset_returns=returns_daily,covariance_matrix=cov_daily, solution='efficient_frontier', asset_names=codes)
# cla_weights = cla.weights
# means, sigma = cla.efficient_frontier_means, cla.efficient_frontier_sigma
# plt.plot(sigma, means)
# plt.show()

cla = CriticalLineAlgorithm()
cla.allocate(asset_prices=table, asset_names=codes, solution='min_volatility')
cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1)
weights = cla_weights.loc[0]
avg_returns = expected_returns.mean_historical_return(table)
np.dot(weights,avg_returns)
print('min std', np.sqrt(cla.min_var))

cla = CriticalLineAlgorithm()
cla.allocate(expected_asset_returns=returns_daily,covariance_matrix=cov_daily, asset_names=codes, solution='max_sharpe')
cla_weights = cla.weights.sort_values(by=0, ascending=False, axis=1)
weights = cla_weights.loc[0]
volatility = np.sqrt(np.dot(weights.T, np.dot(cov_daily, weights)))
returns = np.dot(returns_daily, weights)
max_sharpe_value = cla.max_sharpe 

# # Sampling
# for single_portfolio in range(num_portfolios):
#     # sample weight
#     weights = np.random.random(num_assets)
#     weights = weights/np.sum(weights)
    
#     # return & standard deviation
#     returns = np.dot(weights, returns_daily)
#     volatility = np.sqrt(np.dot(weights.T, np.dot(cov_daily, weights)))
    
#     sharpe = returns / volatility
    
#     # add to sample list
#     sharpe_ratio.append(sharpe)
#     port_returns.append(returns)
#     port_volatility.append(volatility)
#     stock_weights.append(weights)
    
# portfolio ={'Returns' : port_returns,
#            'Volatility' : port_volatility,
#            'Sharpe Ratio': sharpe_ratio}

# # print(portfolio)

# for counter,symbol in enumerate(codes):
#     portfolio[symbol + ' weight'] = [weight[counter] for weight in stock_weights]

# df = pd.DataFrame(portfolio)

# column_order = ['Returns', 'Volatility'] + [stock+' weight' for stock in codes]

# df = df[column_order]

# # df.head()

# plt.style.use('seaborn')
# df.plot.scatter( x= 'Volatility', y = 'Returns', figsize= (10,6), grid = True)
# plt.xlabel('Volatility(Std. Deviation)')
# plt.ylabel('Expected Returns')
# plt.title('Efficient Frontier')

# plt.show()


# # stock_prices = pd.read_csv('./data.csv', parse_dates=True, index_col='Date')

# # from mlfinlab.portfolio_optimization.modern_portfolio_theory import CriticalLineAlgorithm
# # cla = CriticalLineAlgorithm()
# # cla.allocate(asset_prices=table, solution='efficient_frontier')
# # cla_weights = cla.weights
# # means, sigma = cla.efficient_frontier_means, cla.efficient_frontier_sigma
# # plt.plot(sigma, means)
# # plt.show()
# # # from mlfinlab.portfolio_optimization.modern_portfolio_theory import MeanVarianceOptimisation

# # solution = [0.472292, 0.25465, 0.182976, 0.071653, 0.018429, 0, 0, 0]
# # # var = 0.01040912

# # cla = CriticalLineAlgorithm()
# # cla.allocate(expected_asset_returns=returns_daily,covariance_matrix=returns_daily_all.cov(), solution='efficient_frontier', asset_names = codes)
# # cla_weights = cla.weights
# # means, sigma = cla.efficient_frontier_means, cla.efficient_frontier_sigma

import json
with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)
with open('data.txt') as json_file:
    data = json.load(json_file)