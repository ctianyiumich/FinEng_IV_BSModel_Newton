import scipy as sp
import pandas as pd
import numpy as np
import yfinance as yf
from math import log, sqrt, pi, exp
from scipy.stats import norm
from pprint import pprint
import time
from datetime import datetime

#BS Model: Compute d_1
def d_1(S, K, sigma, t, r):
    """
    param S: current stock price
    param K: strike price
    param sigma: volatility (cannot be 0)
    param t: annualized time(days) until maturity (cannot be 0, or the maturity date cannot be the exercise date)
    param r: annualized interest rate
    """
    return (np.log(S/K) + (r + 0.5*np.power(sigma, 2)*t))/(sigma*np.power(t, 0.5))

#BS Model: Compute d_2
def d_2(S, K, sigma, t, r):
    """
    param S: current stock price
    param K: strike price
    param sigma: volatility (cannot be 0)
    param t: annualized time(days) until maturity (cannot be 0, or the maturity date cannot be the exercise date)
    param r: annuaized interest rate
    """
    return d_1(S, K, sigma, t, r) - sigma * np.power(t, 0.5)

#BS Model: Price claim with Black-Scholes formula, and minus its actual price from Yahoo Finance
def BS_func(S, K, sigma, t, r, Pi):
    """
    param S: current stock price
    param K: strike price
    param sigma: volatility (cannot be 0)
    param t: annualized time until maturity (cannot be 0, or the maturity date cannot be the exercise date)
    param r: annualized interest rate
    param Pi: actual price of the claim from Yahoo Finance
    descritption: Formula is used to price simple European options, without considering dividends or consumption.
    """
    #Apply Black-Scholes formula
    BS_price = sp.stats.norm.cdf(d_1(S, K, sigma, t, r))*S - sp.stats.norm.cdf(d_2(S, K, sigma, t, r)) * K * np.exp(-r*t)
    #Return its difference from the observed price in Yahoo Finance
    return  BS_price - Pi

#BS Model: Compute Vega/Kappa
def Vega(S, K, sigma, t, r):
    """
    param S: current stock price
    param K: strike price
    param sigma: volatility (cannot be 0)
    param t: annualized time until maturity (cannot be 0, or the maturity date cannot be the exercise date)
    param r: annualized interest rate
    description: sensitivity of the claim price to a change in volatility, defined by the first derivative of Black Scholes price (BS_func::BS_price) respect to volatility (sigma).
    """
    #Apply formula
    vega = S*sp.stats.norm.pdf(d_1(S, K, sigma, t, r))*np.power(t, 0.5)
    return vega

#Root approximation using Newton's method: single step
def NR_Single(S, K, sigma, t, r, Pi):
    """
    param S: current stock price
    param K: strike price
    param sigma: volatility (cannot be 0)
    param t: annualized time until maturity (cannot be 0, or the maturity date cannot be the exercise date)
    param r: annualized interest rate
    param Pi: actual price of the claim from Yahoo Finance
    """
    return sigma - BS_func(S, K, sigma, t, r, Pi)/Vega(S, K, sigma, t, r)

#Root approximation: iterate NR_Single
def NR_Iteration(S, K, x_0, t, r, Pi):
    """
    param S: current stock price
    param K: strike price
    param x_0: initial guess
    param t: annualized time until maturity (cannot be 0, or the maturity date cannot be the exercise date)
    param r: annualized interest rate
    param Pi: actual price of the claim from Yahoo Finance
    """
    x_a = x_0 #assign initial guess as a local vairable
    x_b = NR_Single(S, K, x_a, t, r, Pi) #operate a single step using Newton's method to approximate the real root
    #Sigma cannot be 0 to make the BS model hold, but reaching zero is possible in the process of root approximation.
    #Once the Newton's method leads us to 0, we restart the algorithm from another close point.
    if x_b == 0:
        x_b = 0.000001
    #Iterate Newton's method
    i=0 #Iteration count
    precision = 1e-6 #Error tolerance
    while (abs(x_b-x_a) > precision) and (i<500): #End the iteration once we reach the capped error tolerance (resutls are no longer volatile) or reach the max itermation to prevent endless loops.
        i += 1
        x_a = x_b
        if Vega(S, K, x_a, t, r) == 0:
            x_a += 0.1
        x_b = NR_Single(S, K, x_b, t, r, Pi)
        #print("d_1:", d_1(S, K, x_b, t, r)) #Check if d_1 is abnormal
        #print([round(x_a, 6), round(x_b, 6), abs(x_b-x_a)]) #Check root guesses by each step, [root guess input, root guess output, difference(expected to be low enought)]
    return x_b

#Company Ticker Symbol
ticker = 'GOOG' #Take Google as an example
#S: Stock Price
S = yf.Ticker(ticker).info['currentPrice']
#r: Interest Rate
r = 0.025

#T: Expiration Date
exp_date_initial_str = yf.Ticker('GOOG').options[0]# Latest expiration date, in format of string
exp_date_initial = datetime.strptime(exp_date_initial_str, '%Y-%m-%d').date()# Convert to datetime.date
print(exp_date_initial)
#Dataframe of calls
call_info = yf.Ticker(ticker).option_chain(exp_date_initial_str).calls
call_info['expirationDate'] = exp_date_initial# Append to call dataframe

#K: Strike Price
K_call = call_info['strike']
#sigma_y: Implied Volatility from Yahoo Finance
sigma_y_call = call_info['impliedVolatility']
#t: Last Trade Date
trade_date_call = call_info['lastTradeDate']# In format of datetime
trade_date_call = trade_date_call.dt.date# Convert to datetime.date
#T-t: Effective Duration
T_t_call = exp_date_initial - trade_date_call
T_t_call = T_t_call.dt.days/252# Convert to float
call_info['effectiveDuration'] = T_t_call
#Pi: Option Price
Pi_call = call_info['lastPrice']

#Assembly into a dataframe
df = pd.DataFrame({ 'S': [S for i in range(len(K_call))],
                    'K': list(K_call),
                    'sigma': list(sigma_y_call),
                    't': list(T_t_call),
                    'r': [r for i in range(len(K_call))],
                    'Pi':list(Pi_call)})

#print(df) #Check dataframe
                   
#df = df[20:21] #Slice datafame for ease of debugging

sigma_sol_Series = []

for id in range(len(list(df.index))):
    row = df[id:id+1]
    #Ensure all variables are float
    S = float(row['S'])
    K = float(row['K'])
    sigma = float(row['sigma'])
    t = float(row['t'])
    r = float(row['r'])
    Pi = float(row['Pi'])
    #Approximate roots with Newton's method
    sigma_sol = NR_Iteration(S, K, 1, t, r, Pi)
    sigma_sol_Series.append(sigma_sol)
#Append calculated implied volatilities to the dataframe
df['IV'] = sigma_sol_Series

print(df)# Check the new data with implied volatilities