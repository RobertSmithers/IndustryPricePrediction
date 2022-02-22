import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import scatter_matrix
import yfinance as yf
import talib
#%matplotlib inline

start = '2017-02-19'
end = '2022-2-19'
sp500 = yf.download('^GSPC', start, end)

# Moving Averages https://www.analyticsvidhya.com/blog/2021/07/stock-prices-analysis-with-python/#h2_5
sp500['MA50'] = sp500['Close'].rolling(50).mean()
sp500['MA200'] = sp500['Close'].rolling(200).mean()

# RSI https://wire.insiderfinance.io/calculate-rsi-with-python-and-yahoo-finance-c8fb78b1c199
def RSI(data, window = 14, adjust = False):
    delta = data['Close'].diff(1).dropna()
    loss = delta.copy()
    gains = delta.copy()

    gains[gains < 0] = 0
    loss[loss > 0] = 0

    gain_ewm = gains.ewm(com = window - 1, adjust = adjust).mean()
    loss_ewm = abs(loss.ewm(com = window - 1, adjust = adjust).mean())

    RS = gain_ewm / loss_ewm
    RSI = 100 - 100/ (1 + RS)

    return RSI
reversed_df = sp500.iloc[::-1]
sp500['RSI'] = talib.RSI(reversed_df['Close'], 14)

locator = mdates.MonthLocator(interval = 3)
fmt = mdates.DateFormatter('%b')

#KDJ https://github.com/Abhay64/KDJ-Indicator
array_close = np.array(sp500['Close'])
array_high = np.array(sp500['High'])
array_low = np.array(sp500['Low'])

z = 0
y = 0
highest = 0
lowest = 0
kperiods = 13 #kperiods are 14 array start from 0 index
array_highest = []
array_lowest = []

for i in range(0, array_high.size - kperiods):
	highest = array_high[y]
	for j in range(0, kperiods):
		if(highest < array_high[y + 1]):
			highest = array_high[y + 1]
		y = y + 1
	# creating list highest of k periods
	array_highest.append(highest)
	y = y - (kperiods - 1)

for i in range(0, array_low.size - kperiods):
	lowest = array_low[z]
	for j in range(0, kperiods):
		if(lowest > array_low[z + 1]):
			lowest = array_low[z + 1]
		z = z + 1
	# creating list lowest of k periods
	array_lowest.append(lowest)
  # skip one from starting after each iteration
	z = z - (kperiods - 1)

#KDJ (K line, D line, J line)
    # K line
Kvalue = []
for i in range(kperiods,array_close.size):
   k = ((array_close[i] - array_lowest[i - kperiods]) * 100 / (array_highest[i - kperiods] - array_lowest[i - kperiods]))
   Kvalue.append(k)
sp500['K'] = pd.Series(Kvalue)

    # D line
x = 0
# dperiods for calculate d values
dperiods = 3
Dvalue = [None, None]
mean = 0
for i in range(0, len(Kvalue) - dperiods + 1):
	sum = 0
	for j in range(0, dperiods):
		sum = Kvalue[x] + sum
		x = x + 1
	mean = sum / dperiods
	# d values for %d line adding in the list Dvalue
	Dvalue.append(mean)
    # skip one from starting after each iteration
	x = x - (dperiods - 1)
sp500['D'] = pd.Series(Dvalue)

    # J line
Jvalue = [None, None]
for i in range(0, len(Dvalue) - dperiods + 1):
	j = (Dvalue[i + 2] * 3) - (Kvalue[i + 2] * 2)
	# j values for %j line
	Jvalue.append(j)
sp500['J'] = pd.Series(Jvalue)

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan = 3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan = 3)
ax3 = plt.subplot2grid((3, 3), (2, 0), colspan = 3)

ax1.set_title('Close Stock Prices of S&P 500, 2/19/2017 - 2/18/2022')
ax1.plot(sp500['Close'], linewidth = 1.5, label = "Close Price")
ax1.plot(sp500['MA50'], linewidth = 1, label = "Moving Average 50 Days")
ax1.plot(sp500['MA200'], linewidth = 1, label = "Moving Average 200 Days")
ax1.legend()
#X1 = ax1.xaxis
#X1.set_major_locator(locator)
#X1.set_major_formatter(fmt)

ax2.set_title('RSI of S&P 500, 2/19/2017 - 2/18/2022')
ax2.plot(sp500['RSI'], color = 'red', linewidth = 1.5, label = "RSI")
ax2.axhline(30, linestyle='--', linewidth=1, color='grey')
ax2.axhline(70, linestyle='--', linewidth=1, color='grey')
ax1.legend()

ax3.set_title('KDJ Indicator')
ax3.plot(Kvalue, linewidth = 0.2, color = 'red', label = "%K line")
ax3.plot(Dvalue, linewidth = 0.2, color = 'blue', label = "%D line")
ax3.plot(Jvalue, linewidth = 0.2, color = 'green', label = "%J line")
ax3.legend()
# Missing correct dates
# Needs to implement KDJ to original dataframe

"""
plt.figure(figsize=(25,15), dpi=50, facecolor='w', edgecolor='k')
ax = plt.gca() 
plt.plot(Kvalue,color='red',label = '%K line')
plt.plot(Dvalue,color='blue',label = '%D line')
plt.plot(Jvalue,color='green',label = '%J line')
"""
plt.show()
