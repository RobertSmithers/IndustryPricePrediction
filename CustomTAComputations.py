import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import scatter_matrix
import yfinance as yf
import talib
#import talib
#%matplotlib inline

start = '2017-02-19'
end = '2022-2-19'
sp500 = yf.download('^GSPC', start, end)

# Moving Averages https://www.analyticsvidhya.com/blog/2021/07/stock-prices-analysis-with-python/#h2_5
def MA(data_frame, days):
	name = 'MA'+str(days)
	data_frame[name] = data_frame['close'].rolling(days).mean()
	return data_frame

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
#sp500['RSI'] = talib.RSI(reversed_df['Close'], 14)

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

# SP500
sp500_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/SP500_7yr_daily.csv')
sp500_frame = pd.DataFrame(sp500_data, columns = ['ticker', 'descr', 'date', 'close', 'retx'])
MA(sp500_frame, 50)
MA(sp500_frame, 200)

# XLB
xlb_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLB_7yr_daily.csv')
xlb_frame = pd.DataFrame(xlb_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xlb_frame, 50)
MA(xlb_frame, 200)

# XLC
xlc_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLC_7yr_daily.csv')
xlc_frame = pd.DataFrame(xlc_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xlc_frame, 50)
MA(xlc_frame, 200)

# XLE
xle_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLE_7yr_daily.csv')
xle_frame = pd.DataFrame(xle_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xle_frame, 50)
MA(xle_frame, 200)

# XLF
xlf_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLF_7yr_daily.csv')
xlf_frame = pd.DataFrame(xlf_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xlf_frame, 50)
MA(xlf_frame, 200)

# XLI
xli_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLI_7yr_daily.csv')
xli_frame = pd.DataFrame(xli_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xli_frame, 50)
MA(xli_frame, 200)

# XLK
xlk_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLK_7yr_daily.csv')
xlk_frame = pd.DataFrame(xlk_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xlk_frame, 50)
MA(xlk_frame, 200)

# XLP
xlp_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLP_7yr_daily.csv')
xlp_frame = pd.DataFrame(xlp_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xlp_frame, 50)
MA(xlp_frame, 200)

# XLRE
xlre_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLRE_7yr_daily.csv')
xlre_frame = pd.DataFrame(xlre_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xlre_frame, 50)
MA(xlre_frame, 200)

# XLU
xlu_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLU_7yr_daily.csv')
xlu_frame = pd.DataFrame(xlu_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xlu_frame, 50)
MA(xlu_frame, 200)

# XLV
xlv_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLV_7yr_daily.csv')
xlv_frame = pd.DataFrame(xlv_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xlv_frame, 50)
MA(xlv_frame, 200)

# XLY
xly_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLY_7yr_daily.csv')
xly_frame = pd.DataFrame(xly_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
MA(xly_frame, 50)
MA(xly_frame, 200)


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

""""
ax2.set_title('RSI of S&P 500, 2/19/2017 - 2/18/2022')
ax2.plot(sp500['RSI'], color = 'red', linewidth = 1.5, label = "RSI")
ax2.axhline(30, linestyle='--', linewidth=1, color='grey')
ax2.axhline(70, linestyle='--', linewidth=1, color='grey')
ax2.legend()
"""

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
