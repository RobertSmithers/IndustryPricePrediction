import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import scatter_matrix
import ta
#https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#momentum-indicators

def MA(df, days):
	name = 'MA'+str(days)
	df[name] = df['close'].rolling(days).mean()
	return df

def AddIndicators(df):
    #df = add_all_ta_features(df, open="###", high="high", low="low", close="close", volume="vol")   Missing open
    ### Momentum ###
    # Moving Averages (50 days and 200 days)
    MA(df, 50)
    MA(df, 200)
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window = 14)
    # Stochastic Oscillator
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window = 14).stoch()
    # Rate of Change
    df['roc'] = ta.momentum.roc(df['close'], window = 14)
    # TSI - True strength index
    df['tsi'] = ta.momentum.tsi(df['close'], window_slow = 25, window_fast = 13)    

    ### Volume ###
    # Accumulation / Distribution Index (ADI)
    df['adi'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['vol']).acc_dist_index()
    # Chaikin Money Flow
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['vol']).chaikin_money_flow()
    # Ease of Movement
    df['eom'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['vol'], window = 14).ease_of_movement()
    # Money Flow Index
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['vol'], window = 14).money_flow_index()
    # Negative Volume Index (NVI)
    df['nvi'] = ta.volume.NegativeVolumeIndexIndicator(df['close'], df['vol']).negative_volume_index()
    # On-balance volume (OBV)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['vol']).on_balance_volume()
    # Volume-price trend (VPT)
    df['vpt'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['vol']).volume_price_trend()
    # Volume Weighted Average Price (VWAP)
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['vol'], window = 14).volume_weighted_average_price()

    ### Volatility ###
    # Bollinger Bands
    df['blband'] = ta.volatility.BollingerBands(df['close'], window = 14).bollinger_lband()
    df['bhband'] = ta.volatility.BollingerBands(df['close'], window = 14).bollinger_hband()
    # Average True Range (ATR)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window = 14).average_true_range()
    # Donchian Channel
    df['dlband'] = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window = 14).donchian_channel_lband()
    df['dhband'] = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window = 14).donchian_channel_hband()
    # Keltner Channels
    df['klband'] = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window = 14).keltner_channel_lband()
    df['khband'] = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window = 14).keltner_channel_hband()

    ### Trend ###
    # Average Directional Movement Index (ADX)
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window = 14).adx()
    # Commodity Channel Index (CCI)
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window = 14).cci()
    # Exponential Moving Average (EMA)
    df['ema'] = ta.trend.EMAIndicator(df['close'], window = 14).ema_indicator()
    # KST Oscillator (KST Signal)
    df['kst'] = ta.trend.KSTIndicator(df['close']).kst()
    # Moving Average Convergence Divergence (MACD)
    df['macd'] = ta.trend.MACD(df['close']).macd()
    return df

# SP500
sp500_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/SP500_7yr_daily.csv')
sp500_frame = pd.DataFrame(sp500_data, columns = ['ticker', 'descr', 'date', 'close', 'retx'])

# XLB
xlb_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLB_7yr_daily.csv')
xlb_frame = pd.DataFrame(xlb_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])
AddIndicators(xlb_frame)
print(xlb_frame)

# XLC
xlc_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLC_7yr_daily.csv')
xlc_frame = pd.DataFrame(xlc_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLE
xle_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLE_7yr_daily.csv')
xle_frame = pd.DataFrame(xle_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLF
xlf_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLF_7yr_daily.csv')
xlf_frame = pd.DataFrame(xlf_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLI
xli_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLI_7yr_daily.csv')
xli_frame = pd.DataFrame(xli_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLK
xlk_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLK_7yr_daily.csv')
xlk_frame = pd.DataFrame(xlk_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLP
xlp_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLP_7yr_daily.csv')
xlp_frame = pd.DataFrame(xlp_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLRE
xlre_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLRE_7yr_daily.csv')
xlre_frame = pd.DataFrame(xlre_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLU
xlu_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLU_7yr_daily.csv')
xlu_frame = pd.DataFrame(xlu_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLV
xlv_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLV_7yr_daily.csv')
xlv_frame = pd.DataFrame(xlv_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])

# XLY
xly_data = pd.read_csv('~/Documents/Github/IndustryPricePrediction/data/sectors/XLY_7yr_daily.csv')
xly_frame = pd.DataFrame(xly_data, columns = ['ticker', 'descr', 'date', 'low', 'high', 'close', 'vol', 'ret', 'bid', 'ask', 'retx'])