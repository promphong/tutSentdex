from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import talib
from sklearn.model_selection import train_test_split

class Strategy(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(self):
        raise NotImplementedError("Should implement generate_signals()!")
        
class Portfolio(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_positions(self):
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        raise NotImplementedError("Should implement backtest_portfolio()!")


# Custom Strategy Mixed indicators       
class CustomStrategy(Strategy):   
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars

    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.where((self.bars['RSI'] > 50) & \
               (self.bars['MA'] > self.bars['Close']) & \
               (self.bars['ADX'] > 20),1,0)
        return signals

# SMA Strategy momentum
class SMA_Strategy(Strategy):
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
        
    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.where((self.bars['Avg.Price'] > self.bars['SMA']),1,0)
        return signals

# EMA Strategy momentum
class EMA_Strategy(Strategy):
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
        
    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.where((self.bars['Avg.Price'] > self.bars['EMA']),1,0)
        return signals
      
# RSI Strategy momentum
class RSI_Strategy(Strategy):
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
    
    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.where((self.bars['RSI'] > 50) & \
                (self.bars['Open'].diff() > 0) & \
                (self.bars['RSI'] > self.bars['Avg.RSI']),1,0)
        return signals

# KAMA Strategy momentum
class KAMA_Strategy(Strategy):
    
  def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
      
  def generate_signals(self):
      signals = pd.DataFrame(index=self.bars.index)
      signals['signal'] = np.where((self.bars['Avg.Price'] > self.bars['KAMA']),1,0)
      return signals
    
# MACD Strategy momentum
class MACD_Strategy(Strategy):
  
  def __init__(self, symbol, bars):
      self.symbol = symbol
      self.bars = bars
      
  def generate_signals(self):
      signals = pd.DataFrame(index=self.bars.index)
      signals['signal'] = np.where((self.bars['MACD_Hist'] > 0) & \
             (self.bars['Open'].diff() > 0),1,0)
      return signals
    
# CCI Strategy momentum
class CCI_Strategy(Strategy):
  
  def __init__(self, symbol, bars):
      self.symbol = symbol
      self.bars = bars
      
  def generate_signals(self):
      signals = pd.DataFrame(index=self.bars.index)
      signals['signal'] = np.where((self.bars['CCI'] > self.bars['Avg.CCI']) & \
               (self.bars['Open'].diff() > 0),1,0)
      return signals
    
# Parabolic SAR Strategy (Trend Reversal)
class ParabolicSAR_Strategy(Strategy):
  
  def __init__(self, symbol, bars):
      self.symbol = symbol
      self.bars = bars
      
  def generate_signals(self):
      signals = pd.DataFrame(index=self.bars.index)
      signals['signal'] = np.where((self.bars['Avg.Price'] > self.bars['SAR']),1,0)
      return signals
    
# ADX Strategy (Trend Reversal)
class ADX_Strategy(Strategy):
  
  def __init__(self, symbol, bars):
      self.symbol = symbol
      self.bars = bars
      
  def generate_signals(self):
      signals = pd.DataFrame(index=self.bars.index)
      signals['signal'] = np.where((self.bars['ADX'] > 25) & \
               (self.bars['DI_Plus'] > self.bars['DI_Min']),1,0)
      return signals
    
class PortfolioExcute(Portfolio):
    
    def __init__(self, symbol, signals, bars, initial_capital=100000.0, pct_hedge = 1):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.pct_hedge = float(pct_hedge)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = (self.initial_capital* self.pct_hedge * self.signals)
        return positions
                    
    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.bars.index)        
        portfolio['Order_Excute'] = self.bars['Close'] - self.bars['Open'] #Logical avg.price(t-1) - close(t)
        portfolio['Signal'] = self.signals['signal'] 
        portfolio['PnL'] = (self.positions[self.symbol])/self.bars['Open'] * portfolio['Order_Excute']
        portfolio['Acc_Port'] = self.initial_capital + portfolio['PnL'].cumsum()
        portfolio['Return'] = portfolio['Acc_Port'].pct_change()
        return portfolio
    
    def performance_summary(self):
        cum_return = gen_cumulative_return(df,1, -1)
        max_drawdown = gen_drawdowns(df)
        sharpe_ratio = gen_sharpe_ratio(df) #use period 356 by default
        total_trade =  df[df['Signal']==1].count()['Signal']
        win_trade = df[df['PnL']>0].count()['PnL']
        pct_win_trade = win_trade/total_trade
        pct_loss_trade = 1-pct_win_trade
        port_value = df['Acc_Port'].iloc[-1]
        
#        stats = [("Accumulative Portfolio Value", "%0.2f" %port_value),
#                 ("Cumulative Return", "%0.4f%%" %(cum_return * 100.0)),
#                 ("Sharpe Ratio", "%0.2f" %sharpe_ratio),
#                 ("Max Drawdown", "%0.2f%%" %(max_drawdown * 100.0))] 
        

        return [port_value, cum_return, sharpe_ratio, 
                max_drawdown, total_trade, pct_win_trade, pct_loss_trade]
      
 
    
# Helping functions     
def gen_cumulative_return(df,index_sd, index_ed):
    date_diff = (df.index.values[index_ed] - df.index.values[index_sd])/np.timedelta64(1,'D')
    return (df['Acc_Port'].iloc[index_ed] / df['Acc_Port'].iloc[index_sd])**(365/date_diff)-1
    
def gen_drawdowns(df):
    hwm = [0]
    eq_idx = df.index
    drawdown = pd.Series(index=eq_idx)
    
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t-1], df['Acc_Port'].iloc[t])
        hwm.append(cur_hwm)
        drawdown[t] =  hwm[t] - df['Acc_Port'].iloc[t]
        
    return drawdown.max() / df['Acc_Port'].iloc[1]

def gen_sharpe_ratio(df, period=365):
    port_mean = np.mean(df['Return'])
    port_std = np.std(df['Return'])
    return np.sqrt(period) * (port_mean / port_std)
  
    
       
# ========================= Strategy Run generate ========================
if __name__ == '__main__':
    
    # ========================= Loop generate parameter =======================
    
    performance_list = {}
    
    for i in range(10,45):
#    for i in [p/100 for p in range(1,21)]:
      file_path = r'D:\Project files\USDTHB Historical Data.csv'
      df = pd.read_csv(file_path)
      df = df.set_index(pd.to_datetime(df['Date']))
      df = df.drop(['Date'], axis=1)
      df = df.dropna()
      symbol='USDTHB'
#      cci = pd.Series(talib.CCI(df['High'], df['Low'], df['Close'], i), name='CCI')
      macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=8, slowperiod=i, signalperiod=8)
      macd = pd.Series(macd, name='MACD')
      macdsignal = pd.Series(macdsignal, name='MACD_Signal')
      macdhist = pd.Series(macdhist, name='MACD_Hist')
#      sma = pd.Series(talib.SMA(df['Close'], i), name='SMA')
#      sar = pd.Series(talib.SAR(df['High'], df['Low'], acceleration=i, maximum=0.2), name='SAR')
#      ema = pd.Series(talib.EMA(df['Close'], i), name='EMA')
#      rsi = pd.Series(talib.RSI(df['Close'], i), name='RSI')
#      kama = pd.Series(talib.KAMA(df['Close'], i), name='KAMA')
#      adx = pd.Series(talib.ADX(df['High'], df['Low'], df['Close'], i), name='ADX')
#      di_min = pd.Series(talib.MINUS_DI(df['High'], df['Low'], df['Close'], i), name='DI_Min')
#      di_plus = pd.Series(talib.PLUS_DI(df['High'], df['Low'], df['Close'], i), name='DI_Plus')
      avg_price = pd.Series((df['Close']*0.7)+(df['Open']*0.1)+(df['High']*0.1)+(df['Low']*0.1), name='Avg.Price').shift(1)
      
      df = df.join(macd)
      df = df.join(macdsignal)
      df = df.join(macdhist)
#      df = df.join(di_plus)
#      df = df.join(di_min)
      
      # Average indicator series
#      avg_cci = pd.Series(talib.MA(df['CCI'], 10), name='Avg.CCI')
#      avg_rsi = pd.Series(talib.MA(df['RSI'], 10), name='Avg.RSI')
      
#      df = df.join(avg_rsi)
      df = df.join(avg_price)

      strategy = MACD_Strategy(symbol, df)
      signals = strategy.generate_signals()
      
      port = PortfolioExcute(symbol, signals, df, initial_capital=1000000)
      returns = port.backtest_portfolio()
      df = df.join(returns) # Write to report 
      performance = port.performance_summary()
      
      performance_list["MACD_8_{}_8".format(i)] = performance
      
      # gen dict to dataframe 
    performance_report = pd.DataFrame.from_dict(performance_list, 
                                                orient='index', 
                                                columns=['Accumulative Portfolio Value',
                                                        'Cumulative Return',
                                                        'Sharpe Ratio',
                                                        'Max Drawdown',
                                                        'Total Trade',
                                                        '%Win Trade',
                                                        '%Lose Trade'])
    
    writer = pd.ExcelWriter('performance_MACD_34.xls')
    performance_report.to_excel(writer, 'MACD')
    writer.save()
    
    #==================================== 3 parameter ==========================
    
#    for i in range(5,27):
#      for j in range(27,50):
#        for k in range(5, 11):
#          file_path = r'D:\Project files\USDTHB Historical Data.csv'
#          df = pd.read_csv(file_path)
#          df = df.set_index(pd.to_datetime(df['Date']))
#          df = df.drop(['Date'], axis=1)
#          #df = pd.read_csv(file_path)
#          df = df.dropna()
#          symbol='USDTHB'
#    #      kama = pd.Series(talib.KAMA(df['Close'], i), name='KAMA')
#          macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=i, slowperiod=j, signalperiod=k)
#          macd = pd.Series(macd, name='MACD')
#          macdsignal = pd.Series(macdsignal, name='MACD_Signal')
#          macdhist = pd.Series(macdhist, name='MACD_Hist')
#          avg_price = pd.Series((df['Close']*0.7)+(df['Open']*0.1)+(df['High']*0.1)+(df['Low']*0.1), name='Avg.Price').shift(1)
#          
#          df = df.join(macdsignal)
#          df = df.join(macdhist)
#          df = df.join(avg_price)
#    
#          strategy = MACD_Strategy(symbol, df)
#          signals = strategy.generate_signals()
#          
#          port = PortfolioExcute(symbol, signals, df, initial_capital=1000000)
#          returns = port.backtest_portfolio()
#          df = df.join(returns) # Write to report 
#          performance = port.performance_summary()
#          
#          performance_list["MACD_{}_{}_{}".format(i,j,k)] = performance
#      
#      # gen dict to dataframe 
#    performance_report = pd.DataFrame.from_dict(performance_list, 
#                                                orient='index', 
#                                                columns=['Accumulative Portfolio Value',
#                                                        'Cumulative Return',
#                                                        'Sharpe Ratio',
#                                                        'Max Drawdown'])
#    
#    writer = pd.ExcelWriter('performance_MACD.xls')
#    performance_report.to_excel(writer, 'MACD')
#    writer.save()
  
    
    # ========================= Loop generate parameter =======================
   
    # ======================== Gen Output report detail =======================
#    file_path = r'D:\Project files\USDTHB Historical Data.csv'
#    df = pd.read_csv(file_path)
#    df = df.set_index(pd.to_datetime(df['Date']))
#    df = df.drop(['Date'], axis=1)
#    df = df.dropna()
#    symbol='USDTHB'
#    
#    ma = pd.Series(talib.MA(df['Close'], 20), name='MA')
#    rsi = pd.Series(talib.RSI(df['Close'], 20), name='RSI')
#    adx = pd.Series(talib.ADX(df['High'], df['Low'], df['Close'], 14), name='ADX')
#    kama = pd.Series(talib.KAMA(df['Close'], 20), name='KAMA')
#    cci = pd.Series(talib.CCI(df['High'], df['Low'], df['Close'], 14), name='CCI')
#    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
#    macd = pd.Series(macd, name='MACD')
#    macdsignal = pd.Series(macdsignal, name='MACD_Signal')
#    macdhist = pd.Series(macdhist, name='MACD_Hist')
#    avg_price = pd.Series((df['Close']*0.7)+(df['Open']*0.1)+(df['High']*0.1)+(df['Low']*0.1), name='Avg.Price').shift(1) #for enter order 
##    price_diff = pd.Series(df['Open'].diff(),name='PriceDiff')
#    
#    indicators = [avg_price,ma, rsi, adx, kama, cci, macd, macdsignal, macdhist]
#    
#    for i in indicators:
#        df = df.join(i)
#    
#    # another inditor
#    avg_cci = pd.Series(talib.MA(df['CCI'], 20), name='Avg.CCI')
#    df = df.join(avg_cci)
#    
###    # ====================== Use class generate ===============================
#    strategy = CCI_Strategy(symbol, df)
#    signals = strategy.generate_signals()
#    
#    ma_port = PortfolioExcute(symbol, signals, df, initial_capital=1000000)
#    returns = ma_port.backtest_portfolio()
#    df = df.join(returns) # Write to report 
#    performance = ma_port.performance_summary()
#    
#    
##     # ====================== Use class generate ==============================
#    
#    
#    writer = pd.ExcelWriter('output.xls')
#    df.to_excel(writer, 'Sheet1')
#    writer.save()

    # ======================== Gen Output report detail =======================    

#     ====================== Hands on generate signal =========================
#    def signal_ma(indi_1, close):
#        if close > indi_1:
#            return 0
#        else:
#            return 1
    
#    signal_func = np.vectorize(signal_ma)
#    signal = pd.Series(signal_func(df['MA'], df['Avg.Price']),name='Signal',index=df.index)
#
#    
#    df = df.join(signal)
#    
#    # Test use self generate signal 
#    ma_port = MarketTestPortfolio(symbol, df, initial_capital=1000000, trading_sum=100)
#    test = ma_port.backtest_portfolio()
#    print(test.tail(15))

#     ====================== Hands on generate signal =========================
    
