from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
#import matplotlib.pylab as plt
import talib
#from sklearn.model_selection import train_test_split

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
               (self.bars['MA'] > self.bars['Avg.Price']) & \
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
                (self.bars['Avg.Price'].diff() > 0) & \
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
             (self.bars['Avg.Price'].diff() > 0) & \
             (self.bars['MACD'] > self.bars['MACD_Signal']),1,0)
      #macd, macdsignal, macdhist 
      return signals
    
# CCI Strategy momentum
class CCI_Strategy(Strategy):
  
  def __init__(self, symbol, bars):
      self.symbol = symbol
      self.bars = bars
      
  def generate_signals(self):
      signals = pd.DataFrame(index=self.bars.index)
      signals['signal'] = np.where((self.bars['CCI'] > self.bars['Avg.CCI']) & \
               (self.bars['Avg.Price'].diff() > 0),1,0)
      return signals
    
# Parabolic SAR Strategy (Trend Reversal)
class SAR_Strategy(Strategy):
  
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
  
# Gen Actual Signal
class Actual_price(Strategy):
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
        
    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.where(self.bars['Last'], 1, 0)
        return signals
    
    
class PortfolioExcute(Portfolio):
    
    def __init__(self, symbol, signals, bars, initial_capital=100000.0, pct_hedge = 1):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals.shift(1)
        self.initial_capital = float(initial_capital)
        self.pct_hedge = float(pct_hedge)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = (self.initial_capital* self.pct_hedge * self.signals)
        return positions
                    
    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.bars.index)        
#        portfolio['Order_Excute'] = (self.bars['Last'] / self.bars['Last'].shift(-1))-1
        portfolio['Order_Excute'] = self.bars['Last'].pct_change()
        portfolio['Signal'] = self.signals['signal']
        portfolio['PnL'] = (self.positions[self.symbol]) * portfolio['Order_Excute']
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
#    print("Date", df.index.values[index_ed], "-", df.index.values[index_sd], ":", date_diff)
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
    workbooks = {}
    
    for f in range(2015, 2019):
    
        for i in range(27,52):
#        for i in [p/100 for p in range(1,21)]:
          file_path = r'D:\Projects\Data\USDTHB Historical Data.csv'  
    #      file_path = r'D:\Project files\USDTHB Historical Data.csv'
          df = pd.read_csv(file_path)
          df = df.set_index(pd.to_datetime(df['Date'], format='%m/%d/%Y'))
          df = df.drop(['Date'], axis=1)
          symbol='USDTHB'
          
          # Filter by year
          df = df[df.index.year == f]
          
          
#          sma = pd.Series(talib.SMA(df['Last'], i), name='SMA')
#          ema = pd.Series(talib.EMA(df['Last'], i), name='EMA')
#          kama = pd.Series(talib.KAMA(df['Last'], i), name='KAMA')
#          cci = pd.Series(talib.CCI(df['High'], df['Low'], df['Last'], i), name='CCI')
#          rsi = pd.Series(talib.RSI(df['Last'], i), name='RSI')
#          sar = pd.Series(talib.SAR(df['High'], df['Low'], acceleration=i, maximum=0.2), name='SAR')
#          adx = pd.Series(talib.ADX(df['High'], df['Low'], df['Last'], i), name='ADX')
#          di_min = pd.Series(talib.MINUS_DI(df['High'], df['Low'], df['Last'], i), name='DI_Min')
#          di_plus = pd.Series(talib.PLUS_DI(df['High'], df['Low'], df['Last'], i), name='DI_Plus')
          macd, macdsignal, macdhist = talib.MACD(df['Last'], fastperiod=24, slowperiod=i, signalperiod=18)
          macd = pd.Series(macd, name='MACD')
          macdsignal = pd.Series(macdsignal, name='MACD_Signal')
          macdhist = pd.Series(macdhist, name='MACD_Hist')
          avg_price = pd.Series((df['Last']*0.7)+(df['Open']*0.1)+(df['High']*0.1)+(df['Low']*0.1),
                                name='Avg.Price')
          
#          df = df.join(adx)
          df = df.join(macd)
          df = df.join(macdsignal)
          df = df.join(macdhist)
#          df = df.join(di_plus)
#          df = df.join(di_min)
          
          # Average indicator series
#          avg_cci = pd.Series(talib.MA(df['CCI'], 10), name='Avg.CCI')
#          avg_rsi = pd.Series(talib.MA(df['RSI'], 10), name='Avg.RSI')
          
#          df = df.join(avg_rsi)
          df = df.join(avg_price)
    
          strategy = MACD_Strategy(symbol, df)
          signals = strategy.generate_signals()
          
          port = PortfolioExcute(symbol, signals, df, initial_capital=1000000)
          returns = port.backtest_portfolio()
          df = df.join(returns) # Write to report 
          performance = port.performance_summary()
          
          performance_list["MACD_24_{}_18".format(i)] = performance
          
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
          
          workbooks["MACD_{}".format(f)] = performance_report
          
    writer = pd.ExcelWriter('performance_MACD_slow.xls')
    for ws, wb in workbooks.items():
        wb.to_excel(writer, sheet_name=ws)
    writer.save()
    
    # ========================= Loop generate parameter =======================
   
    # ======================== Gen Output report detail =======================
#    file_path = r'D:\Projects\Data\USDTHB Historical Data.csv'
#    df = pd.read_csv(file_path)
#    df = df.set_index(pd.to_datetime(df['Date'], format='%m/%d/%Y'))
#    df = df.drop(['Date'], axis=1)
#    symbol='USDTHB'
#    
#    # Filter by Date
#    df = df[df.index.year == 2018]
#    
#    sma = pd.Series(talib.SMA(df['Last'], 20), name='SMA')
#    ema = pd.Series(talib.EMA(df['Last'], 20), name='EMA')
#    rsi = pd.Series(talib.RSI(df['Last'], 20), name='RSI')
#    adx = pd.Series(talib.ADX(df['High'], df['Low'], df['Last'], 20), name='ADX')
#    di_min = pd.Series(talib.MINUS_DI(df['High'], df['Low'], df['Last'], 20), name='DI_Min')
#    di_plus = pd.Series(talib.PLUS_DI(df['High'], df['Low'], df['Last'], 20), name='DI_Plus')
#    kama = pd.Series(talib.KAMA(df['Last'], 20), name='KAMA')
#    cci = pd.Series(talib.CCI(df['High'], df['Low'], df['Last'], 14), name='CCI')
#    macd, macdsignal, macdhist = talib.MACD(df['Last'], fastperiod=12, slowperiod=26, signalperiod=9)
#    macd = pd.Series(macd, name='MACD')
#    macdsignal = pd.Series(macdsignal, name='MACD_Signal')
#    macdhist = pd.Series(macdhist, name='MACD_Hist')
#    avg_price = pd.Series((df['Last']*0.7)+(df['Open']*0.1)+(df['High']*0.1)+(df['Low']*0.1),
#                          name='Avg.Price').shift(1)#for enter order 
#
#    
#    indicators = [avg_price,sma, rsi, adx, kama, cci, macd, macdsignal, macdhist]
#    
#    for i in indicators:
#        df = df.join(i)
#    
#    #another inditor
#    avg_rsi = pd.Series(talib.MA(df['RSI'], 10), name='Avg.RSI')
#    df = df.join(avg_rsi)
#    
###    # ====================== Use class generate ===============================
#    strategy = SMA_Strategy(symbol, df)
#    signals = strategy.generate_signals()
#    
#    ma_port = PortfolioExcute(symbol, signals, df, initial_capital=1000000)
#    returns = ma_port.backtest_portfolio()
#    df = df.join(returns) # Write to report 
#    performance = ma_port.performance_summary()
##    print(performance)
#    
#    
###     # ====================== Use class generate ==============================
##    
##    
#    writer = pd.ExcelWriter('output.xls')
#    df.to_excel(writer, 'Sheet1')
#    writer.save()

    # ======================== Gen Output report detail =======================    

    