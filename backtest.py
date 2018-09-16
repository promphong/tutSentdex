from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
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

        
class CustomStrategy(Strategy):   
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars

    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
#        signals['signal'] = np.vectorize(self.signal_ma(self.bars['MA'], self.bars['Close']))
        signals['signal'] = np.where((self.bars['RSI'] > 50) & \
               (self.bars['MA'] > self.bars['Close']) & \
               (self.bars['ADX'] > 20),1,0)
        return signals
    
class MovingAverageStrategy(Strategy):
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars
        
    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.where((self.bars['RSI'] > 50) & \
           (self.bars['MA'] > self.bars['Close']) & \
           (self.bars['ADX'] > 20),1,0)
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
        positions[self.symbol] = (self.initial_capital*self.pct_hedge *self.signals)
        return positions
                    
    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.bars.index)        
        portfolio['Order_Excute'] = self.bars['Close'] - self.bars['Avg.Price'] #Logical avg.price(t-1) - close(t)
        portfolio['Signal'] = self.signals['signal'] # test generate
        portfolio['PnL'] = (self.positions[self.symbol])/self.bars['Avg.Price'] * portfolio['Order_Excute']
        portfolio['Acc_Port'] = self.initial_capital + portfolio['PnL'].cumsum()
#        portfolio['returns'] = portfolio['total'].pct_change()
        
        # ================ Performace ======================
        
        
        return portfolio
    
    def performance_summary(self):
        sharpe_class = gen_sharpe_ratio(df,1, -1)
        return sharpe_class
    
#class MarketTestPortfolio(Portfolio):
#    
#    def __init__(self, symbol, bars, initial_capital=100000.0, pct_hedge = 1):
#        self.symbol = symbol        
#        self.bars = bars
##        self.signals = signals
#        self.initial_capital = float(initial_capital)
#        self.trading_sum = float(pct_hedge)
#        self.positions = self.generate_positions()
#        
#    def generate_positions(self):
#        positions = pd.DataFrame(index=self.bars.index).fillna(0.0)
#        positions[self.symbol] = self.initial_capital*self.pct_hedge *self.bars['Signal']
#        return positions
#                    
#    def backtest_portfolio(self):
#        portfolio = pd.DataFrame(index=self.positions.index)        
#        portfolio['order_ex'] = (self.bars['Avg.Price'] / self.bars['Close'])-1
#        portfolio['profit'] = self.positions[self.symbol] * portfolio['order_ex']
#        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
#        portfolio['returns'] = portfolio['total'].pct_change()
#        
#        # Performance Report generate 
        # get index -> df.index.values[]
        # date_diff -> pd.to_datetime() - pd.to_datetime()
#        
#        return portfolio
 
    
# Helping functions     
def gen_sharpe_ratio(df,index_sd, index_ed):
    date_diff = pd.to_datetime(df.index.values[index_ed]) - pd.to_datetime(df.index.values[index_sd])
    return (df['Acc_Port'].iloc[index_ed] / df['Acc_Port'].iloc[index_sd])**(365/date_diff.days)-1
    
def gen_drawdowns(df):
    hwm = [0]
    eq_idx = df.index
    drawdown = pd.Series(index=eq_idx)
    #duration = pd.Series(index=eq_idx)
    
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t-1], df['Acc_Port'].iloc[t])
        hwm.append(cur_hwm)
        drawdown[t] = hwm[t] - df['Acc_Port'].iloc[t]
        #duration[t] = 0 if drawdown[t] == 0 else duration[t-1] + 1
    #return drawdown.max(), duration.max()
    return drawdown.max() / df['Acc_Port'].iloc[1]
       

if __name__ == '__main__':
    
    file_path = r'D:\Projects\Data\USDTHB Historical Data.csv'
    df = pd.read_csv(file_path)
    df = df.set_index(pd.to_datetime(df['Date']))
    df = df.drop(['Date'], axis=1)
    #df = pd.read_csv(file_path)
    df = df.dropna()
    symbol='USDTHB'
    
    ma = pd.Series(talib.MA(df['Close'], 20), name='MA')
    rsi = pd.Series(talib.RSI(df['Close'], 14), name='RSI')
    adx = pd.Series(talib.ADX(df['High'], df['Low'], df['Close'], 14), name='ADX')
    kama = pd.Series(talib.KAMA(df['Close'], 20), name='KAMA')
    cci = pd.Series(talib.CCI(df['High'], df['Low'], df['Close'], 14), name='CCI')
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    macd = pd.Series(macd, name='MACD')
    macdsignal = pd.Series(macdsignal, name='MACD_Signal')
    macdhist = pd.Series(macdhist, name='MACD_Hist')
    avg_price = pd.Series((df['Close']*0.7)+(df['Open']*0.1)+(df['High']*0.1)+(df['Low']*0.1), name='Avg.Price').shift(1) #for enter order 
    
    indicators = [avg_price,ma, rsi, adx, kama, cci, macd, macdsignal, macdhist]
    
    for i in indicators:
        df = df.join(i)
    
    # ====================== Hands on generate signal =========================
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

    # ====================== Hands on generate signal =========================
    
    # ====================== Use class generate ===============================
    rfs = CustomStrategy(symbol, df)
    signals = rfs.generate_signals()
    
    ma_port = PortfolioExcute(symbol, signals, df, initial_capital=1000000)
    returns = ma_port.backtest_portfolio()
    df = df.join(returns) # Write to report 
    performance = ma_port.performance_summary()
    print(performance)
    #shape = print(gen_sharpe_ratio(df, 1, -1))
    #max_dd = gen_drawdowns(df)
#    print(returns.tail(15))
     # ====================== Use class generate ==============================
    
    
    
    
#    writer = pd.ExcelWriter('output.xls')
#    df.to_excel(writer, 'Sheet1')
#    writer.save()
    

    
