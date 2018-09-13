from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import talib

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

        
class RandomForecastingStrategy(Strategy):   
    
    def __init__(self, symbol, bars):
        self.symbol = symbol
        self.bars = bars

    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.sign(np.random.randn(len(signals)))
        return signals
    
class MachineLearningForecastingStrategy(Strategy):
    
    def __init__(self, symbol, bars, pred):
        self.symbol = symbol
        self.bars = bars
    
    def generate_signals(self):
        signals = pd.DataFrame(index.self.bars.index)
        signals['signal'] = pred
        return signals
        
    
class MarketIntradayPortfolio(Portfolio):
    
    def __init__(self, symbol, bars, signals, initial_capital=100000.0, trading_sum = 100):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.trading_sum = float(trading_sum)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = self.trading_sum*self.signals['signal']
        return positions
                    
    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.positions.index)
        pos_diff = self.positions.diff()
        
        
        portfolio['price_diff'] = self.bars['Price']-self.bars['Open']
        portfolio['profit'] = self.positions[self.symbol] * portfolio['price_diff']

        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

if __name__ == '__main__':
    
    data_source = r'D:\Projects\Data\USD_THB Historical Data.csv'
    df = pd.read_csv(data_source)
    symbol='USDTHB'
        # Create a set of random forecasting signals for SPY
    
#    def maCross(df):
#        
#    
#    rfs = MachineLearningForecastingStrategy(symbol, df, pred)
#    signals = rfs.generate_signals()
#
#    # Create a portfolio of SPY
#    portfolio = MarketIntradayPortfolio(symbol, df, signals, initial_capital=100000.0)
#    returns = portfolio.backtest_portfolio()
#
#    print(returns.tail(10))
        
    #==================================== Fixed numpy where
#    def conditions(x):
#    if x > 400:
#        return "High"
#    elif x > 200:
#        return "Medium"
#    else:
#        return "Low"
#
#    func = np.vectorize(conditions)
#    energy_class = func(df_energy["consumption_energy"])
#    
#    # Use 
#    df_energy["energy_class"] = energy_class
    #====================================
        
    ma = pd.Series(talib.MA(df['Price'], 20), name='MA')
    rsi = pd.Series(talib.RSI(df['Price'], 14), name='RSI')
    df = df.join(ma)
    df = df.join(rsi)
    #signal = [1 if ma[i] > df['Price'][i] else 0 for i in df['MA']]
    signal = pd.Series(np.where(np.greater(df['Price'], df['MA']),1,0), name='sig1')
    signal2 = pd.Series(np.where(np.logical_and(np.greater_equal(df['RSI'], 50), np.greater(df['Price'], df['MA'])),1,0), name='sig2')
    
    df = df.join(signal)
    df = df.join(signal2)
    
    writer = pd.ExcelWriter('output.xls')
    df.to_excel(writer, 'Sheet1')
    writer.save()
    


