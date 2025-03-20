import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.optimize import minimize

# 创建输出目录
os.makedirs('../data', exist_ok=True)
os.makedirs('../backtest', exist_ok=True)
os.makedirs('../optimization', exist_ok=True)

class StrategyOptimizer:
    """
    量化交易策略优化器
    
    功能:
    1. 参数优化
    2. 策略权重优化
    3. 杠杆优化
    4. 止损优化
    5. 资产配置优化
    """
    
    def __init__(self, backtest_data_path=None, 
                 symbols=['NVDA', 'BTC-USD', 'ETH-USD', 'TSLA'], 
                 start_date=None, 
                 end_date=None,
                 initial_capital=100000):
        """
        初始化优化器
        
        参数:
        backtest_data_path: 回测数据路径，如果提供则从文件加载数据
        symbols: 交易资产列表
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金
        """
        self.symbols = symbols
        
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = end_date
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=365*3)  # 默认3年数据
        else:
            self.start_date = start_date
            
        self.initial_capital = initial_capital
        self.data = {}
        self.signals = {}
        self.positions = {}
        self.performance_metrics = {}
        self.optimized_params = {}
        
        # 加载数据
        self.fetch_data()
    
    def fetch_data(self):
        """获取历史数据并处理多级索引"""
        print(f"获取从 {self.start_date} 到 {self.end_date} 的历史数据...")
        
        for symbol in self.symbols:
            try:
                # 获取日线数据
                data = yf.download(symbol, start=self.start_date, end=self.end_date)
                
                if len(data) < 20:
                    print(f"  {symbol}: 数据不足，跳过")
                    continue
                
                # 检查是否有多级索引并处理
                if isinstance(data.columns, pd.MultiIndex):
                    print(f"  {symbol} 数据有多级索引，进行处理...")
                    # 创建新的DataFrame，只保留需要的列
                    processed_data = pd.DataFrame()
                    processed_data['Open'] = data[('Open', symbol)]
                    processed_data['High'] = data[('High', symbol)]
                    processed_data['Low'] = data[('Low', symbol)]
                    processed_data['Close'] = data[('Close', symbol)]
                    
                    if ('Volume', symbol) in data.columns:
                        processed_data['Volume'] = data[('Volume', symbol)]
                    
                    data = processed_data
                
                # 计算每日回报
                data['Daily_Return'] = data['Close'].pct_change()
                
                # 添加技术指标
                self._add_technical_indicators(data)
                
                # 存储数据
                self.data[symbol] = data
                print(f"  {symbol}: 获取了 {len(data)} 条数据")
                
            except Exception as e:
                print(f"  {symbol}: 获取数据失败 - {str(e)}")
                
        return self.data
    
    def _add_technical_indicators(self, data):
        """添加技术指标"""
        try:
            # 价格指标
            data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['SMA200'] = ta.trend.sma_indicator(data['Close'], window=200)
            
            # 波动率指标
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            bb = ta.volatility.BollingerBands(data['Close'])
            data['BB_Upper'] = bb.bollinger_hband()
            data['BB_Lower'] = bb.bollinger_lband()
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['SMA20']
            
            # 动量指标
            data['RSI'] = ta.momentum.rsi(data['Close'])
            data['MACD'] = ta.trend.macd_diff(data['Close'])
            data['Momentum'] = data['Close'].pct_change(20)
            
            # 趋势指标
            data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
            
            # 成交量指标
            if 'Volume' in data.columns:
                data['Volume_SMA20'] = ta.trend.sma_indicator(data['Volume'], window=20)
                data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA20']
                
            print(f"  成功添加技术指标")
        except Exception as e:
            print(f"  添加技术指标失败: {str(e)}")
        
        return data
    
    def optimize_momentum_parameters(self, symbol, window_range=(5, 100)):
        """优化动量策略参数"""
        print(f"优化 {symbol} 动量策略参数...")
        
        if symbol not in self.data:
            print(f"  {symbol}: 没有数据，跳过")
            return None
        
        data = self.data[symbol]
        
        # 定义评估函数
        def evaluate_window(window):
            window = int(window)
            signals = pd.DataFrame(index=data.index)
            signals['Close'] = data['Close']
            signals['Daily_Return'] = data['Daily_Return']
            
            # 计算动量指标
            signals['Momentum'] = data['Close'].pct_change(window)
            
            # 生成交易信号
            signals['Signal'] = 0
            signals.loc[signals['Momentum'] > 0, 'Signal'] = 1  # 买入信号
            signals.loc[signals['Momentum'] < 0, 'Signal'] = -1  # 卖出信号
            
            # 计算策略收益
            signals['Strategy_Return'] = signals['Daily_Return'] * signals['Signal'].shift(1)
            signals['Strategy_Return'] = signals['Strategy_Return'].fillna(0)
            
            # 计算累积收益
            signals['Cumulative_Return'] = (1 + signals['Strategy_Return']).cumprod()
            
            # 计算年化收益率
            total_return = signals['Cumulative_Return'].iloc[-1] - 1
            days = len(signals)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # 计算胜率
            win_rate = len(signals[signals['Strategy_Return'] > 0]) / len(signals[signals['Strategy_Return'] != 0]) if len(signals[signals['Strategy_Return'] != 0]) > 0 else 0
            
            # 计算夏普比率
            sharpe_ratio = signals['Strategy_Return'].mean() / signals['Strategy_Return'].std() * np.sqrt(252) if signals['Strategy_Return'].std() > 0 else 0
            
            # 计算综合得分 (优化目标)
            # 这里我们优先考虑收益率和胜率，同时兼顾风险
            score = annual_return * 0.5 + win_rate * 0.3 + sharpe_ratio * 0.2
            
            return -score  # 最小化负分数 = 最大化分数
        
        # 使用网格搜索找到最佳窗口
        best_score = float('inf')
        best_window = window_range[0]
        
        for window in range(window_range[0], window_range[1] + 1):
            score = evaluate_window(window)
            if score < best_score:
                best_score = score
                best_window = window
        
        # 使用最佳窗口生成信号
        best_window = int(best_window)
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data['Close']
        signals['Daily_Return'] = data['Daily_Return']
        
        # 计算动量指标
        signals['Momentum'] = data['Close'].pct_change(best_window)
        
        # 生成交易信号
        signals['Signal'] = 0
        signals.loc[signals['Momentum'] > 0, 'Signal'] = 1  # 买入信号
        signals.loc[signals['Momentum'] < 0, 'Signal'] = -1  # 卖出信号
        
        # 计算策略收益
        signals['Strategy_Return'] = signals['Daily_Return'] * signals['Signal'].shift(1)
        signals['Strategy_Return'] = signals['Strategy_Return'].fillna(0)
        
        # 计算累积收益
        signals['Cumulative_Return'] = (1 + signals['Strategy_Return']).cumprod()
        
        # 计算年化收益率
        total_return = signals['Cumulative_Return'].iloc[-1] - 1
        days = len(signals)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算胜率
        win_rate = len(signals[signals['Strategy_Return'] > 0]) / len(signals[signals['Strategy_Return'] != 0]) if len(signals[signals['Strategy_Return'] != 0]) > 0 else 0
        
        print(f"  {symbol} 动量策略最佳窗口: {best_window}")
        print(f"  年化收益率: {annual_return:.2%}, 胜率: {win_rate:.2%}")
        
        # 存储信号
        self.signals[f"{symbol}_momentum_optimized"] = signals
        
        # 存储优化参数
        self.optimized_params[f"{symbol}_momentum"] = {'window': best_window}
        
        return best_window
    
    def optimize_trend_parameters(self, symbol, short_range=(5, 50), long_range=(20, 200)):
        """优化趋势跟踪策略参数"""
        print(f"优化 {symbol} 趋势跟踪策略参数...")
        
        if symbol not in self.data:
            print(f"  {symbol}: 没有数据，跳过")
            return None
        
        data = self.data[symbol]
        
        # 定义评估函数
        def evaluate_windows(params):
            short_window, long_window = int(params[0]), int(params[1])
            
            if short_window >= long_window:
                return 0  # 短期窗口必须小于长期窗口
            
            signals = pd.DataFrame(index=data.index)
            signals['Close'] = data['Close']
            signals['Daily_Return'] = data['Daily_Return']
            
            # 计算短期和长期移动平均线
            signals['Short_MA'] = data['Close'].rolling(window=short_window).mean()
            signals['Long_MA'] = data['Close'].rolling(window=long_window).mean()
            
            # 生成交易信号
            signals['Signal'] = 0
            signals.loc[signals['Short_MA'] > signals['Long_MA'], 'Signal'] = 1  # 买入信号
            signals.loc[signals['Short_MA'] < signals['Long_MA'], 'Signal'] = -1  # 卖出信号
            
            # 计算策略收益
            signals['Strategy_Return'] = signals['Daily_Return'] * signals['Signal'].shift(1)
            signals['Strategy_Return'] = signals['Strategy_Return'].fillna(0)
            
            # 计算累积收益
            signals['Cumulative_Return'] = (1 + signals['Strategy_Return']).cumprod()
            
            # 计算年化收益率
            total_return = signals['Cumulative_Return'].iloc[-1] - 1
            days = len(signals)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # 计算胜率
            win_rate = len(signals[signals['Strategy_Return'] > 0]) / len(signals[signals['Strategy_Return'] != 0]) if len(signals[signals['Strategy_Return'] != 0]) > 0 else 0
            
            # 计算夏普比率
            sharpe_ratio = signals['Strategy_Return'].mean() / signals['Strategy_Return'].std() * np.sqrt(252) if signals['Strategy_Return'].std() > 0 else 0
            
            # 计算综合得分 (优化目标)
            score = annual_return * 0.5 + win_rate * 0.3 + sharpe_ratio * 0.2
            
            return -score  # 最小化负分数 = 最大化分数
        
        # 使用网格搜索找到最佳窗口组合
        best_score = float('inf')
        best_short = short_range[0]
        best_long = long_range[0]
        
        for short_window in range(short_range[0], short_range[1] + 1, 5):
            for long_window in range(long_range[0], long_range[1] + 1, 10):
                if short_window < long_window:
                    score = evaluate_windows([short_window, long_window])
                    if score < best_score:
                        best_score = score
                        best_short = short_window
                        best_long = long_window
        
        # 使用最佳窗口生成信号
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data['Close']
        signals['Daily_Return'] = data['Daily_Return']
        
        # 计算短期和长期移动平均线
        signals['Short_MA'] = data['Close'].rolling(window=best_short).mean()
        signals['Long_MA'] = data['Close'].rolling(window=best_long).mean()
        
        # 生成交易信号
        signals['Signal'] = 0
        signals.loc[signals['Short_MA'] > signals['Long_MA'], 'Signal'] = 1  # 买入信号
        signals.loc[signals['Short_MA'] < signals['Long_MA'], 'Signal'] = -1  # 卖出信号
        
        # 计算策略收益
        signals['Strategy_Return'] = signals['Daily_Return'] * signals['Signal'].shift(1)
        signals['Strategy_Return'] = signals['Strategy_Return'].fillna(0)
        
        # 计算累积收益
        signals['Cumulative_Return'] = (1 + signals['Strategy_Return']).cumprod()
        
        # 计算年化收益率
        total_return = signals['Cumulative_Return'].iloc[-1] - 1
        days = len(signals)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算胜率
        win_rate = len(signals[signals['Strategy_Return'] > 0]) / len(signals[signals['Strategy_Return'] != 0]) if len(signals[signals['Strategy_Return'] != 0]) > 0 else 0
        
        print(f"  {symbol} 趋势跟踪策略最佳参数: 短期窗口={best_short}, 长期窗口={best_long}")
        print(f"  年化收益率: {annual_return:.2%}, 胜率: {win_rate:.2%}")
        
        # 存储信号
        self.signals[f"{symbol}_trend_optimized"] = signals
        
        # 存储优化参数
        self.optimized_params[f"{symbol}_trend"] = {'short_window': best_short, 'long_window': best_long}
        
        return best_short, best_long
    
    def optimize_mean_reversion_parameters(self, symbol, window_range=(10, 30), std_range=(1.5, 3.0, 0.1)):
        """优化均值回归策略参数"""
        print(f"优化 {symbol} 均值回归策略参数...")
        
        if symbol not in self.data:
            print(f"  {symbol}: 没有数据，跳过")
            return None
        
        data = self.data[symbol]
        
        # 定义评估函数
        def evaluate_params(params):
            window, std_dev = int(params[0]), params[1]
            
            signals = pd.DataFrame(index=data.index)
            signals['Close'] = data['Close']
            signals['Daily_Return'] = data['Daily_Return']
            
            # 计算移动平均线和标准差
            signals['SMA'] = data['Close'].rolling(window=window).mean()
            signals['STD'] = data['Close'].rolling(window=window).std()
            
            # 计算上下轨
            signals['Upper_Band'] = signals['SMA'] + std_dev * signals['STD']
            signals['Lower_Band'] = signals['SMA'] - std_dev * signals['STD']
            
            # 生成交易信号
            signals['Signal'] = 0
            signals.loc[signals['Close'] < signals['Lower_Band'], 'Signal'] = 1  # 买入信号(超卖)
            signals.loc[signals['Close'] > signals['Upper_Band'], 'Signal'] = -1  # 卖出信号(超买)
            
            # 填充0值(保持前一个信号)
            signals['Signal'] = signals['Signal'].replace(to_replace=0)
            signals['Signal'] = signals['Signal'].fillna(method='ffill')
            
            # 计算策略收益
            signals['Strategy_Return'] = signals['Daily_Return'] * signals['Signal'].shift(1)
            signals['Strategy_Return'] = signals['Strategy_Return'].fillna(0)
            
            # 计算累积收益
            signals['Cumulative_Return'] = (1 + signals['Strategy_Return']).cumprod()
            
            # 计算年化收益率
            total_return = signals['Cumulative_Return'].iloc[-1] - 1
            days = len(signals)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # 计算胜率
            win_rate = len(signals[signals['Strategy_Return'] > 0]) / len(signals[signals['Strategy_Return'] != 0]) if len(signals[signals['Strategy_Return'] != 0]) > 0 else 0
            
            # 计算夏普比率
            sharpe_ratio = signals['Strategy_Return'].mean() / signals['Strategy_Return'].std() * np.sqrt(252) if signals['Strategy_Return'].std() > 0 else 0
            
            # 计算综合得分 (优化目标)
            score = annual_return * 0.5 + win_rate * 0.3 + sharpe_ratio * 0.2
            
            return -score  # 最小化负分数 = 最大化分数
        
        # 使用网格搜索找到最佳参数组合
        best_score = float('inf')
        best_window = window_range[0]
        best_std = std_range[0]
        
        for window in range(window_range[0], window_range[1] + 1, 2):
            for std_dev in np.arange(std_range[0], std_range[1] + 0.01, std_range[2]):
                score = evaluate_params([window, std_dev])
                if score < best_score:
                    best_score = score
                    best_window = window
                    best_std = std_dev
        
        # 使用最佳参数生成信号
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data['Close']
        signals['Daily_Return'] = data['Daily_Return']
        
        # 计算移动平均线和标准差
        signals['SMA'] = data['Close'].rolling(window=best_window).mean()
        signals['STD'] = data['Close'].rolling(window=best_window).std()
        
        # 计算上下轨
        signals['Upper_Band'] = signals['SMA'] + best_std * signals['STD']
        signals['Lower_Band'] = signals['SMA'] - best_std * signals['STD']
        
        # 生成交易信号
        signals['Signal'] = 0
        signals.loc[signals['Close'] < signals['Lower_Band'], 'Signal'] = 1  # 买入信号(超卖)
        signals.loc[signals['Close'] > signals['Upper_Band'], 'Signal'] = -1  # 卖出信号(超买)
        
        # 填充0值(保持前一个信号)
        signals['Signal'] = signals['Signal'].replace(to_replace=0)
        signals['Signal'] = signals['Signal'].fillna(method='ffill')
        
        # 计算策略收益
        signals['Strategy_Return'] = signals['Daily_Return'] * signals['Signal'].shift(1)
        signals['Strategy_Return'] = signals['Strategy_Return'].fillna(0)
        
        # 计算累积收益
        signals['Cumulative_Return'] = (1 + signals['Strategy_Return']).cumprod()
        
        # 计算年化收益率
        total_return = signals['Cumulative_Return'].iloc[-1] - 1
        days = len(signals)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算胜率
        win_rate = len(signals[signals['Strategy_Return'] > 0]) / len(signals[signals['Strategy_Return'] != 0]) if len(signals[signals['Strategy_Return'] != 0]) > 0 else 0
        
        print(f"  {symbol} 均值回归策略最佳参数: 窗口={best_window}, 标准差倍数={best_std:.2f}")
        print(f"  年化收益率: {annual_return:.2%}, 胜率: {win_rate:.2%}")
        
        # 存储信号
        self.signals[f"{symbol}_mean_reversion_optimized"] = signals
        
        # 存储优化参数
        self.optimized_params[f"{symbol}_mean_reversion"] = {'window': best_window, 'std_dev': best_std}
        
        return best_window, best_std
    
    def optimize_strategy_weights(self, symbol):
        """优化策略权重"""
        print(f"优化 {symbol} 策略权重...")
        
        # 检查是否有所有需要的策略信号
        momentum_key = f"{symbol}_momentum_optimized"
        trend_key = f"{symbol}_trend_optimized"
        mean_reversion_key = f"{symbol}_mean_reversion_optimized"
        
        if momentum_key not in self.signals or trend_key not in self.signals or mean_reversion_key not in self.signals:
            print(f"  {symbol}: 缺少某些优化策略信号，跳过")
            return None
        
        # 获取各策略信号
        momentum_signals = self.signals[momentum_key]
        trend_signals = self.signals[trend_key]
        mean_reversion_signals = self.signals[mean_reversion_key]
        
        # 定义评估函数
        def evaluate_weights(weights):
            # 确保权重和为1
            weights = weights / np.sum(weights)
            
            # 创建组合信号DataFrame
            combined = pd.DataFrame(index=momentum_signals.index)
            combined['Close'] = momentum_signals['Close']
            combined['Daily_Return'] = momentum_signals['Daily_Return']
            
            # 合并信号
            combined['Momentum_Signal'] = momentum_signals['Signal']
            combined['Trend_Signal'] = trend_signals['Signal']
            combined['Mean_Reversion_Signal'] = mean_reversion_signals['Signal']
            
            # 计算加权组合信号
            combined['Combined_Signal'] = (
                weights[0] * combined['Momentum_Signal'] +
                weights[1] * combined['Trend_Signal'] +
                weights[2] * combined['Mean_Reversion_Signal']
            )
            
            # 最终信号：大于0为买入，小于0为卖出
            combined['Signal'] = np.sign(combined['Combined_Signal'])
            
            # 计算策略收益
            combined['Strategy_Return'] = combined['Daily_Return'] * combined['Signal'].shift(1)
            combined['Strategy_Return'] = combined['Strategy_Return'].fillna(0)
            
            # 计算累积收益
            combined['Cumulative_Return'] = (1 + combined['Strategy_Return']).cumprod()
            
            # 计算年化收益率
            total_return = combined['Cumulative_Return'].iloc[-1] - 1
            days = len(combined)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # 计算胜率
            win_rate = len(combined[combined['Strategy_Return'] > 0]) / len(combined[combined['Strategy_Return'] != 0]) if len(combined[combined['Strategy_Return'] != 0]) > 0 else 0
            
            # 计算夏普比率
            sharpe_ratio = combined['Strategy_Return'].mean() / combined['Strategy_Return'].std() * np.sqrt(252) if combined['Strategy_Return'].std() > 0 else 0
            
            # 计算综合得分 (优化目标)
            # 这里我们更加重视胜率，以达到用户要求的68%胜率目标
            score = annual_return * 0.4 + win_rate * 0.4 + sharpe_ratio * 0.2
            
            return -score  # 最小化负分数 = 最大化分数
        
        # 使用优化算法找到最佳权重
        initial_weights = np.array([0.33, 0.33, 0.34])  # 初始权重
        bounds = [(0, 1), (0, 1), (0, 1)]  # 权重范围
        
        # 添加约束条件：权重和为1
        constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        
        # 优化
        result = minimize(evaluate_weights, initial_weights, method='SLSQP', bounds=bounds, constraints=constraint)
        
        if result.success:
            best_weights = result.x
            # 确保权重和为1
            best_weights = best_weights / np.sum(best_weights)
        else:
            print(f"  {symbol}: 权重优化失败，使用均等权重")
            best_weights = np.array([0.33, 0.33, 0.34])
        
        # 使用最佳权重生成组合信号
        combined = pd.DataFrame(index=momentum_signals.index)
        combined['Close'] = momentum_signals['Close']
        combined['Daily_Return'] = momentum_signals['Daily_Return']
        
        # 合并信号
        combined['Momentum_Signal'] = momentum_signals['Signal']
        combined['Trend_Signal'] = trend_signals['Signal']
        combined['Mean_Reversion_Signal'] = mean_reversion_signals['Signal']
        
        # 计算加权组合信号
        combined['Combined_Signal'] = (
            best_weights[0] * combined['Momentum_Signal'] +
            best_weights[1] * combined['Trend_Signal'] +
            best_weights[2] * combined['Mean_Reversion_Signal']
        )
        
        # 最终信号：大于0为买入，小于0为卖出
        combined['Signal'] = np.sign(combined['Combined_Signal'])
        
        # 计算策略收益
        combined['Strategy_Return'] = combined['Daily_Return'] * combined['Signal'].shift(1)
        combined['Strategy_Return'] = combined['Strategy_Return'].fillna(0)
        
        # 计算累积收益
        combined['Cumulative_Return'] = (1 + combined['Strategy_Return']).cumprod()
        
        # 计算年化收益率
        total_return = combined['Cumulative_Return'].iloc[-1] - 1
        days = len(combined)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算胜率
        win_rate = len(combined[combined['Strategy_Return'] > 0]) / len(combined[combined['Strategy_Return'] != 0]) if len(combined[combined['Strategy_Return'] != 0]) > 0 else 0
        
        print(f"  {symbol} 最佳策略权重: 动量={best_weights[0]:.2f}, 趋势={best_weights[1]:.2f}, 均值回归={best_weights[2]:.2f}")
        print(f"  年化收益率: {annual_return:.2%}, 胜率: {win_rate:.2%}")
        
        # 存储信号
        self.signals[f"{symbol}_combined_optimized"] = combined
        
        # 存储优化参数
        self.optimized_params[f"{symbol}_combined"] = {
            'momentum_weight': best_weights[0],
            'trend_weight': best_weights[1],
            'mean_reversion_weight': best_weights[2]
        }
        
        return best_weights
    
    def optimize_leverage(self, symbol, max_leverage=5, step=0.5):
        """优化杠杆倍数"""
        print(f"优化 {symbol} 杠杆倍数...")
        
        # 检查是否有组合策略信号
        combined_key = f"{symbol}_combined_optimized"
        
        if combined_key not in self.signals:
            print(f"  {symbol}: 没有组合策略信号，跳过")
            return None
        
        combined_signals = self.signals[combined_key]
        
        # 定义评估函数
        def evaluate_leverage(leverage):
            # 创建杠杆策略信号
            leveraged = combined_signals.copy()
            
            # 应用杠杆
            leveraged['Leveraged_Signal'] = leveraged['Signal'] * leverage
            
            # 计算策略收益
            leveraged['Strategy_Return'] = leveraged['Daily_Return'] * leveraged['Leveraged_Signal'].shift(1)
            leveraged['Strategy_Return'] = leveraged['Strategy_Return'].fillna(0)
            
            # 计算累积收益
            leveraged['Cumulative_Return'] = (1 + leveraged['Strategy_Return']).cumprod()
            
            # 计算年化收益率
            total_return = leveraged['Cumulative_Return'].iloc[-1] - 1
            days = len(leveraged)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # 计算胜率
            win_rate = len(leveraged[leveraged['Strategy_Return'] > 0]) / len(leveraged[leveraged['Strategy_Return'] != 0]) if len(leveraged[leveraged['Strategy_Return'] != 0]) > 0 else 0
            
            # 计算最大回撤
            peak = leveraged['Cumulative_Return'].cummax()
            drawdown = (leveraged['Cumulative_Return'] - peak) / peak
            max_drawdown = drawdown.min()
            
            # 计算夏普比率
            sharpe_ratio = leveraged['Strategy_Return'].mean() / leveraged['Strategy_Return'].std() * np.sqrt(252) if leveraged['Strategy_Return'].std() > 0 else 0
            
            # 计算综合得分 (优化目标)
            # 随着杠杆增加，我们更加重视风险控制
            score = annual_return * 0.4 + win_rate * 0.3 + sharpe_ratio * 0.2 - abs(max_drawdown) * 0.1
            
            return -score  # 最小化负分数 = 最大化分数
        
        # 使用网格搜索找到最佳杠杆
        best_score = float('inf')
        best_leverage = 1.0
        
        for leverage in np.arange(1.0, max_leverage + 0.01, step):
            score = evaluate_leverage(leverage)
            if score < best_score:
                best_score = score
                best_leverage = leverage
        
        # 使用最佳杠杆生成信号
        leveraged = combined_signals.copy()
        
        # 应用杠杆
        leveraged['Leveraged_Signal'] = leveraged['Signal'] * best_leverage
        
        # 计算策略收益
        leveraged['Strategy_Return'] = leveraged['Daily_Return'] * leveraged['Leveraged_Signal'].shift(1)
        leveraged['Strategy_Return'] = leveraged['Strategy_Return'].fillna(0)
        
        # 计算累积收益
        leveraged['Cumulative_Return'] = (1 + leveraged['Strategy_Return']).cumprod()
        
        # 计算年化收益率
        total_return = leveraged['Cumulative_Return'].iloc[-1] - 1
        days = len(leveraged)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算胜率
        win_rate = len(leveraged[leveraged['Strategy_Return'] > 0]) / len(leveraged[leveraged['Strategy_Return'] != 0]) if len(leveraged[leveraged['Strategy_Return'] != 0]) > 0 else 0
        
        # 计算最大回撤
        peak = leveraged['Cumulative_Return'].cummax()
        drawdown = (leveraged['Cumulative_Return'] - peak) / peak
        max_drawdown = drawdown.min()
        
        print(f"  {symbol} 最佳杠杆倍数: {best_leverage:.1f}")
        print(f"  年化收益率: {annual_return:.2%}, 胜率: {win_rate:.2%}, 最大回撤: {max_drawdown:.2%}")
        
        # 存储信号
        self.signals[f"{symbol}_leveraged_optimized"] = leveraged
        
        # 存储优化参数
        self.optimized_params[f"{symbol}_leverage"] = {'leverage': best_leverage}
        
        return best_leverage
    
    def optimize_stop_loss(self, symbol, stop_loss_range=(0.01, 0.1, 0.01)):
        """优化止损参数"""
        print(f"优化 {symbol} 止损参数...")
        
        # 检查是否有杠杆策略信号
        leveraged_key = f"{symbol}_leveraged_optimized"
        
        if leveraged_key not in self.signals:
            print(f"  {symbol}: 没有杠杆策略信号，跳过")
            return None
        
        leveraged_signals = self.signals[leveraged_key]
        
        # 定义评估函数
        def evaluate_stop_loss(stop_loss):
            # 创建止损策略信号
            with_stop_loss = leveraged_signals.copy()
            
            # 初始化止损状态
            with_stop_loss['In_Position'] = True
            with_stop_loss['Stop_Loss_Price'] = np.nan
            
            # 模拟交易
            position = 0
            entry_price = 0
            stop_loss_price = 0
            
            for i in range(1, len(with_stop_loss)):
                prev_signal = with_stop_loss['Leveraged_Signal'].iloc[i-1]
                curr_price = with_stop_loss['Close'].iloc[i]
                
                # 如果前一天有信号变化，更新持仓
                if prev_signal != position:
                    position = prev_signal
                    entry_price = with_stop_loss['Close'].iloc[i-1]
                    
                    # 设置止损价格
                    if position > 0:  # 多头
                        stop_loss_price = entry_price * (1 - stop_loss)
                    elif position < 0:  # 空头
                        stop_loss_price = entry_price * (1 + stop_loss)
                    else:  # 空仓
                        stop_loss_price = 0
                
                # 检查是否触发止损
                if position > 0 and curr_price < stop_loss_price:  # 多头止损
                    with_stop_loss['In_Position'].iloc[i] = False
                    position = 0
                elif position < 0 and curr_price > stop_loss_price:  # 空头止损
                    with_stop_loss['In_Position'].iloc[i] = False
                    position = 0
                
                # 记录止损价格
                with_stop_loss['Stop_Loss_Price'].iloc[i] = stop_loss_price
            
            # 应用止损
            with_stop_loss['Effective_Signal'] = with_stop_loss['Leveraged_Signal'] * with_stop_loss['In_Position']
            
            # 计算策略收益
            with_stop_loss['Strategy_Return'] = with_stop_loss['Daily_Return'] * with_stop_loss['Effective_Signal'].shift(1)
            with_stop_loss['Strategy_Return'] = with_stop_loss['Strategy_Return'].fillna(0)
            
            # 计算累积收益
            with_stop_loss['Cumulative_Return'] = (1 + with_stop_loss['Strategy_Return']).cumprod()
            
            # 计算年化收益率
            total_return = with_stop_loss['Cumulative_Return'].iloc[-1] - 1
            days = len(with_stop_loss)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # 计算胜率
            win_rate = len(with_stop_loss[with_stop_loss['Strategy_Return'] > 0]) / len(with_stop_loss[with_stop_loss['Strategy_Return'] != 0]) if len(with_stop_loss[with_stop_loss['Strategy_Return'] != 0]) > 0 else 0
            
            # 计算最大回撤
            peak = with_stop_loss['Cumulative_Return'].cummax()
            drawdown = (with_stop_loss['Cumulative_Return'] - peak) / peak
            max_drawdown = drawdown.min()
            
            # 计算夏普比率
            sharpe_ratio = with_stop_loss['Strategy_Return'].mean() / with_stop_loss['Strategy_Return'].std() * np.sqrt(252) if with_stop_loss['Strategy_Return'].std() > 0 else 0
            
            # 计算综合得分 (优化目标)
            score = annual_return * 0.3 + win_rate * 0.4 + sharpe_ratio * 0.2 - abs(max_drawdown) * 0.1
            
            return -score  # 最小化负分数 = 最大化分数
        
        # 使用网格搜索找到最佳止损参数
        best_score = float('inf')
        best_stop_loss = stop_loss_range[0]
        
        for stop_loss in np.arange(stop_loss_range[0], stop_loss_range[1] + 0.001, stop_loss_range[2]):
            score = evaluate_stop_loss(stop_loss)
            if score < best_score:
                best_score = score
                best_stop_loss = stop_loss
        
        # 使用最佳止损参数生成信号
        with_stop_loss = leveraged_signals.copy()
        
        # 初始化止损状态
        with_stop_loss['In_Position'] = True
        with_stop_loss['Stop_Loss_Price'] = np.nan
        
        # 模拟交易
        position = 0
        entry_price = 0
        stop_loss_price = 0
        
        for i in range(1, len(with_stop_loss)):
            prev_signal = with_stop_loss['Leveraged_Signal'].iloc[i-1]
            curr_price = with_stop_loss['Close'].iloc[i]
            
            # 如果前一天有信号变化，更新持仓
            if prev_signal != position:
                position = prev_signal
                entry_price = with_stop_loss['Close'].iloc[i-1]
                
                # 设置止损价格
                if position > 0:  # 多头
                    stop_loss_price = entry_price * (1 - best_stop_loss)
                elif position < 0:  # 空头
                    stop_loss_price = entry_price * (1 + best_stop_loss)
                else:  # 空仓
                    stop_loss_price = 0
            
            # 检查是否触发止损
            if position > 0 and curr_price < stop_loss_price:  # 多头止损
                with_stop_loss['In_Position'].iloc[i] = False
                position = 0
            elif position < 0 and curr_price > stop_loss_price:  # 空头止损
                with_stop_loss['In_Position'].iloc[i] = False
                position = 0
            
            # 记录止损价格
            with_stop_loss['Stop_Loss_Price'].iloc[i] = stop_loss_price
        
        # 应用止损
        with_stop_loss['Effective_Signal'] = with_stop_loss['Leveraged_Signal'] * with_stop_loss['In_Position']
        
        # 计算策略收益
        with_stop_loss['Strategy_Return'] = with_stop_loss['Daily_Return'] * with_stop_loss['Effective_Signal'].shift(1)
        with_stop_loss['Strategy_Return'] = with_stop_loss['Strategy_Return'].fillna(0)
        
        # 计算累积收益
        with_stop_loss['Cumulative_Return'] = (1 + with_stop_loss['Strategy_Return']).cumprod()
        
        # 计算年化收益率
        total_return = with_stop_loss['Cumulative_Return'].iloc[-1] - 1
        days = len(with_stop_loss)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算胜率
        win_rate = len(with_stop_loss[with_stop_loss['Strategy_Return'] > 0]) / len(with_stop_loss[with_stop_loss['Strategy_Return'] != 0]) if len(with_stop_loss[with_stop_loss['Strategy_Return'] != 0]) > 0 else 0
        
        # 计算最大回撤
        peak = with_stop_loss['Cumulative_Return'].cummax()
        drawdown = (with_stop_loss['Cumulative_Return'] - peak) / peak
        max_drawdown = drawdown.min()
        
        print(f"  {symbol} 最佳止损比例: {best_stop_loss:.2%}")
        print(f"  年化收益率: {annual_return:.2%}, 胜率: {win_rate:.2%}, 最大回撤: {max_drawdown:.2%}")
        
        # 存储信号
        self.signals[f"{symbol}_final_optimized"] = with_stop_loss
        
        # 存储优化参数
        self.optimized_params[f"{symbol}_stop_loss"] = {'stop_loss': best_stop_loss}
        
        return best_stop_loss
    
    def optimize_portfolio_allocation(self):
        """优化投资组合资产配置"""
        print("优化投资组合资产配置...")
        
        # 收集各资产的最终优化策略
        final_signals = {}
        for symbol in self.symbols:
            final_key = f"{symbol}_final_optimized"
            if final_key in self.signals:
                final_signals[symbol] = self.signals[final_key]
        
        if not final_signals:
            print("  没有可用的最终优化策略，跳过")
            return None
        
        # 提取各资产的收益率序列
        returns = {}
        for symbol, signals in final_signals.items():
            returns[symbol] = signals['Strategy_Return']
        
        # 创建收益率DataFrame
        returns_df = pd.DataFrame(returns)
        
        # 计算协方差矩阵
        cov_matrix = returns_df.cov() * 252  # 年化协方差
        
        # 计算各资产的年化收益率
        annual_returns = {}
        for symbol, signals in final_signals.items():
            total_return = (1 + signals['Strategy_Return']).prod() - 1
            days = len(signals)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            annual_returns[symbol] = annual_return
        
        # 定义评估函数
        def evaluate_weights(weights):
            # 确保权重和为1
            weights = weights / np.sum(weights)
            
            # 计算投资组合年化收益率
            portfolio_return = sum(annual_returns[symbol] * weight for symbol, weight in zip(returns.keys(), weights))
            
            # 计算投资组合年化波动率
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 计算夏普比率
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # 计算综合得分 (优化目标)
            score = portfolio_return * 0.6 + sharpe_ratio * 0.4
            
            return -score  # 最小化负分数 = 最大化分数
        
        # 使用优化算法找到最佳权重
        n_assets = len(returns)
        initial_weights = np.ones(n_assets) / n_assets  # 初始均等权重
        bounds = [(0, 1) for _ in range(n_assets)]  # 权重范围
        
        # 添加约束条件：权重和为1
        constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        
        # 优化
        result = minimize(evaluate_weights, initial_weights, method='SLSQP', bounds=bounds, constraints=constraint)
        
        if result.success:
            best_weights = result.x
            # 确保权重和为1
            best_weights = best_weights / np.sum(best_weights)
        else:
            print("  投资组合优化失败，使用均等权重")
            best_weights = np.ones(n_assets) / n_assets
        
        # 计算最优投资组合的性能
        portfolio_return = sum(annual_returns[symbol] * weight for symbol, weight in zip(returns.keys(), best_weights))
        portfolio_volatility = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        print("最优资产配置权重:")
        for symbol, weight in zip(returns.keys(), best_weights):
            print(f"  {symbol}: {weight:.2%}")
        
        print(f"投资组合年化收益率: {portfolio_return:.2%}")
        print(f"投资组合年化波动率: {portfolio_volatility:.2%}")
        print(f"投资组合夏普比率: {sharpe_ratio:.2f}")
        
        # 存储优化参数
        self.optimized_params['portfolio_allocation'] = {symbol: weight for symbol, weight in zip(returns.keys(), best_weights)}
        
        return {symbol: weight for symbol, weight in zip(returns.keys(), best_weights)}
    
    def run_optimization(self):
        """运行完整优化流程"""
        # 1. 优化各策略参数
        for symbol in self.symbols:
            # 优化动量策略参数
            self.optimize_momentum_parameters(symbol)
            
            # 优化趋势跟踪策略参数
            self.optimize_trend_parameters(symbol)
            
            # 优化均值回归策略参数
            self.optimize_mean_reversion_parameters(symbol)
            
            # 优化策略权重
            self.optimize_strategy_weights(symbol)
            
            # 优化杠杆倍数
            self.optimize_leverage(symbol)
            
            # 优化止损参数
            self.optimize_stop_loss(symbol)
        
        # 2. 优化投资组合资产配置
        self.optimize_portfolio_allocation()
        
        # 3. 生成优化报告
        self.generate_optimization_report()
        
        return self.optimized_params
    
    def generate_optimization_report(self):
        """生成优化报告"""
        report = "# 量化交易策略优化报告\n\n"
        
        # 添加优化参数概览
        report += "## 优化参数概览\n\n"
        
        # 添加各资产的策略参数
        for symbol in self.symbols:
            report += f"### {symbol} 策略参数\n\n"
            
            # 动量策略参数
            momentum_key = f"{symbol}_momentum"
            if momentum_key in self.optimized_params:
                params = self.optimized_params[momentum_key]
                report += f"#### 动量策略\n"
                report += f"- 窗口: {params['window']}\n\n"
            
            # 趋势跟踪策略参数
            trend_key = f"{symbol}_trend"
            if trend_key in self.optimized_params:
                params = self.optimized_params[trend_key]
                report += f"#### 趋势跟踪策略\n"
                report += f"- 短期窗口: {params['short_window']}\n"
                report += f"- 长期窗口: {params['long_window']}\n\n"
            
            # 均值回归策略参数
            mean_reversion_key = f"{symbol}_mean_reversion"
            if mean_reversion_key in self.optimized_params:
                params = self.optimized_params[mean_reversion_key]
                report += f"#### 均值回归策略\n"
                report += f"- 窗口: {params['window']}\n"
                report += f"- 标准差倍数: {params['std_dev']:.2f}\n\n"
            
            # 组合策略权重
            combined_key = f"{symbol}_combined"
            if combined_key in self.optimized_params:
                params = self.optimized_params[combined_key]
                report += f"#### 组合策略权重\n"
                report += f"- 动量策略权重: {params['momentum_weight']:.2f}\n"
                report += f"- 趋势跟踪策略权重: {params['trend_weight']:.2f}\n"
                report += f"- 均值回归策略权重: {params['mean_reversion_weight']:.2f}\n\n"
            
            # 杠杆参数
            leverage_key = f"{symbol}_leverage"
            if leverage_key in self.optimized_params:
                params = self.optimized_params[leverage_key]
                report += f"#### 杠杆参数\n"
                report += f"- 杠杆倍数: {params['leverage']:.1f}\n\n"
            
            # 止损参数
            stop_loss_key = f"{symbol}_stop_loss"
            if stop_loss_key in self.optimized_params:
                params = self.optimized_params[stop_loss_key]
                report += f"#### 止损参数\n"
                report += f"- 止损比例: {params['stop_loss']:.2%}\n\n"
        
        # 添加投资组合资产配置
        if 'portfolio_allocation' in self.optimized_params:
            report += "## 投资组合资产配置\n\n"
            for symbol, weight in self.optimized_params['portfolio_allocation'].items():
                report += f"- {symbol}: {weight:.2%}\n"
            report += "\n"
        
        # 添加策略性能概览
        report += "## 策略性能概览\n\n"
        report += "| 资产 | 策略 | 年化收益率 | 胜率 | 最大回撤 |\n"
        report += "|------|------|------------|------|----------|\n"
        
        for symbol in self.symbols:
            final_key = f"{symbol}_final_optimized"
            if final_key in self.signals:
                signals = self.signals[final_key]
                
                # 计算年化收益率
                total_return = (1 + signals['Strategy_Return']).prod() - 1
                days = len(signals)
                annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
                
                # 计算胜率
                win_rate = len(signals[signals['Strategy_Return'] > 0]) / len(signals[signals['Strategy_Return'] != 0]) if len(signals[signals['Strategy_Return'] != 0]) > 0 else 0
                
                # 计算最大回撤
                peak = signals['Cumulative_Return'].cummax()
                drawdown = (signals['Cumulative_Return'] - peak) / peak
                max_drawdown = drawdown.min()
                
                report += f"| {symbol} | 优化后策略 | {annual_return:.2%} | {win_rate:.2%} | {max_drawdown:.2%} |\n"
        
        # 添加投资组合性能
        if 'portfolio_allocation' in self.optimized_params:
            # 收集各资产的最终优化策略
            final_signals = {}
            for symbol in self.symbols:
                final_key = f"{symbol}_final_optimized"
                if final_key in self.signals:
                    final_signals[symbol] = self.signals[final_key]
            
            if final_signals:
                # 提取各资产的收益率序列
                returns = {}
                for symbol, signals in final_signals.items():
                    returns[symbol] = signals['Strategy_Return']
                
                # 创建收益率DataFrame
                returns_df = pd.DataFrame(returns)
                
                # 计算投资组合收益率
                weights = np.array([self.optimized_params['portfolio_allocation'].get(symbol, 0) for symbol in returns.keys()])
                portfolio_returns = returns_df.dot(weights)
                
                # 计算年化收益率
                total_return = (1 + portfolio_returns).prod() - 1
                days = len(portfolio_returns)
                annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
                
                # 计算胜率
                win_rate = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns[portfolio_returns != 0]) if len(portfolio_returns[portfolio_returns != 0]) > 0 else 0
                
                # 计算最大回撤
                cumulative_return = (1 + portfolio_returns).cumprod()
                peak = cumulative_return.cummax()
                drawdown = (cumulative_return - peak) / peak
                max_drawdown = drawdown.min()
                
                report += f"| 投资组合 | 优化后组合 | {annual_return:.2%} | {win_rate:.2%} | {max_drawdown:.2%} |\n"
        
        # 添加结论和建议
        report += "\n## 结论与建议\n\n"
        
        # 检查是否达到目标
        target_return = 1.0  # 100%
        target_win_rate = 0.68  # 68%
        
        # 找出最佳资产和策略
        best_return = 0
        best_win_rate = 0
        best_return_asset = ""
        best_win_rate_asset = ""
        
        for symbol in self.symbols:
            final_key = f"{symbol}_final_optimized"
            if final_key in self.signals:
                signals = self.signals[final_key]
                
                # 计算年化收益率
                total_return = (1 + signals['Strategy_Return']).prod() - 1
                days = len(signals)
                annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
                
                # 计算胜率
                win_rate = len(signals[signals['Strategy_Return'] > 0]) / len(signals[signals['Strategy_Return'] != 0]) if len(signals[signals['Strategy_Return'] != 0]) > 0 else 0
                
                if annual_return > best_return:
                    best_return = annual_return
                    best_return_asset = symbol
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_win_rate_asset = symbol
        
        report += f"1. 最高年化收益率: {best_return:.2%} ({best_return_asset})\n"
        report += f"2. 最高胜率: {best_win_rate:.2%} ({best_win_rate_asset})\n\n"
        
        # 评估是否达到目标
        if best_return >= target_return:
            report += "3. 已达到年化收益率100%的目标\n"
        else:
            report += f"3. 未达到年化收益率100%的目标，最高为{best_return:.2%}\n"
        
        if best_win_rate >= target_win_rate:
            report += "4. 已达到胜率68%的目标\n"
        else:
            report += f"4. 未达到胜率68%的目标，最高为{best_win_rate:.2%}\n\n"
        
        # 添加进一步优化建议
        report += "5. 进一步优化建议:\n"
        
        if best_return < target_return:
            report += "   - 考虑更高的杠杆倍数，但需要更严格的风险管理\n"
            report += "   - 探索更多高波动性资产，如加密货币期权\n"
            report += "   - 尝试更短的交易周期，如日内交易\n"
        
        if best_win_rate < target_win_rate:
            report += "   - 优化入场和出场条件，增加信号过滤器\n"
            report += "   - 结合机器学习模型预测市场方向\n"
            report += "   - 添加更多技术指标，提高信号质量\n"
            report += "   - 实施更严格的止损和止盈策略\n"
        
        report += "   - 考虑添加更多资产类别，如商品、债券等，增加多元化\n"
        report += "   - 实施动态资产配置，根据市场状况调整权重\n"
        
        # 保存报告
        with open('../optimization/optimization_report.md', 'w') as f:
            f.write(report)
        
        print("优化报告已保存到 ../optimization/optimization_report.md")
        
        return report

# 运行优化
if __name__ == "__main__":
    # 创建优化器实例
    optimizer = StrategyOptimizer(
        symbols=['NVDA', 'BTC-USD', 'ETH-USD', 'TSLA'],
        start_date=datetime.now() - timedelta(days=365*3),
        end_date=datetime.now()
    )
    
    # 运行优化
    optimized_params = optimizer.run_optimization()
    
    # 输出结果
    print("\n优化完成!")
    print("查看 ../optimization/optimization_report.md 获取详细报告")
