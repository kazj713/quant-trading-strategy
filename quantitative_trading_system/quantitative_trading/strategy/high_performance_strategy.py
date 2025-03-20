import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 创建输出目录
os.makedirs('../data', exist_ok=True)
os.makedirs('../strategy', exist_ok=True)

class HighPerformanceStrategy:
    """
    高性能量化交易策略，目标年化收益率100%，胜率68%
    
    策略特点:
    1. 多策略组合: 结合动量、趋势跟踪和机器学习
    2. 多资产配置: 同时交易多个高波动性资产
    3. 杠杆应用: 使用适当杠杆放大收益
    4. 高频交易: 日内和短期交易相结合
    5. 自适应风险管理: 动态调整仓位和止损
    """
    
    def __init__(self, symbols=['NVDA', 'BTC-USD'], 
                 lookback_period=365, 
                 max_leverage=3,
                 risk_tolerance=0.02,
                 rebalance_frequency='daily'):
        """
        初始化策略
        
        参数:
        symbols: 交易资产列表
        lookback_period: 回测历史数据天数
        max_leverage: 最大杠杆倍数
        risk_tolerance: 风险容忍度(每笔交易最大亏损比例)
        rebalance_frequency: 再平衡频率('daily', 'weekly')
        """
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.max_leverage = max_leverage
        self.risk_tolerance = risk_tolerance
        self.rebalance_frequency = rebalance_frequency
        self.data = {}
        self.models = {}
        self.signals = {}
        self.positions = {}
        self.portfolio_value = 100000  # 初始资金
        self.current_leverage = 1.0
        
    def fetch_data(self, start_date=None, end_date=None):
        """获取历史数据"""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.lookback_period)
            
        print(f"获取从 {start_date} 到 {end_date} 的历史数据...")
        
        for symbol in self.symbols:
            try:
                # 获取日线数据
                daily_data = yf.download(symbol, start=start_date, end=end_date)
                
                if len(daily_data) < 20:
                    print(f"  {symbol}: 数据不足，跳过")
                    continue
                
                # 打印数据列以便调试
                print(f"  {symbol} 数据列: {daily_data.columns.tolist()}")
                
                # 计算每日回报
                daily_data['Daily_Return'] = daily_data['Close'].pct_change()
                
                # 添加技术指标
                self._add_technical_indicators(daily_data)
                
                # 存储数据
                self.data[symbol] = daily_data
                print(f"  {symbol}: 获取了 {len(daily_data)} 条数据")
                
                # 尝试获取小时级数据用于日内交易
                try:
                    # 只获取最近30天的小时数据
                    recent_start = end_date - timedelta(days=30)
                    hourly_data = yf.download(symbol, start=recent_start, end=end_date, interval='1h')
                    
                    if len(hourly_data) > 0:
                        hourly_data['Hourly_Return'] = hourly_data['Close'].pct_change()
                        self._add_technical_indicators(hourly_data)
                        self.data[f"{symbol}_hourly"] = hourly_data
                        print(f"  {symbol}: 获取了 {len(hourly_data)} 条小时数据")
                except Exception as e:
                    print(f"  {symbol}: 获取小时数据失败 - {str(e)}")
                    
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
        except Exception as e:
            print(f"添加技术指标失败: {str(e)}")
        
        return data
    
    def train_ml_models(self):
        """训练机器学习模型"""
        print("训练机器学习模型...")
        
        if not self.data:
            print("没有可用数据，跳过模型训练")
            return self.models
        
        for symbol, data in self.data.items():
            if '_hourly' in symbol:
                continue  # 跳过小时数据，只用日线数据训练模型
                
            try:
                # 准备特征和目标
                features = [
                    'RSI', 'MACD', 'Momentum', 'ADX', 
                    'BB_Width', 'ATR',
                    'SMA20', 'SMA50', 'SMA200'
                ]
                
                # 添加成交量特征(如果有)
                if 'Volume_Ratio' in data.columns:
                    features.append('Volume_Ratio')
                
                # 创建目标变量：未来5天的收益率是否为正
                data['Target_5d'] = np.where(data['Close'].shift(-5) > data['Close'], 1, 0)
                
                # 创建目标变量：未来1天的收益率是否为正
                data['Target_1d'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
                
                # 移除NaN值
                clean_data = data.dropna()
                
                if len(clean_data) < 100:
                    print(f"  {symbol}: 数据不足以训练模型，跳过")
                    continue
                
                # 检查特征是否都存在
                missing_features = [f for f in features if f not in clean_data.columns]
                if missing_features:
                    print(f"  {symbol}: 缺少特征 {missing_features}，跳过")
                    continue
                
                # 分割训练集和测试集
                split_idx = int(len(clean_data) * 0.7)
                
                # 训练5天预测模型
                X = clean_data[features].values
                y = clean_data['Target_5d'].values
                
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                model_5d = RandomForestClassifier(n_estimators=100, random_state=42)
                model_5d.fit(X_train, y_train)
                
                # 评估模型
                y_pred = model_5d.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                
                print(f"  {symbol} 5天预测模型: 准确率={accuracy:.2%}, 精确率={precision:.2%}")
                
                # 训练1天预测模型
                y = clean_data['Target_1d'].values
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                model_1d = RandomForestClassifier(n_estimators=100, random_state=42)
                model_1d.fit(X_train, y_train)
                
                # 评估模型
                y_pred = model_1d.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                
                print(f"  {symbol} 1天预测模型: 准确率={accuracy:.2%}, 精确率={precision:.2%}")
                
                # 存储模型
                self.models[f"{symbol}_5d"] = model_5d
                self.models[f"{symbol}_1d"] = model_1d
                
            except Exception as e:
                print(f"  {symbol}: 训练模型失败 - {str(e)}")
        
        return self.models
    
    def generate_signals(self):
        """生成交易信号"""
        print("生成交易信号...")
        
        if not self.data:
            print("没有可用数据，跳过信号生成")
            return self.signals
        
        for symbol, data in self.data.items():
            if '_hourly' in symbol:
                continue  # 暂时跳过小时数据
                
            try:
                # 创建信号DataFrame
                signals = pd.DataFrame(index=data.index)
                signals['Close'] = data['Close']
                signals['Daily_Return'] = data['Daily_Return']
                
                # 检查必要的技术指标是否存在
                required_indicators = ['Momentum', 'SMA20', 'SMA50', 'BB_Lower', 'BB_Upper', 'RSI', 'MACD']
                missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
                
                if missing_indicators:
                    print(f"  {symbol}: 缺少指标 {missing_indicators}，跳过")
                    continue
                
                # 1. 动量信号
                signals['Momentum_Signal'] = 0
                signals.loc[data['Momentum'] > 0, 'Momentum_Signal'] = 1
                signals.loc[data['Momentum'] < 0, 'Momentum_Signal'] = -1
                
                # 2. 趋势跟踪信号
                signals['Trend_Signal'] = 0
                signals.loc[data['SMA20'] > data['SMA50'], 'Trend_Signal'] = 1
                signals.loc[data['SMA20'] < data['SMA50'], 'Trend_Signal'] = -1
                
                # 3. 布林带信号(均值回归)
                signals['BB_Signal'] = 0
                signals.loc[data['Close'] < data['BB_Lower'], 'BB_Signal'] = 1  # 超卖
                signals.loc[data['Close'] > data['BB_Upper'], 'BB_Signal'] = -1  # 超买
                
                # 4. RSI信号
                signals['RSI_Signal'] = 0
                signals.loc[data['RSI'] < 30, 'RSI_Signal'] = 1  # 超卖
                signals.loc[data['RSI'] > 70, 'RSI_Signal'] = -1  # 超买
                
                # 5. MACD信号
                signals['MACD_Signal'] = 0
                signals.loc[data['MACD'] > 0, 'MACD_Signal'] = 1
                signals.loc[data['MACD'] < 0, 'MACD_Signal'] = -1
                
                # 6. ADX信号(趋势强度)
                if 'ADX' in data.columns:
                    signals['ADX_Strong_Trend'] = data['ADX'] > 25
                
                # 7. 成交量信号(如果有)
                if 'Volume_Ratio' in data.columns:
                    signals['Volume_Signal'] = 0
                    signals.loc[data['Volume_Ratio'] > 1.5, 'Volume_Signal'] = 1  # 放量
                
                # 8. 机器学习信号(如果有训练好的模型)
                if f"{symbol}_1d" in self.models and f"{symbol}_5d" in self.models:
                    # 准备特征
                    features = [
                        'RSI', 'MACD', 'Momentum', 'ADX', 
                        'BB_Width', 'ATR',
                        'SMA20', 'SMA50', 'SMA200'
                    ]
                    
                    # 添加成交量特征(如果有)
                    if 'Volume_Ratio' in data.columns:
                        features.append('Volume_Ratio')
                    
                    # 移除NaN值
                    clean_data = data.dropna()
                    
                    # 检查特征是否都存在
                    missing_features = [f for f in features if f not in clean_data.columns]
                    if not missing_features:
                        # 使用模型预测
                        X = clean_data[features].values
                        
                        # 1天预测
                        model_1d = self.models[f"{symbol}_1d"]
                        signals.loc[clean_data.index, 'ML_Signal_1d'] = model_1d.predict(X)
                        
                        # 5天预测
                        model_5d = self.models[f"{symbol}_5d"]
                        signals.loc[clean_data.index, 'ML_Signal_5d'] = model_5d.predict(X)
                
                # 组合信号
                signals['Combined_Signal'] = (
                    0.25 * signals['Momentum_Signal'] + 
                    0.25 * signals['Trend_Signal'] + 
                    0.15 * signals['BB_Signal'] + 
                    0.15 * signals['RSI_Signal'] + 
                    0.20 * signals['MACD_Signal']
                )
                
                # 如果有机器学习信号，加入组合
                if 'ML_Signal_1d' in signals.columns and 'ML_Signal_5d' in signals.columns:
                    # 将0/1转换为-1/1
                    ml_signal_1d = signals['ML_Signal_1d'] * 2 - 1
                    ml_signal_5d = signals['ML_Signal_5d'] * 2 - 1
                    
                    # 重新计算组合信号，加入机器学习
                    signals['Combined_Signal'] = (
                        0.15 * signals['Momentum_Signal'] + 
                        0.15 * signals['Trend_Signal'] + 
                        0.10 * signals['BB_Signal'] + 
                        0.10 * signals['RSI_Signal'] + 
                        0.15 * signals['MACD_Signal'] +
                        0.20 * ml_signal_1d +
                        0.15 * ml_signal_5d
                    )
                
                # 最终信号：大于0为买入，小于0为卖出
                signals['Final_Signal'] = np.sign(signals['Combined_Signal'])
                
                # 存储信号
                self.signals[symbol] = signals
                
                print(f"  {symbol}: 生成了 {len(signals)} 条信号")
                
            except Exception as e:
                print(f"  {symbol}: 生成信号失败 - {str(e)}")
        
        return self.signals
    
    def calculate_position_sizes(self):
        """计算仓位大小"""
        print("计算仓位大小...")
        
        if not self.signals:
            print("没有可用信号，跳过仓位计算")
            return self.positions
        
        # 计算每个资产的波动率
        volatilities = {}
        for symbol, data in self.data.items():
            if '_hourly' in symbol or symbol not in self.signals:
                continue
            volatilities[symbol] = data['Daily_Return'].std() * np.sqrt(252)
        
        if not volatilities:
            print("没有可用的波动率数据，跳过仓位计算")
            return self.positions
        
        # 计算风险平价权重
        total_inverse_vol = sum(1/vol for vol in volatilities.values() if vol > 0)
        if total_inverse_vol == 0:
            print("波动率计算异常，使用均等权重")
            weights = {symbol: 1.0/len(volatilities) for symbol in volatilities}
        else:
            weights = {symbol: (1/vol)/total_inverse_vol if vol > 0 else 0 
                      for symbol, vol in volatilities.items()}
        
        print("资产权重分配:")
        for symbol, weight in weights.items():
            print(f"  {symbol}: {weight:.2%}")
        
        # 计算每个资产的仓位
        for symbol, signals in self.signals.items():
            if symbol not in weights:
                continue
                
            # 初始化仓位列
            signals['Position'] = 0
            
            # 根据信号和权重计算基础仓位
            signals['Position'] = signals['Final_Signal'] * weights[symbol]
            
            # 应用自适应杠杆
            signals['Leverage'] = self._calculate_adaptive_leverage(signals)
            signals['Position'] = signals['Position'] * signals['Leverage']
            
            # 应用止损
            signals['Stop_Loss'] = self._apply_stop_loss(signals)
            signals['Position'] = signals['Position'] * signals['Stop_Loss']
            
            # 存储仓位
            self.positions[symbol] = signals['Position']
            
            print(f"  {symbol}: 计算了 {len(signals)} 条仓位数据")
        
        return self.positions
    
    def _calculate_adaptive_leverage(self, signals):
        """计算自适应杠杆"""
        # 基于信号强度和市场波动性调整杠杆
        leverage = np.ones(len(signals))
        
        # 信号强度越强，杠杆越大
        if 'Combined_Signal' in signals.columns:
            signal_strength = abs(signals['Combined_Signal'])
            max_signal = signal_strength.max()
            if max_signal > 0:
                normalized_strength = signal_strength / max_signal
                leverage = 1 + normalized_strength * (self.max_leverage - 1)
        
        # 限制最大杠杆
        leverage = np.minimum(leverage, self.max_leverage)
        
        # 当信号为0时，杠杆为0
        if 'Final_Signal' in signals.columns:
            leverage[signals['Final_Signal'] == 0] = 0
        
        return leverage
    
    def _apply_stop_loss(self, signals):
        """应用止损策略"""
        # 初始化为1(不触发止损)
        stop_loss = np.ones(len(signals))
        
        if 'Daily_Return' in signals.columns and 'Position' in signals.columns:
            # 计算累积收益
            strategy_returns = signals['Daily_Return'] * signals['Position'].shift(1)
            strategy_returns = strategy_returns.fillna(0)  # 填充NaN值
            
            if len(strategy_returns) > 0:
                cumulative_return = (1 + strategy_returns).cumprod()
                
                # 计算回撤
                peak = cumulative_return.cummax()
                drawdown = (cumulative_return - peak) / peak
                
                # 当回撤超过风险容忍度时触发止损
                stop_loss[drawdown < -self.risk_tolerance] = 0
        
        return stop_loss
    
    def backtest(self):
        """回测策略"""
        print("开始回测策略...")
        
        if not self.positions:
            print("没有可用仓位数据，跳过回测")
            return {
                'annual_return': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'results': pd.DataFrame()
            }
        
        # 初始化回测结果
        results = pd.DataFrame()
        
        # 计算每个资产的收益
        asset_returns = {}
        for symbol, position in self.positions.items():
            if symbol not in self.signals:
                continue
                
            signals = self.signals[symbol]
            
            # 计算策略收益
            if 'Daily_Return' in signals.columns:
                strategy_return = signals['Daily_Return'] * position.shift(1)
                strategy_return = strategy_return.fillna(0)  # 填充NaN值
                asset_returns[symbol] = strategy_return
                
                print(f"  {symbol}: 年化收益率 = {self._calculate_annual_return(strategy_return):.2%}")
        
        if not asset_returns:
            print("没有可用的资产收益数据，跳过回测")
            return {
                'annual_return': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'results': pd.DataFrame()
            }
        
        # 合并所有资产的收益
        all_returns = pd.DataFrame(asset_returns)
        
        # 计算组合收益
        results['Portfolio_Return'] = all_returns.mean(axis=1)
        
        # 计算累积收益
        results['Cumulative_Return'] = (1 + results['Portfolio_Return']).cumprod()
        
        # 计算回撤
        results['Peak'] = results['Cumulative_Return'].cummax()
        results['Drawdown'] = (results['Cumulative_Return'] - results['Peak']) / results['Peak']
        
        # 计算胜率
        non_zero_returns = results[results['Portfolio_Return'] != 0]
        if len(non_zero_returns) > 0:
            win_rate = len(non_zero_returns[non_zero_returns['Portfolio_Return'] > 0]) / len(non_zero_returns)
        else:
            win_rate = 0
        
        # 计算年化收益率
        annual_return = self._calculate_annual_return(results['Portfolio_Return'])
        
        # 计算最大回撤
        max_drawdown = results['Drawdown'].min()
        
        # 计算夏普比率
        if results['Portfolio_Return'].std() > 0:
            sharpe_ratio = results['Portfolio_Return'].mean() / results['Portfolio_Return'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        print("\n回测结果:")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"胜率: {win_rate:.2%}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        
        # 绘制回测结果
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(results['Cumulative_Return'])
        plt.title('策略累积收益')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(results['Drawdown'])
        plt.title('策略回撤')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../data/backtest_results.png')
        
        # 保存回测结果
        results.to_csv('../data/backtest_results.csv')
        
        # 返回回测指标
        return {
            'annual_return': annual_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'results': results
        }
    
    def _calculate_annual_return(self, returns):
        """计算年化收益率"""
        if len(returns) == 0:
            return 0
            
        returns = returns.fillna(0)  # 填充NaN值
        cumulative_return = (1 + returns).prod()
        n_days = len(returns)
        if n_days > 0 and cumulative_return > 0:
            annual_return = cumulative_return ** (252 / n_days) - 1
            return annual_return
        return 0
    
    def optimize(self):
        """优化策略参数"""
        # 这里可以实现参数优化逻辑
        # 例如网格搜索最佳参数组合
        pass
    
    def save_strategy(self, filename='high_performance_strategy.py'):
        """保存策略代码"""
        import inspect
        
        # 获取当前类的源代码
        source = inspect.getsource(self.__class__)
        
        # 保存到文件
        with open(f'../strategy/{filename}', 'w') as f:
            f.write(source)
        
        print(f"策略代码已保存到 ../strategy/{filename}")
    
    def generate_trading_signals(self, live_data=None):
        """生成实时交易信号"""
        if live_data is None:
            # 使用最新数据
            live_data = {}
            for symbol in self.symbols:
                if symbol in self.data:
                    live_data[symbol] = self.data[symbol].iloc[-1:]
        
        # 生成信号逻辑与回测相同
        # 这里可以实现实时信号生成
        pass
    
    def run_strategy(self):
        """运行完整策略流程"""
        # 1. 获取数据
        self.fetch_data()
        
        # 2. 训练机器学习模型
        self.train_ml_models()
        
        # 3. 生成交易信号
        self.generate_signals()
        
        # 4. 计算仓位大小
        self.calculate_position_sizes()
        
        # 5. 回测策略
        results = self.backtest()
        
        # 6. 保存策略
        self.save_strategy()
        
        return results

# 运行策略
if __name__ == "__main__":
    # 创建策略实例
    strategy = HighPerformanceStrategy(
        symbols=['NVDA', 'BTC-USD', 'ETH-USD', 'TSLA'],
        max_leverage=3,
        risk_tolerance=0.02
    )
    
    # 运行策略
    results = strategy.run_strategy()
    
    # 输出结果
    print("\n策略执行完成!")
    print(f"年化收益率: {results['annual_return']:.2%}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
