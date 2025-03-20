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
os.makedirs('../backtest', exist_ok=True)

class BacktestFramework:
    """
    量化交易策略回测框架
    
    功能:
    1. 数据获取和预处理
    2. 策略回测
    3. 性能评估
    4. 可视化分析
    """
    
    def __init__(self, symbols=['NVDA', 'BTC-USD'], 
                 start_date=None, 
                 end_date=None,
                 initial_capital=100000):
        """
        初始化回测框架
        
        参数:
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
        self.portfolio = None
        self.performance_metrics = {}
        
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
    
    def run_momentum_strategy(self, window=20):
        """运行动量策略"""
        print("运行动量策略...")
        
        for symbol, data in self.data.items():
            try:
                # 创建信号DataFrame
                signals = pd.DataFrame(index=data.index)
                signals['Close'] = data['Close']
                signals['Daily_Return'] = data['Daily_Return']
                
                # 计算动量指标
                signals['Momentum'] = data['Close'].pct_change(window)
                
                # 生成交易信号
                signals['Signal'] = 0
                signals.loc[signals['Momentum'] > 0, 'Signal'] = 1  # 买入信号
                signals.loc[signals['Momentum'] < 0, 'Signal'] = -1  # 卖出信号
                
                # 存储信号
                self.signals[f"{symbol}_momentum"] = signals
                
                print(f"  {symbol}: 生成了 {len(signals)} 条动量策略信号")
                
            except Exception as e:
                print(f"  {symbol}: 生成动量策略信号失败 - {str(e)}")
        
        return self.signals
    
    def run_trend_following_strategy(self, short_window=20, long_window=50):
        """运行趋势跟踪策略"""
        print("运行趋势跟踪策略...")
        
        for symbol, data in self.data.items():
            try:
                # 创建信号DataFrame
                signals = pd.DataFrame(index=data.index)
                signals['Close'] = data['Close']
                signals['Daily_Return'] = data['Daily_Return']
                
                # 计算短期和长期移动平均线
                signals['Short_MA'] = data['SMA20']
                signals['Long_MA'] = data['SMA50']
                
                # 生成交易信号
                signals['Signal'] = 0
                signals.loc[signals['Short_MA'] > signals['Long_MA'], 'Signal'] = 1  # 买入信号
                signals.loc[signals['Short_MA'] < signals['Long_MA'], 'Signal'] = -1  # 卖出信号
                
                # 存储信号
                self.signals[f"{symbol}_trend"] = signals
                
                print(f"  {symbol}: 生成了 {len(signals)} 条趋势跟踪策略信号")
                
            except Exception as e:
                print(f"  {symbol}: 生成趋势跟踪策略信号失败 - {str(e)}")
        
        return self.signals
    
    def run_mean_reversion_strategy(self):
        """运行均值回归策略"""
        print("运行均值回归策略...")
        
        for symbol, data in self.data.items():
            try:
                # 创建信号DataFrame
                signals = pd.DataFrame(index=data.index)
                signals['Close'] = data['Close']
                signals['Daily_Return'] = data['Daily_Return']
                
                # 使用布林带
                signals['BB_Upper'] = data['BB_Upper']
                signals['BB_Lower'] = data['BB_Lower']
                
                # 生成交易信号
                signals['Signal'] = 0
                signals.loc[signals['Close'] < signals['BB_Lower'], 'Signal'] = 1  # 买入信号(超卖)
                signals.loc[signals['Close'] > signals['BB_Upper'], 'Signal'] = -1  # 卖出信号(超买)
                
                # 填充0值(保持前一个信号)
                # 使用前向填充代替已弃用的method参数
                signals['Signal'] = signals['Signal'].replace(to_replace=0)
                signals['Signal'] = signals['Signal'].fillna(method='ffill')
                
                # 存储信号
                self.signals[f"{symbol}_mean_reversion"] = signals
                
                print(f"  {symbol}: 生成了 {len(signals)} 条均值回归策略信号")
                
            except Exception as e:
                print(f"  {symbol}: 生成均值回归策略信号失败 - {str(e)}")
        
        return self.signals
    
    def run_combined_strategy(self, weights={'momentum': 0.4, 'trend': 0.4, 'mean_reversion': 0.2}):
        """运行组合策略"""
        print("运行组合策略...")
        
        for symbol in self.symbols:
            try:
                # 检查是否有所有需要的策略信号
                momentum_key = f"{symbol}_momentum"
                trend_key = f"{symbol}_trend"
                mean_reversion_key = f"{symbol}_mean_reversion"
                
                if momentum_key not in self.signals or trend_key not in self.signals or mean_reversion_key not in self.signals:
                    print(f"  {symbol}: 缺少某些策略信号，跳过")
                    continue
                
                # 获取各策略信号
                momentum_signals = self.signals[momentum_key]
                trend_signals = self.signals[trend_key]
                mean_reversion_signals = self.signals[mean_reversion_key]
                
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
                    weights['momentum'] * combined['Momentum_Signal'] +
                    weights['trend'] * combined['Trend_Signal'] +
                    weights['mean_reversion'] * combined['Mean_Reversion_Signal']
                )
                
                # 最终信号：大于0为买入，小于0为卖出
                combined['Signal'] = np.sign(combined['Combined_Signal'])
                
                # 存储信号
                self.signals[f"{symbol}_combined"] = combined
                
                print(f"  {symbol}: 生成了 {len(combined)} 条组合策略信号")
                
            except Exception as e:
                print(f"  {symbol}: 生成组合策略信号失败 - {str(e)}")
        
        return self.signals
    
    def run_leveraged_strategy(self, base_strategy_key, leverage=3):
        """运行杠杆策略"""
        print(f"运行{leverage}倍杠杆策略...")
        
        # 创建信号字典的副本，避免在迭代过程中修改字典
        signals_copy = dict(self.signals)
        
        for key, signals in signals_copy.items():
            if base_strategy_key in key:
                try:
                    # 创建杠杆策略信号
                    leveraged = signals.copy()
                    
                    # 应用杠杆
                    leveraged['Leveraged_Signal'] = leveraged['Signal'] * leverage
                    
                    # 存储信号
                    self.signals[f"{key}_leveraged_{leverage}x"] = leveraged
                    
                    print(f"  {key}: 生成了 {len(leveraged)} 条{leverage}倍杠杆策略信号")
                    
                except Exception as e:
                    print(f"  {key}: 生成杠杆策略信号失败 - {str(e)}")
        
        return self.signals
    
    def calculate_positions(self, strategy_key, capital_allocation=1.0):
        """计算策略仓位"""
        print(f"计算 {strategy_key} 策略仓位...")
        
        # 创建信号字典的副本，避免在迭代过程中修改字典
        signals_copy = dict(self.signals)
        
        for key, signals in signals_copy.items():
            if strategy_key in key:
                try:
                    # 创建仓位DataFrame
                    position = pd.DataFrame(index=signals.index)
                    position['Close'] = signals['Close']
                    
                    # 计算仓位
                    if 'Leveraged_Signal' in signals.columns:
                        position['Position'] = signals['Leveraged_Signal'] * capital_allocation
                    else:
                        position['Position'] = signals['Signal'] * capital_allocation
                    
                    # 存储仓位
                    self.positions[key] = position
                    
                    print(f"  {key}: 计算了 {len(position)} 条仓位数据")
                    
                except Exception as e:
                    print(f"  {key}: 计算仓位失败 - {str(e)}")
        
        return self.positions
    
    def backtest_strategy(self, strategy_key):
        """回测特定策略"""
        print(f"回测 {strategy_key} 策略...")
        
        results = {}
        
        # 创建仓位字典的副本，避免在迭代过程中修改字典
        positions_copy = dict(self.positions)
        
        for key, position in positions_copy.items():
            if strategy_key in key:
                try:
                    # 获取对应的信号数据
                    if key not in self.signals:
                        print(f"  {key}: 没有找到对应的信号数据，跳过")
                        continue
                        
                    signals = self.signals[key]
                    
                    # 创建回测结果DataFrame
                    result = pd.DataFrame(index=position.index)
                    result['Close'] = position['Close']
                    result['Position'] = position['Position']
                    result['Market_Return'] = signals['Daily_Return']
                    
                    # 计算策略收益
                    result['Strategy_Return'] = result['Market_Return'] * result['Position'].shift(1)
                    
                    # 填充NaN值
                    result['Strategy_Return'] = result['Strategy_Return'].fillna(0)
                    
                    # 计算累积收益
                    result['Cumulative_Market_Return'] = (1 + result['Market_Return']).cumprod()
                    result['Cumulative_Strategy_Return'] = (1 + result['Strategy_Return']).cumprod()
                    
                    # 计算回撤
                    result['Peak'] = result['Cumulative_Strategy_Return'].cummax()
                    result['Drawdown'] = (result['Cumulative_Strategy_Return'] - result['Peak']) / result['Peak']
                    
                    # 存储结果
                    results[key] = result
                    
                    # 计算性能指标
                    metrics = self._calculate_performance_metrics(result)
                    self.performance_metrics[key] = metrics
                    
                    print(f"  {key}: 回测完成，年化收益率 = {metrics['annual_return']:.2%}, 胜率 = {metrics['win_rate']:.2%}")
                    
                except Exception as e:
                    print(f"  {key}: 回测失败 - {str(e)}")
        
        return results
    
    def _calculate_performance_metrics(self, result):
        """计算策略性能指标"""
        # 提取策略收益
        returns = result['Strategy_Return']
        
        # 计算胜率
        win_rate = len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        
        # 计算年化收益率
        total_return = result['Cumulative_Strategy_Return'].iloc[-1] - 1
        days = len(returns)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算最大回撤
        max_drawdown = result['Drawdown'].min()
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        excess_return = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # 计算索提诺比率
        downside_deviation = returns[returns < 0].std() * np.sqrt(252)
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # 计算卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def plot_equity_curve(self, strategy_keys=None):
        """绘制权益曲线"""
        plt.figure(figsize=(15, 10))
        
        # 如果没有指定策略，则绘制所有策略
        if strategy_keys is None:
            strategy_keys = list(self.performance_metrics.keys())
        
        # 确保strategy_keys是列表
        if not isinstance(strategy_keys, list):
            strategy_keys = [strategy_keys]
        
        # 绘制每个策略的权益曲线
        for key in strategy_keys:
            if key in self.performance_metrics:
                # 获取回测结果
                results_dict = self.backtest_strategy(key)
                result = results_dict.get(key)
                
                if result is not None:
                    metrics = self.performance_metrics[key]
                    plt.plot(
                        result.index, 
                        result['Cumulative_Strategy_Return'], 
                        label=f"{key} (年化: {metrics['annual_return']:.2%}, 胜率: {metrics['win_rate']:.2%})"
                    )
        
        # 绘制基准(市场)收益
        for key in strategy_keys:
            if key in self.performance_metrics:
                # 获取回测结果
                results_dict = self.backtest_strategy(key)
                result = results_dict.get(key)
                
                if result is not None:
                    plt.plot(
                        result.index, 
                        result['Cumulative_Market_Return'], 
                        'k--', 
                        label='Market'
                    )
                    break
        
        plt.title('策略权益曲线')
        plt.xlabel('日期')
        plt.ylabel('累积收益')
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        plt.savefig('../backtest/equity_curves.png')
        plt.close()
        
        print("权益曲线已保存到 ../backtest/equity_curves.png")
    
    def plot_drawdowns(self, strategy_keys=None):
        """绘制回撤曲线"""
        plt.figure(figsize=(15, 10))
        
        # 如果没有指定策略，则绘制所有策略
        if strategy_keys is None:
            strategy_keys = list(self.performance_metrics.keys())
        
        # 确保strategy_keys是列表
        if not isinstance(strategy_keys, list):
            strategy_keys = [strategy_keys]
        
        # 绘制每个策略的回撤曲线
        for key in strategy_keys:
            if key in self.performance_metrics:
                # 获取回测结果
                results_dict = self.backtest_strategy(key)
                result = results_dict.get(key)
                
                if result is not None:
                    metrics = self.performance_metrics[key]
                    plt.plot(
                        result.index, 
                        result['Drawdown'], 
                        label=f"{key} (最大回撤: {metrics['max_drawdown']:.2%})"
                    )
        
        plt.title('策略回撤曲线')
        plt.xlabel('日期')
        plt.ylabel('回撤')
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        plt.savefig('../backtest/drawdowns.png')
        plt.close()
        
        print("回撤曲线已保存到 ../backtest/drawdowns.png")
    
    def generate_performance_report(self):
        """生成性能报告"""
        report = "# 量化交易策略回测报告\n\n"
        report += f"回测期间: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}\n\n"
        
        # 添加策略性能概览
        report += "## 策略性能概览\n\n"
        report += "| 策略 | 年化收益率 | 胜率 | 最大回撤 | 夏普比率 | 索提诺比率 | 卡玛比率 |\n"
        report += "|------|------------|------|----------|----------|------------|----------|\n"
        
        for key, metrics in self.performance_metrics.items():
            report += f"| {key} | {metrics['annual_return']:.2%} | {metrics['win_rate']:.2%} | {metrics['max_drawdown']:.2%} | {metrics['sharpe_ratio']:.2f} | {metrics['sortino_ratio']:.2f} | {metrics['calmar_ratio']:.2f} |\n"
        
        # 添加高收益率策略
        high_return_strategies = {k: v for k, v in self.performance_metrics.items() if v['annual_return'] > 0.5}
        if high_return_strategies:
            report += "\n## 高收益率策略\n\n"
            report += "| 策略 | 年化收益率 | 胜率 | 最大回撤 | 夏普比率 |\n"
            report += "|------|------------|------|----------|----------|\n"
            
            for key, metrics in sorted(high_return_strategies.items(), key=lambda x: x[1]['annual_return'], reverse=True):
                report += f"| {key} | {metrics['annual_return']:.2%} | {metrics['win_rate']:.2%} | {metrics['max_drawdown']:.2%} | {metrics['sharpe_ratio']:.2f} |\n"
        
        # 添加高胜率策略
        high_winrate_strategies = {k: v for k, v in self.performance_metrics.items() if v['win_rate'] > 0.6}
        if high_winrate_strategies:
            report += "\n## 高胜率策略\n\n"
            report += "| 策略 | 年化收益率 | 胜率 | 最大回撤 | 夏普比率 |\n"
            report += "|------|------------|------|----------|----------|\n"
            
            for key, metrics in sorted(high_winrate_strategies.items(), key=lambda x: x[1]['win_rate'], reverse=True):
                report += f"| {key} | {metrics['annual_return']:.2%} | {metrics['win_rate']:.2%} | {metrics['max_drawdown']:.2%} | {metrics['sharpe_ratio']:.2f} |\n"
        
        # 添加结论和建议
        report += "\n## 结论与建议\n\n"
        
        # 找出最佳策略
        if self.performance_metrics:
            best_return_key = max(self.performance_metrics.items(), key=lambda x: x[1]['annual_return'])[0]
            best_winrate_key = max(self.performance_metrics.items(), key=lambda x: x[1]['win_rate'])[0]
            best_sharpe_key = max(self.performance_metrics.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
            
            report += f"1. 最高收益率策略是 {best_return_key}，年化收益率为 {self.performance_metrics[best_return_key]['annual_return']:.2%}。\n"
            report += f"2. 最高胜率策略是 {best_winrate_key}，胜率为 {self.performance_metrics[best_winrate_key]['win_rate']:.2%}。\n"
            report += f"3. 最高夏普比率策略是 {best_sharpe_key}，夏普比率为 {self.performance_metrics[best_sharpe_key]['sharpe_ratio']:.2f}。\n\n"
        
        # 添加对目标的评估
        report += "4. 关于目标年化收益率100%和胜率68%的评估：\n"
        
        # 检查是否有策略接近目标
        close_to_return_target = {k: v for k, v in self.performance_metrics.items() if v['annual_return'] > 0.7}
        close_to_winrate_target = {k: v for k, v in self.performance_metrics.items() if v['win_rate'] > 0.6}
        
        if close_to_return_target:
            report += "   - 有策略接近年化收益率100%的目标，可以通过进一步优化和增加杠杆来实现。\n"
        else:
            report += "   - 没有策略接近年化收益率100%的目标，需要更激进的策略组合或更高的杠杆。\n"
            
        if close_to_winrate_target:
            report += "   - 有策略接近胜率68%的目标，可以通过优化入场和出场条件来进一步提高。\n"
        else:
            report += "   - 没有策略接近胜率68%的目标，需要改进信号生成算法或结合机器学习模型。\n"
        
        # 添加改进建议
        report += "\n5. 改进建议：\n"
        report += "   - 尝试更高的杠杆倍数，但需要更严格的风险管理\n"
        report += "   - 结合多种策略的优势，创建更复杂的组合策略\n"
        report += "   - 考虑更短的交易周期，如日内交易\n"
        report += "   - 添加止损和止盈条件，提高胜率和风险调整后收益\n"
        report += "   - 探索更多资产类别和市场，寻找更多交易机会\n"
        
        # 保存报告
        with open('../backtest/performance_report.md', 'w') as f:
            f.write(report)
        
        print("性能报告已保存到 ../backtest/performance_report.md")
        
        return report
    
    def run_backtest(self):
        """运行完整回测流程"""
        # 1. 获取数据
        self.fetch_data()
        
        # 2. 运行各种策略
        self.run_momentum_strategy()
        self.run_trend_following_strategy()
        self.run_mean_reversion_strategy()
        self.run_combined_strategy()
        
        # 3. 添加杠杆策略
        self.run_leveraged_strategy('combined', leverage=2)
        self.run_leveraged_strategy('combined', leverage=3)
        
        # 4. 计算仓位
        for strategy_key in ['momentum', 'trend', 'mean_reversion', 'combined', 'leveraged']:
            self.calculate_positions(strategy_key)
        
        # 5. 回测策略
        for strategy_key in ['momentum', 'trend', 'mean_reversion', 'combined', 'leveraged']:
            self.backtest_strategy(strategy_key)
        
        # 6. 绘制图表
        self.plot_equity_curve()
        self.plot_drawdowns()
        
        # 7. 生成报告
        self.generate_performance_report()
        
        return self.performance_metrics

# 运行回测
if __name__ == "__main__":
    # 创建回测框架实例
    backtest = BacktestFramework(
        symbols=['NVDA', 'BTC-USD', 'ETH-USD', 'TSLA'],
        start_date=datetime.now() - timedelta(days=365*3),
        end_date=datetime.now(),
        initial_capital=100000
    )
    
    # 运行回测
    metrics = backtest.run_backtest()
    
    # 输出结果
    print("\n回测完成!")
    print("查看 ../backtest/performance_report.md 获取详细报告")
    print("查看 ../backtest/equity_curves.png 和 ../backtest/drawdowns.png 获取可视化结果")
