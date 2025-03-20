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
os.makedirs('../docs', exist_ok=True)

# 设置时间范围
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)  # 获取3年数据

# 基于市场分析选择表现最好的资产
assets = ['NVDA', 'GC=F', 'BTC-USD']

# 创建结果DataFrame
strategy_results = pd.DataFrame(columns=['资产', '策略类型', '年化收益率', '胜率', '最大回撤', '夏普比率'])

def calculate_metrics(returns, positions):
    """计算策略绩效指标"""
    strategy_returns = returns * positions.shift(1)
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # 计算胜率
    win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns[strategy_returns != 0])
    
    # 计算年化收益率
    days = (returns.index[-1] - returns.index[0]).days
    annual_return = (cumulative_returns.iloc[-1] ** (365 / days)) - 1
    
    # 计算最大回撤
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # 计算夏普比率
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    
    return annual_return, win_rate, max_drawdown, sharpe_ratio, cumulative_returns

def momentum_strategy(data, window=20):
    """动量策略"""
    # 计算过去window天的收益率
    data['momentum'] = data['Close'].pct_change(window)
    
    # 生成交易信号：momentum > 0为买入信号，momentum < 0为卖出信号
    data['position'] = np.where(data['momentum'] > 0, 1, -1)
    
    # 移除NaN值
    data = data.dropna()
    
    return data['Daily_Return'], data['position']

def trend_following_strategy(data, short_window=20, long_window=50):
    """趋势跟踪策略"""
    # 计算短期和长期移动平均线
    data['short_ma'] = data['Close'].rolling(window=short_window).mean()
    data['long_ma'] = data['Close'].rolling(window=long_window).mean()
    
    # 生成交易信号：短期均线上穿长期均线为买入信号，下穿为卖出信号
    data['position'] = np.where(data['short_ma'] > data['long_ma'], 1, -1)
    
    # 移除NaN值
    data = data.dropna()
    
    return data['Daily_Return'], data['position']

def mean_reversion_strategy(data, window=20, std_dev=2):
    """均值回归策略"""
    # 计算移动平均线和标准差
    data['ma'] = data['Close'].rolling(window=window).mean()
    data['std'] = data['Close'].rolling(window=window).std()
    
    # 计算上下轨
    data['upper_band'] = data['ma'] + (data['std'] * std_dev)
    data['lower_band'] = data['ma'] - (data['std'] * std_dev)
    
    # 生成交易信号：价格低于下轨为买入信号，高于上轨为卖出信号
    data['position'] = 0
    data.loc[data['Close'] < data['lower_band'], 'position'] = 1
    data.loc[data['Close'] > data['upper_band'], 'position'] = -1
    
    # 填充0的位置为前一个信号
    data['position'] = data['position'].replace(to_replace=0, method='ffill')
    
    # 移除NaN值
    data = data.dropna()
    
    return data['Daily_Return'], data['position']

def machine_learning_strategy(data):
    """机器学习策略"""
    # 添加技术指标作为特征
    data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['macd'] = ta.trend.MACD(data['Close']).macd()
    data['macd_signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['macd_diff'] = ta.trend.MACD(data['Close']).macd_diff()
    data['bb_high'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
    data['bb_low'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
    data['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    
    # 创建目标变量：未来5天的收益率是否为正
    data['target'] = np.where(data['Close'].shift(-5) > data['Close'], 1, 0)
    
    # 选择特征和目标
    features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'atr']
    X = data[features]
    y = data['target']
    
    # 移除NaN值
    X = X.dropna()
    y = y.loc[X.index]
    
    # 分割训练集和测试集
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"机器学习模型准确率: {accuracy:.2%}, 精确率: {precision:.2%}, 召回率: {recall:.2%}")
    
    # 生成交易信号
    data.loc[X_test.index, 'ml_signal'] = y_pred
    data['position'] = np.where(data['ml_signal'] == 1, 1, -1)
    
    # 只考虑测试集部分
    test_data = data.loc[X_test.index]
    
    return test_data['Daily_Return'], test_data['position']

def combined_strategy(data):
    """组合策略"""
    # 添加技术指标
    data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['macd_diff'] = ta.trend.MACD(data['Close']).macd_diff()
    data['bb_high'] = ta.volatility.BollingerBands(data['Close']).bollinger_hband()
    data['bb_low'] = ta.volatility.BollingerBands(data['Close']).bollinger_lband()
    
    # 计算短期和长期移动平均线
    data['short_ma'] = data['Close'].rolling(window=20).mean()
    data['long_ma'] = data['Close'].rolling(window=50).mean()
    
    # 计算动量指标
    data['momentum'] = data['Close'].pct_change(20)
    
    # 生成各个策略的信号
    data['trend_signal'] = np.where(data['short_ma'] > data['long_ma'], 1, -1)
    data['momentum_signal'] = np.where(data['momentum'] > 0, 1, -1)
    data['mean_reversion_signal'] = 0
    data.loc[data['Close'] < data['bb_low'], 'mean_reversion_signal'] = 1
    data.loc[data['Close'] > data['bb_high'], 'mean_reversion_signal'] = -1
    data['mean_reversion_signal'] = data['mean_reversion_signal'].replace(to_replace=0, method='ffill')
    
    # 组合信号：使用加权投票
    data['combined_signal'] = (
        0.4 * data['trend_signal'] + 
        0.4 * data['momentum_signal'] + 
        0.2 * data['mean_reversion_signal']
    )
    
    # 最终信号：combined_signal > 0为买入，否则为卖出
    data['position'] = np.where(data['combined_signal'] > 0, 1, -1)
    
    # 移除NaN值
    data = data.dropna()
    
    return data['Daily_Return'], data['position']

def leveraged_strategy(data, leverage=2):
    """杠杆策略"""
    # 使用组合策略生成基础信号
    returns, positions = combined_strategy(data)
    
    # 应用杠杆
    leveraged_positions = positions * leverage
    
    return returns, leveraged_positions

# 分析每个资产的不同策略
for asset in assets:
    print(f"\n分析 {asset} 的不同策略...")
    
    try:
        # 获取数据
        data = yf.download(asset, start=start_date, end=end_date)
        
        if len(data) < 100:  # 确保有足够的数据
            print(f"  {asset}: 数据不足，跳过")
            continue
            
        # 计算每日回报
        data['Daily_Return'] = data['Close'].pct_change()
        
        # 移除NaN值
        data = data.dropna()
        
        # 测试不同策略
        strategies = {
            '动量策略': momentum_strategy,
            '趋势跟踪策略': trend_following_strategy,
            '均值回归策略': mean_reversion_strategy,
            '机器学习策略': machine_learning_strategy,
            '组合策略': combined_strategy,
            '杠杆策略(2倍)': lambda d: leveraged_strategy(d, 2),
            '杠杆策略(3倍)': lambda d: leveraged_strategy(d, 3)
        }
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='价格')
        plt.title(f'{asset} 价格走势')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        
        # 测试每种策略
        for strategy_name, strategy_func in strategies.items():
            try:
                returns, positions = strategy_func(data.copy())
                annual_return, win_rate, max_drawdown, sharpe_ratio, cumulative_returns = calculate_metrics(returns, positions)
                
                # 添加到结果
                strategy_results = pd.concat([strategy_results, pd.DataFrame({
                    '资产': [asset],
                    '策略类型': [strategy_name],
                    '年化收益率': [annual_return],
                    '胜率': [win_rate],
                    '最大回撤': [max_drawdown],
                    '夏普比率': [sharpe_ratio]
                })], ignore_index=True)
                
                # 绘制累积收益曲线
                plt.plot(cumulative_returns.index, cumulative_returns, label=f'{strategy_name} (年化: {annual_return:.2%}, 胜率: {win_rate:.2%})')
                
                print(f"  {strategy_name}: 年化收益率 = {annual_return:.2%}, 胜率 = {win_rate:.2%}, 最大回撤 = {max_drawdown:.2%}, 夏普比率 = {sharpe_ratio:.2f}")
                
            except Exception as e:
                print(f"  {strategy_name} 分析失败: {str(e)}")
        
        plt.title(f'{asset} 不同策略的累积收益')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'../data/{asset}_strategies.png')
        plt.close()
        
    except Exception as e:
        print(f"  {asset} 整体分析失败: {str(e)}")

# 保存结果
strategy_results.to_csv('../data/strategy_research_results.csv', index=False)

# 打印高收益率策略
high_return_strategies = strategy_results[strategy_results['年化收益率'] > 1.0].sort_values('年化收益率', ascending=False)
print("\n潜在高收益率策略:")
print(high_return_strategies)

# 打印高胜率策略
high_winrate_strategies = strategy_results[strategy_results['胜率'] > 0.68].sort_values('胜率', ascending=False)
print("\n潜在高胜率策略:")
print(high_winrate_strategies)

# 生成报告
with open('../docs/strategy_research_report.md', 'w') as f:
    f.write("# 量化交易策略研究报告\n\n")
    f.write(f"分析日期: {datetime.now().strftime('%Y-%m-%d')}\n\n")
    
    f.write("## 策略概览\n\n")
    f.write("| 资产 | 策略类型 | 年化收益率 | 胜率 | 最大回撤 | 夏普比率 |\n")
    f.write("|------|----------|------------|------|----------|----------|\n")
    
    for _, row in strategy_results.iterrows():
        f.write(f"| {row['资产']} | {row['策略类型']} | {row['年化收益率']:.2%} | {row['胜率']:.2%} | {row['最大回撤']:.2%} | {row['夏普比率']:.2f} |\n")
    
    f.write("\n## 高收益率策略\n\n")
    if len(high_return_strategies) > 0:
        f.write("| 资产 | 策略类型 | 年化收益率 | 胜率 | 最大回撤 | 夏普比率 |\n")
        f.write("|------|----------|------------|------|----------|----------|\n")
        
        for _, row in high_return_strategies.iterrows():
            f.write(f"| {row['资产']} | {row['策略类型']} | {row['年化收益率']:.2%} | {row['胜率']:.2%} | {row['最大回撤']:.2%} | {row['夏普比率']:.2f} |\n")
    else:
        f.write("没有发现年化收益率超过100%的策略。\n")
    
    f.write("\n## 高胜率策略\n\n")
    if len(high_winrate_strategies) > 0:
        f.write("| 资产 | 策略类型 | 年化收益率 | 胜率 | 最大回撤 | 夏普比率 |\n")
        f.write("|------|----------|------------|------|----------|----------|\n")
        
        for _, row in high_winrate_strategies.iterrows():
            f.write(f"| {row['资产']} | {row['策略类型']} | {row['年化收益率']:.2%} | {row['胜率']:.2%} | {row['最大回撤']:.2%} | {row['夏普比率']:.2f} |\n")
    else:
        f.write("没有发现胜率超过68%的策略。\n")
    
    f.write("\n## 策略说明\n\n")
    f.write("### 动量策略\n")
    f.write("动量策略基于价格趋势的延续性，当价格呈现上升趋势时买入，下降趋势时卖出。\n")
    f.write("具体实现是通过计算过去20天的价格变化率，正值表示上升趋势，负值表示下降趋势。\n\n")
    
    f.write("### 趋势跟踪策略\n")
    f.write("趋势跟踪策略使用短期和长期移动平均线的交叉来识别趋势。\n")
    f.write("当短期均线(20日)上穿长期均线(50日)时买入，下穿时卖出。\n\n")
    
    f.write("### 均值回归策略\n")
    f.write("均值回归策略基于价格围绕均值波动的特性，当价格远离均值时预期会回归。\n")
    f.write("使用布林带指标，当价格低于下轨时买入，高于上轨时卖出。\n\n")
    
    f.write("### 机器学习策略\n")
    f.write("机器学习策略使用随机森林算法，基于多种技术指标预测未来价格走势。\n")
    f.write("模型训练使用历史数据的70%，并在剩余30%上进行测试和交易。\n\n")
    
    f.write("### 组合策略\n")
    f.write("组合策略整合了动量、趋势跟踪和均值回归三种策略的信号，通过加权投票方式生成最终交易决策。\n")
    f.write("权重分配为：趋势跟踪40%，动量40%，均值回归20%。\n\n")
    
    f.write("### 杠杆策略\n")
    f.write("杠杆策略基于组合策略，但使用2倍或3倍杠杆来放大收益(同时也放大风险)。\n")
    f.write("这种策略适合风险承受能力较高的投资者，需要更严格的风险管理。\n\n")
    
    f.write("## 结论与建议\n\n")
    f.write("基于以上分析，我们可以得出以下结论：\n\n")
    
    if len(high_return_strategies) > 0:
        f.write("1. 最高收益率策略是 ")
        top_return = high_return_strategies.iloc[0]
        f.write(f"{top_return['资产']} 的 {top_return['策略类型']}，年化收益率达到 {top_return['年化收益率']:.2%}。\n")
    else:
        f.write("1. 没有策略能达到100%的年化收益率，最高的是 ")
        top_return = strategy_results.sort_values('年化收益率', ascending=False).iloc[0]
        f.write(f"{top_return['资产']} 的 {top_return['策略类型']}，年化收益率为 {top_return['年化收益率']:.2%}。\n")
    
    if len(high_winrate_strategies) > 0:
        f.write("2. 最高胜率策略是 ")
        top_winrate = high_winrate_strategies.iloc[0]
        f.write(f"{top_winrate['资产']} 的 {top_winrate['策略类型']}，胜率达到 {top_winrate['胜率']:.2%}。\n")
    else:
        f.write("2. 没有策略能达到68%的胜率，最高的是 ")
        top_winrate = strategy_results.sort_values('胜率', ascending=False).iloc[0]
        f.write(f"{top_winrate['资产']} 的 {top_winrate['策略类型']}，胜率为 {top_winrate['胜率']:.2%}。\n")
    
    f.write("3. 要实现年化收益率100%和胜率68%的目标，建议：\n")
    f.write("   - 使用杠杆策略，但需要严格的风险管理\n")
    f.write("   - 组合多种策略以提高稳定性\n")
    f.write("   - 专注于波动性较高的资产，如NVDA和加密货币\n")
    f.write("   - 考虑更短的交易周期，如日内交易或摆动交易\n")
    f.write("   - 实施止损策略以控制最大回撤\n\n")
    
    f.write("4. 下一步建议：\n")
    f.write("   - 进一步优化组合策略的参数\n")
    f.write("   - 测试更多的资产组合\n")
    f.write("   - 实现更复杂的机器学习模型\n")
    f.write("   - 开发自适应的风险管理系统\n")

print("\n策略研究完成，结果已保存到 ../data/strategy_research_results.csv 和 ../docs/strategy_research_report.md")
