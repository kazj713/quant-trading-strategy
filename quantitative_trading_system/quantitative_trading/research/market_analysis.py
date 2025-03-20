import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# 创建输出目录
os.makedirs('../data', exist_ok=True)
os.makedirs('../docs', exist_ok=True)

# 设置时间范围
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)  # 获取3年数据

# 定义要分析的市场和资产
markets = {
    '美股': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA'],
    '加密货币': ['BTC-USD', 'ETH-USD'],
    '商品': ['GC=F', 'CL=F'],  # 黄金和原油期货
    '外汇': ['EURUSD=X', 'USDJPY=X']
}

# 创建结果DataFrame
results = pd.DataFrame(columns=['市场', '资产', '年化收益率', '波动率', '最大回撤', '夏普比率'])

# 分析每个市场和资产
for market, assets in markets.items():
    print(f"分析 {market} 市场...")
    
    for asset in assets:
        try:
            # 获取数据
            data = yf.download(asset, start=start_date, end=end_date)
            
            if len(data) < 20:  # 确保有足够的数据
                print(f"  {asset}: 数据不足，跳过")
                continue
                
            # 检查数据列
            print(f"  {asset} 数据列: {data.columns.tolist()}")
            
            # 使用Close列代替Adj Close
            data['Daily_Return'] = data['Close'].pct_change()
            
            # 计算累积回报
            data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
            
            # 计算回撤
            data['Peak'] = data['Cumulative_Return'].cummax()
            data['Drawdown'] = (data['Cumulative_Return'] - data['Peak']) / data['Peak']
            
            # 计算指标
            days = (data.index[-1] - data.index[0]).days
            annual_return = (data['Cumulative_Return'].iloc[-1] ** (365 / days)) - 1
            volatility = data['Daily_Return'].std() * np.sqrt(252)
            max_drawdown = data['Drawdown'].min()
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            
            # 添加到结果
            results = pd.concat([results, pd.DataFrame({
                '市场': [market],
                '资产': [asset],
                '年化收益率': [annual_return],
                '波动率': [volatility],
                '最大回撤': [max_drawdown],
                '夏普比率': [sharpe_ratio]
            })], ignore_index=True)
            
            # 绘制价格图表
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(data['Close'])
            plt.title(f'{asset} 价格走势')
            plt.grid(True)
            
            # 绘制回撤图表
            plt.subplot(2, 1, 2)
            plt.plot(data['Drawdown'])
            plt.title(f'{asset} 回撤')
            plt.grid(True)
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(f'../data/{asset}_analysis.png')
            plt.close()
            
            print(f"  {asset}: 年化收益率 = {annual_return:.2%}, 波动率 = {volatility:.2%}, 最大回撤 = {max_drawdown:.2%}, 夏普比率 = {sharpe_ratio:.2f}")
            
        except Exception as e:
            print(f"  {asset}: 分析失败 - {str(e)}")

# 保存结果
results.to_csv('../data/market_analysis_results.csv', index=False)

# 打印高收益率资产
high_return_assets = results[results['年化收益率'] > 0.5].sort_values('年化收益率', ascending=False)
print("\n潜在高收益率资产:")
print(high_return_assets)

# 打印高夏普比率资产
high_sharpe_assets = results[results['夏普比率'] > 1].sort_values('夏普比率', ascending=False)
print("\n潜在高夏普比率资产:")
print(high_sharpe_assets)

# 生成报告
with open('../docs/market_analysis_report.md', 'w') as f:
    f.write("# 市场分析报告\n\n")
    f.write(f"分析日期: {datetime.now().strftime('%Y-%m-%d')}\n\n")
    
    f.write("## 市场概览\n\n")
    f.write("| 市场 | 资产 | 年化收益率 | 波动率 | 最大回撤 | 夏普比率 |\n")
    f.write("|------|------|------------|--------|----------|----------|\n")
    
    for _, row in results.iterrows():
        f.write(f"| {row['市场']} | {row['资产']} | {row['年化收益率']:.2%} | {row['波动率']:.2%} | {row['最大回撤']:.2%} | {row['夏普比率']:.2f} |\n")
    
    f.write("\n## 高收益率资产\n\n")
    if len(high_return_assets) > 0:
        f.write("| 市场 | 资产 | 年化收益率 | 波动率 | 最大回撤 | 夏普比率 |\n")
        f.write("|------|------|------------|--------|----------|----------|\n")
        
        for _, row in high_return_assets.iterrows():
            f.write(f"| {row['市场']} | {row['资产']} | {row['年化收益率']:.2%} | {row['波动率']:.2%} | {row['最大回撤']:.2%} | {row['夏普比率']:.2f} |\n")
    else:
        f.write("没有发现年化收益率超过50%的资产。\n")
    
    f.write("\n## 高夏普比率资产\n\n")
    if len(high_sharpe_assets) > 0:
        f.write("| 市场 | 资产 | 年化收益率 | 波动率 | 最大回撤 | 夏普比率 |\n")
        f.write("|------|------|------------|--------|----------|----------|\n")
        
        for _, row in high_sharpe_assets.iterrows():
            f.write(f"| {row['市场']} | {row['资产']} | {row['年化收益率']:.2%} | {row['波动率']:.2%} | {row['最大回撤']:.2%} | {row['夏普比率']:.2f} |\n")
    else:
        f.write("没有发现夏普比率超过1的资产。\n")
    
    f.write("\n## 结论\n\n")
    f.write("基于以上分析，我们可以看出哪些市场和资产在过去表现最好，这将有助于我们选择适合的交易品种和市场。\n")
    f.write("然而，需要注意的是，过去的表现并不能保证未来的结果，我们需要结合更多的分析来设计一个稳健的交易策略。\n")

print("\n市场分析完成，结果已保存到 ../data/market_analysis_results.csv 和 ../docs/market_analysis_report.md")
