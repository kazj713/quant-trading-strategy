# 高性能量化交易策略系统

这是一个旨在实现年化收益率100%和胜率68%的高性能量化交易策略系统。系统采用多策略组合方法，结合动量策略、趋势跟踪策略和均值回归策略，并通过优化参数、杠杆倍数和止损条件来提高性能。

## 系统特点

- **多策略组合**：结合多种交易策略，在不同市场环境下保持稳定表现
- **自适应参数**：根据市场状况动态调整策略参数
- **风险管理**：严格的止损机制和资金管理规则
- **多资产配置**：分散投资于多个高波动性资产
- **杠杆优化**：根据资产特性和市场状况调整杠杆倍数
- **全自动化**：从数据获取到交易执行的全流程自动化

## 目录结构

```
quantitative_trading/
├── backtest/                # 回测框架
│   └── backtest_framework.py
├── config/                  # 配置文件
│   └── trading_config.json
├── data/                    # 数据存储
├── docs/                    # 文档
│   ├── strategy_documentation.md  # 策略说明文档
│   ├── backtest_report.md         # 回测报告
│   ├── deployment_guide.md        # 部署指南
│   └── user_manual.md             # 用户手册
├── logs/                    # 日志文件
├── optimization/            # 策略优化
│   └── strategy_optimizer.py
├── research/                # 研究分析
│   ├── market_analysis.py
│   └── strategy_research.py
├── strategy/                # 策略实现
│   └── high_performance_strategy.py
├── trading/                 # 交易系统
│   └── automated_trading_system.py
└── README.md                # 项目说明
```

## 快速开始

### 环境配置

1. 安装Python 3.8+
2. 安装依赖库：
   ```bash
   pip install pandas numpy matplotlib scikit-learn statsmodels yfinance backtrader alpaca-trade-api python-dotenv schedule ta
   ```

### 运行回测

```bash
cd quantitative_trading
python backtest/backtest_framework.py
```

### 运行策略优化

```bash
python optimization/strategy_optimizer.py
```

### 运行模拟交易

```bash
python trading/automated_trading_system.py
```

## 策略性能

基于历史数据的回测结果显示：

- **NVDA组合策略（2倍杠杆）**：
  - 年化收益率：75.88%
  - 胜率：50.07%
  - 最大回撤：约25%

- **整体投资组合**：
  - 年化收益率：约60-70%
  - 胜率：约50-55%
  - 最大回撤：约30%

## 文档说明

- **[策略说明文档](docs/strategy_documentation.md)**：详细介绍策略原理、参数和性能
- **[回测报告](docs/backtest_report.md)**：提供回测结果和分析
- **[部署指南](docs/deployment_guide.md)**：指导如何部署和运行系统
- **[用户手册](docs/user_manual.md)**：提供系统日常操作指南

## 注意事项

- 本系统涉及杠杆交易，具有较高风险，请根据个人风险承受能力谨慎使用
- 建议先在模拟环境中运行系统一段时间，确认系统稳定且策略表现符合预期后，再考虑使用实际资金
- 过去的表现不代表未来的结果，市场环境变化可能导致策略表现不如预期

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱：support@quanttrading.com
- 网站：www.quanttrading.com

## 致谢

感谢所有开源项目和数据提供者，他们的工作使本项目成为可能。
