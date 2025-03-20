# 量化交易策略部署指南

本指南详细介绍如何部署和运行高性能量化交易策略系统，包括环境配置、API设置、系统部署和监控等内容。

## 1. 系统要求

### 1.1 硬件要求

- **CPU**: 至少4核心处理器
- **内存**: 至少8GB RAM
- **存储**: 至少50GB可用空间
- **网络**: 稳定的高速互联网连接

### 1.2 软件要求

- **操作系统**: Linux (推荐Ubuntu 20.04+), macOS, 或Windows 10+
- **Python**: 3.8+
- **依赖库**: pandas, numpy, matplotlib, scikit-learn, ta, yfinance, alpaca-trade-api等

## 2. 环境配置

### 2.1 Python环境设置

1. **安装Python**:
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install python3 python3-pip python3-dev build-essential
   
   # macOS (使用Homebrew)
   brew install python
   
   # Windows
   # 从Python官网下载安装程序
   ```

2. **创建虚拟环境** (推荐):
   ```bash
   # 安装virtualenv
   pip3 install virtualenv
   
   # 创建虚拟环境
   virtualenv venv
   
   # 激活虚拟环境
   # Linux/macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **安装依赖库**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn statsmodels yfinance backtrader alpaca-trade-api python-dotenv schedule ta
   ```

### 2.2 获取代码

1. **克隆或下载代码**:
   ```bash
   git clone <repository-url>
   # 或下载ZIP文件并解压
   
   cd quantitative_trading
   ```

2. **目录结构**:
   ```
   quantitative_trading/
   ├── backtest/
   │   └── backtest_framework.py
   ├── config/
   │   └── trading_config.json
   ├── data/
   ├── docs/
   │   ├── strategy_documentation.md
   │   ├── backtest_report.md
   │   ├── deployment_guide.md
   │   └── user_manual.md
   ├── logs/
   ├── optimization/
   │   └── strategy_optimizer.py
   ├── research/
   │   ├── market_analysis.py
   │   └── strategy_research.py
   ├── strategy/
   │   └── high_performance_strategy.py
   └── trading/
       └── automated_trading_system.py
   ```

## 3. API配置

### 3.1 Alpaca API设置

1. **创建Alpaca账户**:
   - 访问 [Alpaca官网](https://alpaca.markets/)
   - 注册账户 (纸面交易或实盘账户)
   - 获取API密钥和密钥ID

2. **配置API凭证**:
   - 创建`.env`文件在项目根目录:
     ```
     ALPACA_API_KEY=your_api_key
     ALPACA_API_SECRET=your_api_secret
     ALPACA_BASE_URL=https://paper-api.alpaca.markets  # 纸面交易
     # ALPACA_BASE_URL=https://api.alpaca.markets  # 实盘交易
     ```

### 3.2 其他API配置 (可选)

如果需要使用其他数据源或交易平台，可以类似地配置相应的API凭证:

- **Interactive Brokers**:
  ```
  IB_ACCOUNT=your_account
  IB_PASSWORD=your_password
  IB_HOST=127.0.0.1
  IB_PORT=7496  # TWS
  # IB_PORT=4001  # IB Gateway
  ```

- **Binance** (加密货币):
  ```
  BINANCE_API_KEY=your_api_key
  BINANCE_API_SECRET=your_api_secret
  ```

## 4. 系统配置

### 4.1 交易配置文件

系统会自动创建默认配置文件`config/trading_config.json`，您可以根据需要修改:

```json
{
    "symbols": ["NVDA", "BTC-USD", "ETH-USD", "TSLA"],
    "data_source": "yahoo",
    "trading_platform": "alpaca",
    "api_key": "",
    "api_secret": "",
    "base_url": "https://paper-api.alpaca.markets",
    "initial_capital": 100000,
    "max_position_size": 0.2,
    "max_leverage": 2.0,
    "risk_tolerance": 0.02,
    "trading_frequency": "daily",
    "trading_hours": {
        "start": "09:30",
        "end": "16:00"
    },
    "strategy_params": {
        "NVDA": {
            "momentum": {"window": 15},
            "trend": {"short_window": 10, "long_window": 30},
            "mean_reversion": {"window": 20, "std_dev": 2.0},
            "combined": {
                "momentum_weight": 0.4,
                "trend_weight": 0.4,
                "mean_reversion_weight": 0.2
            },
            "leverage": 2.0,
            "stop_loss": 0.05
        },
        "BTC-USD": {
            "momentum": {"window": 20},
            "trend": {"short_window": 10, "long_window": 50},
            "mean_reversion": {"window": 20, "std_dev": 2.5},
            "combined": {
                "momentum_weight": 0.5,
                "trend_weight": 0.3,
                "mean_reversion_weight": 0.2
            },
            "leverage": 1.5,
            "stop_loss": 0.07
        },
        "ETH-USD": {
            "momentum": {"window": 20},
            "trend": {"short_window": 10, "long_window": 50},
            "mean_reversion": {"window": 20, "std_dev": 2.5},
            "combined": {
                "momentum_weight": 0.5,
                "trend_weight": 0.3,
                "mean_reversion_weight": 0.2
            },
            "leverage": 1.5,
            "stop_loss": 0.07
        },
        "TSLA": {
            "momentum": {"window": 15},
            "trend": {"short_window": 10, "long_window": 30},
            "mean_reversion": {"window": 20, "std_dev": 2.0},
            "combined": {
                "momentum_weight": 0.4,
                "trend_weight": 0.4,
                "mean_reversion_weight": 0.2
            },
            "leverage": 2.0,
            "stop_loss": 0.05
        }
    },
    "portfolio_allocation": {
        "NVDA": 0.4,
        "BTC-USD": 0.2,
        "ETH-USD": 0.2,
        "TSLA": 0.2
    }
}
```

### 4.2 日志配置

系统默认将日志保存在`logs/trading_system.log`文件中。日志级别默认为INFO，可以在代码中修改:

```python
logging.basicConfig(
    level=logging.INFO,  # 可修改为 logging.DEBUG 获取更详细日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
```

## 5. 系统部署

### 5.1 本地部署

1. **运行回测**:
   ```bash
   cd quantitative_trading
   python backtest/backtest_framework.py
   ```

2. **运行策略优化**:
   ```bash
   python optimization/strategy_optimizer.py
   ```

3. **运行模拟交易**:
   ```bash
   python trading/automated_trading_system.py
   ```

### 5.2 服务器部署

对于长期运行的交易系统，建议部署在服务器上:

1. **使用Screen或Tmux**:
   ```bash
   # 安装screen
   sudo apt install screen
   
   # 创建新会话
   screen -S trading
   
   # 运行交易系统
   cd quantitative_trading
   python trading/automated_trading_system.py
   
   # 分离会话 (Ctrl+A, D)
   # 重新连接会话
   screen -r trading
   ```

2. **使用Systemd服务** (Linux):
   
   创建服务文件:
   ```bash
   sudo nano /etc/systemd/system/trading-system.service
   ```
   
   添加以下内容:
   ```
   [Unit]
   Description=Quantitative Trading System
   After=network.target
   
   [Service]
   User=your_username
   WorkingDirectory=/path/to/quantitative_trading
   ExecStart=/path/to/python /path/to/quantitative_trading/trading/automated_trading_system.py
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   启动服务:
   ```bash
   sudo systemctl enable trading-system
   sudo systemctl start trading-system
   sudo systemctl status trading-system
   ```

### 5.3 云部署

也可以将系统部署在云服务器上，如AWS、Google Cloud或Azure:

1. **创建云服务器实例**
2. **配置安全组和网络**
3. **SSH连接到服务器**
4. **按照上述步骤设置环境和部署系统**

## 6. 监控和维护

### 6.1 系统监控

1. **日志监控**:
   ```bash
   tail -f logs/trading_system.log
   ```

2. **性能监控**:
   - 查看`data/performance.csv`文件
   - 查看`data/equity_curve.png`图表

3. **设置警报** (可选):
   - 可以编写脚本监控日志文件，当出现错误或特定事件时发送邮件或短信通知

### 6.2 系统维护

1. **定期更新**:
   ```bash
   git pull  # 如果使用Git管理代码
   pip install -r requirements.txt  # 更新依赖
   ```

2. **数据备份**:
   ```bash
   # 备份配置和数据
   cp -r config/ backup/config_$(date +%Y%m%d)/
   cp -r data/ backup/data_$(date +%Y%m%d)/
   ```

3. **参数调整**:
   - 定期检查策略性能
   - 根据市场变化调整策略参数
   - 运行优化器重新优化参数

## 7. 故障排除

### 7.1 常见问题

1. **API连接失败**:
   - 检查API凭证是否正确
   - 检查网络连接
   - 检查API服务是否可用

2. **数据获取错误**:
   - 检查数据源是否可用
   - 尝试更换数据源
   - 检查资产代码是否正确

3. **交易执行失败**:
   - 检查账户资金是否充足
   - 检查交易限制
   - 检查市场是否开放

### 7.2 日志分析

查看日志文件以获取详细错误信息:
```bash
grep "ERROR" logs/trading_system.log
```

## 8. 升级和扩展

### 8.1 添加新资产

1. 修改配置文件中的`symbols`列表
2. 为新资产添加策略参数
3. 更新投资组合配置

### 8.2 添加新策略

1. 在`trading/automated_trading_system.py`中添加新的策略生成函数
2. 更新组合策略逻辑
3. 修改配置文件添加新策略参数

### 8.3 性能优化

1. 使用更高效的数据结构
2. 优化计算密集型操作
3. 考虑使用并行处理

## 9. 安全注意事项

1. **API密钥安全**:
   - 不要在代码中硬编码API密钥
   - 使用环境变量或加密存储
   - 定期轮换API密钥

2. **资金安全**:
   - 开始时使用小额资金测试
   - 设置交易限额
   - 实施紧急停止机制

3. **系统安全**:
   - 保持系统和依赖库更新
   - 使用防火墙限制访问
   - 定期备份数据

## 10. 结论

按照本指南的步骤，您应该能够成功部署和运行高性能量化交易策略系统。记住，交易系统需要持续监控和维护，以适应不断变化的市场环境。

建议先在模拟环境中运行系统一段时间，确认系统稳定且策略表现符合预期后，再考虑使用实际资金进行交易。
