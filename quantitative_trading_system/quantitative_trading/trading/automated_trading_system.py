import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import time
import json
import ta
import logging
import threading
import schedule
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# 创建输出目录
os.makedirs('../logs', exist_ok=True)
os.makedirs('../data', exist_ok=True)
os.makedirs('../config', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/trading_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('AutomatedTradingSystem')

class AutomatedTradingSystem:
    """
    自动化交易系统
    
    功能:
    1. 实时数据获取
    2. 信号生成
    3. 交易执行
    4. 风险管理
    5. 性能监控
    """
    
    def __init__(self, config_path=None):
        """
        初始化交易系统
        
        参数:
        config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化数据存储
        self.data = {}
        self.signals = {}
        self.positions = {}
        self.performance = {}
        
        # 初始化交易API
        self._init_trading_api()
        
        # 初始化运行状态
        self.is_running = False
        self.scheduler_thread = None
    
    def _load_config(self, config_path=None):
        """加载配置"""
        default_config = {
            'symbols': ['NVDA', 'BTC-USD', 'ETH-USD', 'TSLA'],
            'data_source': 'yahoo',  # yahoo, alpaca, binance等
            'trading_platform': 'alpaca',  # alpaca, interactive_brokers等
            'api_key': '',
            'api_secret': '',
            'base_url': 'https://paper-api.alpaca.markets',  # 模拟交易
            'initial_capital': 100000,
            'max_position_size': 0.2,  # 单个资产最大仓位比例
            'max_leverage': 2.0,  # 最大杠杆倍数
            'risk_tolerance': 0.02,  # 最大回撤容忍度
            'trading_frequency': 'daily',  # daily, hourly等
            'trading_hours': {
                'start': '09:30',
                'end': '16:00'
            },
            'strategy_params': {
                'NVDA': {
                    'momentum': {'window': 15},
                    'trend': {'short_window': 10, 'long_window': 30},
                    'mean_reversion': {'window': 20, 'std_dev': 2.0},
                    'combined': {
                        'momentum_weight': 0.4,
                        'trend_weight': 0.4,
                        'mean_reversion_weight': 0.2
                    },
                    'leverage': 2.0,
                    'stop_loss': 0.05
                },
                'BTC-USD': {
                    'momentum': {'window': 20},
                    'trend': {'short_window': 10, 'long_window': 50},
                    'mean_reversion': {'window': 20, 'std_dev': 2.5},
                    'combined': {
                        'momentum_weight': 0.5,
                        'trend_weight': 0.3,
                        'mean_reversion_weight': 0.2
                    },
                    'leverage': 1.5,
                    'stop_loss': 0.07
                },
                'ETH-USD': {
                    'momentum': {'window': 20},
                    'trend': {'short_window': 10, 'long_window': 50},
                    'mean_reversion': {'window': 20, 'std_dev': 2.5},
                    'combined': {
                        'momentum_weight': 0.5,
                        'trend_weight': 0.3,
                        'mean_reversion_weight': 0.2
                    },
                    'leverage': 1.5,
                    'stop_loss': 0.07
                },
                'TSLA': {
                    'momentum': {'window': 15},
                    'trend': {'short_window': 10, 'long_window': 30},
                    'mean_reversion': {'window': 20, 'std_dev': 2.0},
                    'combined': {
                        'momentum_weight': 0.4,
                        'trend_weight': 0.4,
                        'mean_reversion_weight': 0.2
                    },
                    'leverage': 2.0,
                    'stop_loss': 0.05
                }
            },
            'portfolio_allocation': {
                'NVDA': 0.4,
                'BTC-USD': 0.2,
                'ETH-USD': 0.2,
                'TSLA': 0.2
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # 合并用户配置和默认配置
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"配置已从 {config_path} 加载")
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
        else:
            # 保存默认配置
            try:
                if not config_path:
                    config_path = '../config/trading_config.json'
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                logger.info(f"默认配置已保存到 {config_path}")
            except Exception as e:
                logger.error(f"保存默认配置失败: {str(e)}")
        
        return default_config
    
    def _init_trading_api(self):
        """初始化交易API"""
        # 加载环境变量
        load_dotenv()
        
        # 获取API凭证
        api_key = self.config.get('api_key') or os.getenv('ALPACA_API_KEY')
        api_secret = self.config.get('api_secret') or os.getenv('ALPACA_API_SECRET')
        base_url = self.config.get('base_url') or os.getenv('ALPACA_BASE_URL')
        
        if not api_key or not api_secret:
            logger.warning("未找到API凭证，交易执行将被禁用")
            self.trading_enabled = False
            self.api = None
        else:
            try:
                self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
                account = self.api.get_account()
                logger.info(f"交易API初始化成功，账户ID: {account.id}")
                logger.info(f"账户状态: {account.status}")
                logger.info(f"账户现金: ${float(account.cash):.2f}")
                logger.info(f"账户投资组合价值: ${float(account.portfolio_value):.2f}")
                self.trading_enabled = True
            except Exception as e:
                logger.error(f"交易API初始化失败: {str(e)}")
                self.trading_enabled = False
                self.api = None
    
    def fetch_data(self, symbol, period='1d', lookback_days=30):
        """获取实时市场数据"""
        logger.info(f"获取 {symbol} 的市场数据，周期: {period}, 回溯天数: {lookback_days}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            # 使用Yahoo Finance获取数据
            data = yf.download(symbol, start=start_date, end=end_date, interval=period)
            
            if len(data) < 5:
                logger.warning(f"{symbol}: 数据不足，跳过")
                return None
            
            # 检查是否有多级索引并处理
            if isinstance(data.columns, pd.MultiIndex):
                logger.info(f"{symbol} 数据有多级索引，进行处理...")
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
            logger.info(f"{symbol}: 获取了 {len(data)} 条数据")
            
            return data
            
        except Exception as e:
            logger.error(f"{symbol}: 获取数据失败 - {str(e)}")
            return None
    
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
            logger.error(f"添加技术指标失败: {str(e)}")
        
        return data
    
    def generate_signals(self, symbol):
        """生成交易信号"""
        logger.info(f"为 {symbol} 生成交易信号")
        
        if symbol not in self.data:
            logger.warning(f"{symbol}: 没有数据，跳过信号生成")
            return None
        
        data = self.data[symbol]
        
        # 获取策略参数
        strategy_params = self.config['strategy_params'].get(symbol, {})
        
        # 生成动量策略信号
        momentum_params = strategy_params.get('momentum', {'window': 20})
        momentum_signals = self._generate_momentum_signals(data, momentum_params)
        
        # 生成趋势跟踪策略信号
        trend_params = strategy_params.get('trend', {'short_window': 20, 'long_window': 50})
        trend_signals = self._generate_trend_signals(data, trend_params)
        
        # 生成均值回归策略信号
        mean_reversion_params = strategy_params.get('mean_reversion', {'window': 20, 'std_dev': 2.0})
        mean_reversion_signals = self._generate_mean_reversion_signals(data, mean_reversion_params)
        
        # 生成组合策略信号
        combined_params = strategy_params.get('combined', {
            'momentum_weight': 0.33,
            'trend_weight': 0.33,
            'mean_reversion_weight': 0.34
        })
        combined_signals = self._generate_combined_signals(
            momentum_signals, 
            trend_signals, 
            mean_reversion_signals, 
            combined_params
        )
        
        # 应用杠杆
        leverage = strategy_params.get('leverage', 1.0)
        leveraged_signals = self._apply_leverage(combined_signals, leverage)
        
        # 应用止损
        stop_loss = strategy_params.get('stop_loss', 0.05)
        final_signals = self._apply_stop_loss(leveraged_signals, stop_loss)
        
        # 存储信号
        self.signals[symbol] = final_signals
        
        logger.info(f"{symbol}: 生成了交易信号，最新信号: {final_signals['Signal'].iloc[-1]}")
        
        return final_signals
    
    def _generate_momentum_signals(self, data, params):
        """生成动量策略信号"""
        window = params.get('window', 20)
        
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data['Close']
        signals['Daily_Return'] = data['Daily_Return']
        
        # 计算动量指标
        signals['Momentum'] = data['Close'].pct_change(window)
        
        # 生成交易信号
        signals['Signal'] = 0
        signals.loc[signals['Momentum'] > 0, 'Signal'] = 1  # 买入信号
        signals.loc[signals['Momentum'] < 0, 'Signal'] = -1  # 卖出信号
        
        return signals
    
    def _generate_trend_signals(self, data, params):
        """生成趋势跟踪策略信号"""
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
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
        
        return signals
    
    def _generate_mean_reversion_signals(self, data, params):
        """生成均值回归策略信号"""
        window = params.get('window', 20)
        std_dev = params.get('std_dev', 2.0)
        
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
        
        return signals
    
    def _generate_combined_signals(self, momentum_signals, trend_signals, mean_reversion_signals, params):
        """生成组合策略信号"""
        momentum_weight = params.get('momentum_weight', 0.33)
        trend_weight = params.get('trend_weight', 0.33)
        mean_reversion_weight = params.get('mean_reversion_weight', 0.34)
        
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
            momentum_weight * combined['Momentum_Signal'] +
            trend_weight * combined['Trend_Signal'] +
            mean_reversion_weight * combined['Mean_Reversion_Signal']
        )
        
        # 最终信号：大于0为买入，小于0为卖出
        combined['Signal'] = np.sign(combined['Combined_Signal'])
        
        return combined
    
    def _apply_leverage(self, signals, leverage):
        """应用杠杆"""
        leveraged = signals.copy()
        leveraged['Leveraged_Signal'] = leveraged['Signal'] * leverage
        return leveraged
    
    def _apply_stop_loss(self, signals, stop_loss):
        """应用止损"""
        with_stop_loss = signals.copy()
        
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
                with_stop_loss.loc[with_stop_loss.index[i], 'In_Position'] = False
                position = 0
            elif position < 0 and curr_price > stop_loss_price:  # 空头止损
                with_stop_loss.loc[with_stop_loss.index[i], 'In_Position'] = False
                position = 0
            
            # 记录止损价格
            with_stop_loss.loc[with_stop_loss.index[i], 'Stop_Loss_Price'] = stop_loss_price
        
        # 应用止损
        with_stop_loss['Effective_Signal'] = with_stop_loss['Leveraged_Signal'] * with_stop_loss['In_Position']
        
        # 最终信号
        with_stop_loss['Signal'] = with_stop_loss['Effective_Signal']
        
        return with_stop_loss
    
    def calculate_position_sizes(self):
        """计算仓位大小"""
        logger.info("计算仓位大小")
        
        # 获取投资组合配置
        portfolio_allocation = self.config.get('portfolio_allocation', {})
        
        # 如果没有配置，使用均等权重
        if not portfolio_allocation:
            symbols = list(self.signals.keys())
            portfolio_allocation = {symbol: 1.0 / len(symbols) for symbol in symbols}
        
        # 计算每个资产的仓位
        for symbol, signals in self.signals.items():
            if symbol not in portfolio_allocation:
                logger.warning(f"{symbol}: 没有配置投资组合权重，跳过")
                continue
            
            # 获取资产权重
            weight = portfolio_allocation[symbol]
            
            # 创建仓位DataFrame
            position = pd.DataFrame(index=signals.index)
            position['Close'] = signals['Close']
            position['Signal'] = signals['Signal']
            
            # 计算仓位大小
            position['Position_Size'] = position['Signal'] * weight
            
            # 存储仓位
            self.positions[symbol] = position
            
            logger.info(f"{symbol}: 计算了仓位大小，最新仓位: {position['Position_Size'].iloc[-1]:.2f}")
        
        return self.positions
    
    def execute_trades(self):
        """执行交易"""
        if not self.trading_enabled:
            logger.warning("交易执行被禁用，请检查API凭证")
            return False
        
        logger.info("执行交易...")
        
        try:
            # 获取当前持仓
            current_positions = {}
            positions = self.api.list_positions()
            for position in positions:
                current_positions[position.symbol] = {
                    'qty': float(position.qty),
                    'market_value': float(position.market_value),
                    'current_price': float(position.current_price)
                }
            
            # 获取账户信息
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            
            # 执行交易
            for symbol, position in self.positions.items():
                # 获取最新信号和仓位大小
                latest_signal = position['Signal'].iloc[-1]
                latest_position_size = position['Position_Size'].iloc[-1]
                
                # 计算目标持仓金额
                target_value = portfolio_value * latest_position_size
                
                # 获取当前持仓
                current_position = current_positions.get(symbol, {'qty': 0, 'market_value': 0, 'current_price': 0})
                current_value = current_position['market_value']
                
                # 计算需要调整的金额
                adjustment = target_value - current_value
                
                # 如果调整金额很小，跳过
                if abs(adjustment) < 100:
                    logger.info(f"{symbol}: 调整金额较小 (${adjustment:.2f})，跳过")
                    continue
                
                # 获取最新价格
                latest_price = position['Close'].iloc[-1]
                
                # 计算需要买入/卖出的股数
                shares_to_trade = int(adjustment / latest_price)
                
                if shares_to_trade > 0:
                    # 买入
                    logger.info(f"{symbol}: 买入 {shares_to_trade} 股，价格: ${latest_price:.2f}")
                    self.api.submit_order(
                        symbol=symbol,
                        qty=shares_to_trade,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                elif shares_to_trade < 0:
                    # 卖出
                    logger.info(f"{symbol}: 卖出 {abs(shares_to_trade)} 股，价格: ${latest_price:.2f}")
                    self.api.submit_order(
                        symbol=symbol,
                        qty=abs(shares_to_trade),
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                else:
                    logger.info(f"{symbol}: 无需调整")
            
            logger.info("交易执行完成")
            return True
            
        except Exception as e:
            logger.error(f"执行交易失败: {str(e)}")
            return False
    
    def update_performance(self):
        """更新性能指标"""
        logger.info("更新性能指标")
        
        try:
            # 获取账户信息
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            
            # 获取历史性能
            if not os.path.exists('../data/performance.csv'):
                # 创建新的性能记录
                performance = pd.DataFrame({
                    'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Portfolio_Value': [portfolio_value],
                    'Daily_Return': [0]
                })
            else:
                # 加载历史性能
                performance = pd.read_csv('../data/performance.csv')
                
                # 计算日收益率
                prev_value = performance['Portfolio_Value'].iloc[-1]
                daily_return = (portfolio_value - prev_value) / prev_value
                
                # 添加新记录
                new_record = pd.DataFrame({
                    'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Portfolio_Value': [portfolio_value],
                    'Daily_Return': [daily_return]
                })
                
                performance = pd.concat([performance, new_record], ignore_index=True)
            
            # 保存性能记录
            performance.to_csv('../data/performance.csv', index=False)
            
            # 计算性能指标
            if len(performance) > 1:
                # 计算累积收益
                performance['Cumulative_Return'] = (1 + performance['Daily_Return']).cumprod()
                
                # 计算回撤
                performance['Peak'] = performance['Cumulative_Return'].cummax()
                performance['Drawdown'] = (performance['Cumulative_Return'] - performance['Peak']) / performance['Peak']
                
                # 计算年化收益率
                total_return = performance['Cumulative_Return'].iloc[-1] - 1
                days = (datetime.now() - datetime.strptime(performance['Date'].iloc[0], '%Y-%m-%d %H:%M:%S')).days
                annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1
                
                # 计算胜率
                win_rate = len(performance[performance['Daily_Return'] > 0]) / len(performance[performance['Daily_Return'] != 0]) if len(performance[performance['Daily_Return'] != 0]) > 0 else 0
                
                # 计算最大回撤
                max_drawdown = performance['Drawdown'].min()
                
                # 计算夏普比率
                risk_free_rate = 0.02 / 365  # 日化无风险利率
                excess_return = performance['Daily_Return'] - risk_free_rate
                sharpe_ratio = excess_return.mean() / excess_return.std() * np.sqrt(365) if excess_return.std() > 0 else 0
                
                # 记录性能指标
                metrics = {
                    'portfolio_value': portfolio_value,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio
                }
                
                self.performance = metrics
                
                logger.info(f"投资组合价值: ${portfolio_value:.2f}")
                logger.info(f"总收益率: {total_return:.2%}")
                logger.info(f"年化收益率: {annual_return:.2%}")
                logger.info(f"胜率: {win_rate:.2%}")
                logger.info(f"最大回撤: {max_drawdown:.2%}")
                logger.info(f"夏普比率: {sharpe_ratio:.2f}")
                
                # 绘制权益曲线
                self._plot_equity_curve(performance)
            
            return True
            
        except Exception as e:
            logger.error(f"更新性能指标失败: {str(e)}")
            return False
    
    def _plot_equity_curve(self, performance):
        """绘制权益曲线"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 绘制权益曲线
            plt.subplot(2, 1, 1)
            plt.plot(performance['Cumulative_Return'])
            plt.title('累积收益')
            plt.grid(True)
            
            # 绘制回撤曲线
            plt.subplot(2, 1, 2)
            plt.plot(performance['Drawdown'])
            plt.title('回撤')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('../data/equity_curve.png')
            plt.close()
            
            logger.info("权益曲线已保存到 ../data/equity_curve.png")
            
        except Exception as e:
            logger.error(f"绘制权益曲线失败: {str(e)}")
    
    def trading_cycle(self):
        """执行一个完整的交易周期"""
        logger.info("开始执行交易周期")
        
        # 1. 获取市场数据
        for symbol in self.config['symbols']:
            self.fetch_data(symbol)
        
        # 2. 生成交易信号
        for symbol in self.config['symbols']:
            self.generate_signals(symbol)
        
        # 3. 计算仓位大小
        self.calculate_position_sizes()
        
        # 4. 执行交易
        self.execute_trades()
        
        # 5. 更新性能指标
        self.update_performance()
        
        logger.info("交易周期执行完成")
    
    def start(self):
        """启动交易系统"""
        if self.is_running:
            logger.warning("交易系统已经在运行")
            return
        
        logger.info("启动交易系统")
        self.is_running = True
        
        # 设置交易频率
        trading_frequency = self.config.get('trading_frequency', 'daily')
        
        if trading_frequency == 'daily':
            # 每天执行一次
            schedule.every().day.at("09:30").do(self.trading_cycle)
            logger.info("设置每天 09:30 执行交易")
        elif trading_frequency == 'hourly':
            # 每小时执行一次
            schedule.every().hour.do(self.trading_cycle)
            logger.info("设置每小时执行交易")
        else:
            logger.warning(f"未知的交易频率: {trading_frequency}，默认为每天执行")
            schedule.every().day.at("09:30").do(self.trading_cycle)
        
        # 立即执行一次
        self.trading_cycle()
        
        # 启动调度器线程
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("交易系统已启动")
    
    def stop(self):
        """停止交易系统"""
        if not self.is_running:
            logger.warning("交易系统未在运行")
            return
        
        logger.info("停止交易系统")
        self.is_running = False
        
        # 等待调度器线程结束
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("交易系统已停止")
    
    def run_simulation(self, days=30):
        """运行模拟交易"""
        logger.info(f"开始运行 {days} 天的模拟交易")
        
        # 创建模拟性能记录
        performance = pd.DataFrame({
            'Date': [datetime.now().strftime('%Y-%m-%d')],
            'Portfolio_Value': [self.config['initial_capital']],
            'Daily_Return': [0]
        })
        
        # 模拟每天交易
        for day in range(1, days + 1):
            # 模拟日期
            sim_date = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
            
            # 1. 获取市场数据
            for symbol in self.config['symbols']:
                self.fetch_data(symbol)
            
            # 2. 生成交易信号
            for symbol in self.config['symbols']:
                self.generate_signals(symbol)
            
            # 3. 计算仓位大小
            self.calculate_position_sizes()
            
            # 4. 模拟交易结果
            portfolio_value = performance['Portfolio_Value'].iloc[-1]
            daily_return = 0
            
            for symbol, position in self.positions.items():
                # 获取最新信号和仓位大小
                latest_signal = position['Signal'].iloc[-1]
                latest_position_size = position['Position_Size'].iloc[-1]
                
                # 模拟每日收益
                if latest_signal != 0:
                    # 使用随机收益率模拟
                    symbol_return = np.random.normal(0.001, 0.02)  # 均值0.1%，标准差2%
                    
                    # 如果信号为卖空，反转收益率
                    if latest_signal < 0:
                        symbol_return = -symbol_return
                    
                    # 计算贡献的收益率
                    contribution = symbol_return * latest_position_size
                    daily_return += contribution
            
            # 更新投资组合价值
            new_portfolio_value = portfolio_value * (1 + daily_return)
            
            # 添加新记录
            new_record = pd.DataFrame({
                'Date': [sim_date],
                'Portfolio_Value': [new_portfolio_value],
                'Daily_Return': [daily_return]
            })
            
            performance = pd.concat([performance, new_record], ignore_index=True)
            
            logger.info(f"模拟交易 Day {day}: 日期={sim_date}, 收益率={daily_return:.2%}, 投资组合价值=${new_portfolio_value:.2f}")
        
        # 计算性能指标
        performance['Cumulative_Return'] = (1 + performance['Daily_Return']).cumprod()
        performance['Peak'] = performance['Cumulative_Return'].cummax()
        performance['Drawdown'] = (performance['Cumulative_Return'] - performance['Peak']) / performance['Peak']
        
        # 计算年化收益率
        total_return = performance['Cumulative_Return'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 计算胜率
        win_rate = len(performance[performance['Daily_Return'] > 0]) / len(performance[performance['Daily_Return'] != 0])
        
        # 计算最大回撤
        max_drawdown = performance['Drawdown'].min()
        
        # 计算夏普比率
        risk_free_rate = 0.02 / 365  # 日化无风险利率
        excess_return = performance['Daily_Return'] - risk_free_rate
        sharpe_ratio = excess_return.mean() / excess_return.std() * np.sqrt(365)
        
        # 记录性能指标
        metrics = {
            'portfolio_value': performance['Portfolio_Value'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
        self.performance = metrics
        
        logger.info(f"模拟交易完成")
        logger.info(f"最终投资组合价值: ${metrics['portfolio_value']:.2f}")
        logger.info(f"总收益率: {metrics['total_return']:.2%}")
        logger.info(f"年化收益率: {metrics['annual_return']:.2%}")
        logger.info(f"胜率: {metrics['win_rate']:.2%}")
        logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        logger.info(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        
        # 绘制权益曲线
        self._plot_equity_curve(performance)
        
        # 保存性能记录
        performance.to_csv('../data/simulation_performance.csv', index=False)
        
        return metrics

# 运行交易系统
if __name__ == "__main__":
    # 创建交易系统实例
    trading_system = AutomatedTradingSystem()
    
    # 运行模拟交易
    trading_system.run_simulation(days=365)
    
    # 如果要启动实时交易系统，取消下面的注释
    # trading_system.start()
    
    # 保持程序运行
    # try:
    #     while True:
    #         time.sleep(60)
    # except KeyboardInterrupt:
    #     trading_system.stop()
    #     logger.info("程序已退出")
