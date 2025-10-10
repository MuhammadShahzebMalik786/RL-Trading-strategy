# RL Forex Trading Agent

A complete reinforcement learning framework for forex trading with realistic market conditions and risk management.

## 🎯 Objectives
- Achieve ≥70% trade accuracy OR $20 daily profit
- Trade with $10 initial capital using 1:2000 leverage
- Implement realistic spread costs and margin requirements

## 🚀 Quick Start

1. **Setup Environment**
```bash
python setup.py
```

2. **Test Environment**
```bash
python demo.py
```

3. **Train Agent**
```bash
python train_agent.py
```

## 📊 Features

### Trading Environment
- **Initial Balance**: $10 USDT
- **Leverage**: 1:2000
- **Spread**: $1.6 per lot
- **Min Lot Size**: 0.1
- **Margin**: $1 per lot

### State Space
- Normalized OHLCV data (20 bars)
- Technical indicators (RSI, MACD, EMA, Bollinger Bands)
- Account information (balance, margin, positions)
- Performance metrics (win rate, volatility, drawdown)

### Action Space
- 0: Hold
- 1: Buy (0.1 lot)
- 2: Sell (0.1 lot)  
- 3: Close all positions

### Risk Management
- Max drawdown: 20%
- Stop loss: 2% per trade
- Take profit: 4% per trade
- Max concurrent trades: 3
- Trade cooldown: 3-5 bars

### Reward Function
- +5 for profitable trades
- -7 for losing trades
- -2 × (drawdown/balance) penalty
- +50 bonus for $20+ daily profit
- -20 penalty for $10+ daily loss

## 📈 Data Sources
- **Primary**: Binance API (ETH/USDT real-time data)
- **Fallback**: Synthetic OHLCV data generator

## 🧠 RL Algorithm
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MLP (Multi-Layer Perceptron)
- **Training Steps**: 100,000
- **Evaluation**: Every 2,000 steps

## 📊 Monitoring & Visualization
- Real-time training progress
- Win rate tracking
- PnL evolution
- Drawdown monitoring
- Performance metrics dashboard

## 🎯 Success Criteria
The agent is considered successful if it achieves:
- **Accuracy**: ≥70% win rate ✅
- **Profit**: ≥$20 daily profit ✅

## 📁 File Structure
```
RL Trading strategy/
├── trading_env.py      # Main trading environment
├── train_agent.py      # Training script
├── demo.py            # Environment testing
├── setup.py           # Setup script
├── requirements.txt   # Dependencies
├── README.md         # This file
├── models/           # Saved models
├── logs/            # Training logs
├── tensorboard_logs/ # TensorBoard logs
└── plots/           # Generated plots
```

## 🔧 Configuration
All trading parameters are configurable in `TradingEnv.__init__()`:
- Initial balance
- Leverage ratio
- Spread costs
- Risk limits
- Reward parameters

## 📊 Performance Metrics
- **Win Rate**: Percentage of profitable trades
- **Total PnL**: Net profit/loss in USD
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum equity decline
- **Final Balance**: End-of-episode balance

## 🚨 Risk Disclaimer
This is for educational purposes only. Real trading involves significant risk of loss. Always use proper risk management and never trade with money you cannot afford to lose.
"# RL-Trading-strategy" 
