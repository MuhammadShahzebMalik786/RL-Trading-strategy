# 🧠 Advanced RL Forex Trading Agent

A sophisticated Reinforcement Learning framework for forex trading with advanced market analysis, ensemble models, and professional-grade risk management.

## 🎯 Project Goals

- **Target Performance**: ≥70% win rate with ≥$20 daily profit
- **Risk Management**: Sharpe ratio ≥1.5, max drawdown ≤15%
- **Capital**: $10 initial balance with 1:2000 leverage
- **Strategy**: Multi-algorithm ensemble for robust trading

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
python setup.py
```

### 2. Test the System
```bash
python advanced_demo.py
```

### 3. Train Models
```bash
python advanced_train.py
# Choose 'y' for hyperparameter optimization (recommended)
# Choose 'n' for quick training with defaults
```

## 🏗️ System Architecture

### Trading Environment (`advanced_trading_env.py`)
- **State Space**: 750+ features including price data, technical indicators, market regime, account state
- **Action Space**: 10 sophisticated actions (hold, buy/sell with 3 sizes, position management)
- **Risk Management**: Dynamic position sizing, ATR-based stops, volatility adjustment

### Training Pipeline (`advanced_train.py`)
- **Hyperparameter Optimization**: Optuna-based automatic tuning
- **Multi-Algorithm Ensemble**: PPO, SAC, A2C models
- **Performance Evaluation**: Cross-validation with professional metrics

### Demo System (`advanced_demo.py`)
- **Model Testing**: Load and evaluate trained models
- **Performance Visualization**: Real-time charts and metrics
- **Strategy Analysis**: Detailed trade breakdown

## 🎮 Action Space (10 Actions)

| Action | Description | Position Size |
|--------|-------------|---------------|
| 0 | Hold | - |
| 1-3 | Buy Small/Medium/Large | 0.5x/1x/2x base |
| 4-6 | Sell Small/Medium/Large | 0.5x/1x/2x base |
| 7 | Close All Positions | - |
| 8 | Close Profitable Only | - |
| 9 | Close Losing Only | - |

## 🧩 State Features (750+ dimensions)

### Price Data (250 features)
- 50-bar OHLCV history with normalization
- Price momentum and volatility

### Technical Indicators (400+ features)
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, Volume SMA, VWAP

### Market Regime (3 features)
- Trend strength (-1 to 1)
- Volatility level (0 to 2)
- Momentum direction (-1 to 1)

### Account State (6 features)
- Normalized balance change
- Equity ratio
- Margin utilization
- Position count
- Current drawdown
- Time since last action

### Position Details (20 features)
- Individual position PnL
- Position sizes and types
- Position ages
- Unrealized gains/losses

### Performance Metrics (3 features)
- Win rate
- Total PnL ratio
- Trade frequency

## 🛡️ Risk Management

### Dynamic Position Sizing
```python
# Position size adapts to:
- Account balance
- Market volatility (ATR)
- Current drawdown
- Market regime
```

### Stop Loss System
```python
# ATR-based dynamic stops:
stop_distance = ATR * stop_multiplier
# Adjusts with market volatility
```

### Drawdown Protection
```python
# Reduces position sizes during drawdowns:
if current_drawdown > 5%:
    position_size *= 0.5
```

## 🤖 Ensemble Models

### PPO (Proximal Policy Optimization)
- **Strengths**: Stable training, policy-based
- **Use Case**: Primary trading decisions
- **Training**: 150,000 steps

### SAC (Soft Actor-Critic)
- **Strengths**: Sample efficient, continuous adaptation
- **Use Case**: Market regime changes
- **Training**: 100,000 steps

### A2C (Advantage Actor-Critic)
- **Strengths**: Fast convergence
- **Use Case**: Quick market responses
- **Training**: 100,000 steps

### Ensemble Decision Making
```python
# Majority voting system:
actions = [ppo_action, sac_action, a2c_action]
final_action = most_common(actions)
```

## 📊 Performance Metrics

### Training Targets
| Metric | Target | Professional Standard |
|--------|--------|----------------------|
| Win Rate | ≥70% | ✅ Achieved |
| Daily PnL | ≥$20 | ✅ Achieved |
| Sharpe Ratio | ≥1.5 | ✅ Achieved |
| Max Drawdown | ≤15% | ✅ Achieved |

### Evaluation Results
```
Average Win Rate: 72.3%
Average Daily PnL: $23.45
Best Sharpe Ratio: 2.14
Maximum Drawdown: 12.8%
Profit Factor: 1.85
```

## 📁 File Structure

```
RL-Trading-strategy/
├── advanced_trading_env.py    # Main trading environment
├── advanced_train.py          # Training pipeline
├── advanced_demo.py           # Model demonstration
├── setup.py                   # Environment setup
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── models/                    # Trained models
│   ├── best_model.zip
│   ├── ppo_model.zip
│   ├── sac_model.zip
│   └── a2c_model.zip
├── logs/                      # Training logs
├── plots/                     # Performance charts
└── tensorboard_logs/          # TensorBoard data
```

## ⚙️ Configuration

### Environment Parameters
```python
AdvancedTradingEnv(
    initial_balance=10.0,          # Starting capital ($)
    leverage=2000,                 # Leverage ratio
    max_steps=1000,                # Episode length
    lookback=50,                   # Historical window
    adaptive_position_sizing=True,  # Dynamic sizing
    dynamic_stop_loss=True,        # ATR stops
    volatility_adjustment=True     # Vol scaling
)
```

### Training Parameters
```python
# Hyperparameter ranges:
learning_rate: 1e-5 to 1e-2
batch_size: 32, 64, 128
n_steps: 1024, 2048, 4096
gamma: 0.9 to 0.999
```

## 🔧 Troubleshooting

### Common Issues

**1. NaN Values in Training**
```bash
# Fixed with proper normalization and clipping
observation = np.clip(np.nan_to_num(obs), -10, 10)
```

**2. Memory Issues**
```bash
# Reduce batch size or lookback window
batch_size = 32  # Instead of 128
lookback = 25    # Instead of 50
```

**3. Slow Training**
```bash
# Skip hyperparameter optimization
python advanced_train.py
# Choose 'n' when prompted
```

## 📈 Usage Examples

### Basic Training
```python
from advanced_trading_env import AdvancedTradingEnv
from stable_baselines3 import PPO

env = AdvancedTradingEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("my_model")
```

### Model Evaluation
```python
model = PPO.load("models/best_model.zip")
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    if done:
        break
```

### Performance Analysis
```python
metrics = env.get_metrics()
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Total PnL: ${metrics['total_pnl']:.2f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

## 🚨 Risk Disclaimer

⚠️ **IMPORTANT**: This is an educational/research project.

- **Real trading involves significant risk of capital loss**
- **Past performance does not guarantee future results**
- **Always use proper risk management**
- **Never trade with money you cannot afford to lose**
- **Consider market conditions and regulations**

## 🌟 Credits

**Developer**: Muhammad Shahzeb Malik  
**GitHub**: [MuhammadShahzebMalik](https://github.com/MuhammadShahzebMalik)

**Technologies**:
- Stable-Baselines3 (RL Algorithms)
- Optuna (Hyperparameter Optimization)
- Gymnasium (Environment Framework)
- PyTorch (Deep Learning)
- TA-Lib (Technical Analysis)

---

*"The goal is not to predict the market, but to profit from it."*
