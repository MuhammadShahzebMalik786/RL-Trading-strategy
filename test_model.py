from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import numpy as np

print("🚀 Testing Trained Model")
print("=" * 50)

# Load your excellent trained model
try:
    model = PPO.load("models/best_model")
    print("✅ Loaded best_model.zip")
except:
    model = PPO.load("models/ppo_model")
    print("✅ Loaded ppo_model.zip")

# Create environment with correct settings
env = AdvancedTradingEnv(leverage=10)  # Use correct leverage
obs, _ = env.reset()

print(f"Initial Balance: ${env.balance}")
print(f"Leverage: {env.leverage}x")
print("\n🎯 Running trained model for 100 steps...")

total_reward = 0
for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    
    if step % 20 == 0:
        print(f"Step {step}: Balance=${env.balance:.2f}, PnL=${info['total_pnl']:.2f}, Positions={info['open_positions']}")
    
    if done:
        print(f"Episode ended at step {step}")
        break

print("\n📊 Final Results:")
print(f"Final Balance: ${env.balance:.2f}")
print(f"Total PnL: ${info['total_pnl']:.2f}")
print(f"Win Rate: {info['win_rate']:.1%}")
print(f"Total Trades: {info['total_trades']}")
print(f"Sharpe Ratio: {info['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {info['max_drawdown']:.1%}")

if info['total_pnl'] > 20:
    print("🎯 SUCCESS! Model achieved target profit!")
else:
    print("📈 Model performance logged")
