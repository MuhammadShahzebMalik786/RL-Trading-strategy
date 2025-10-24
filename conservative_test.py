from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import requests
import pandas as pd

print("🚀 CONSERVATIVE LIVE TEST")
print("=" * 40)

# Load model
model = PPO.load("models/best_model")
print("✅ Model loaded")

# Get current ETH price
try:
    response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT", timeout=5)
    current_price = float(response.json()['price'])
    print(f"📊 Current ETH: ${current_price:.2f}")
except:
    current_price = 3970
    print(f"📊 Using fallback price: ${current_price:.2f}")

# Create environment with conservative settings
env = AdvancedTradingEnv(
    leverage=5,  # Lower leverage for safety
    initial_balance=100,  # Higher balance
    max_steps=200
)

print(f"💰 Starting Balance: ${env.balance}")
print(f"⚡ Leverage: {env.leverage}x (conservative)")

# Run multiple short episodes
total_sessions = 5
profitable_sessions = 0

for session in range(total_sessions):
    obs, _ = env.reset()
    session_start_balance = env.balance
    
    print(f"\n📈 Session {session + 1}/5:")
    
    for step in range(20):  # Short sessions
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        if step % 5 == 0:
            print(f"  Step {step:2d}: Balance=${env.balance:6.2f} | PnL=${info['total_pnl']:6.2f} | Pos={info['open_positions']}")
        
        if done:
            break
    
    session_profit = env.balance - session_start_balance + info['total_pnl']
    if session_profit > 0:
        profitable_sessions += 1
        print(f"  ✅ Session profit: ${session_profit:.2f}")
    else:
        print(f"  📊 Session result: ${session_profit:.2f}")

print(f"\n" + "=" * 40)
print("📊 OVERALL RESULTS:")
print(f"💰 Final Balance: ${env.balance:.2f}")
print(f"📈 Total PnL: ${info['total_pnl']:.2f}")
print(f"🎯 Profitable Sessions: {profitable_sessions}/{total_sessions}")
print(f"📊 Success Rate: {profitable_sessions/total_sessions:.1%}")

if profitable_sessions >= 3:
    print("\n🎉 GOOD PERFORMANCE!")
    print("✅ Model shows consistent profitability")
elif profitable_sessions >= 2:
    print("\n📈 MODERATE PERFORMANCE")
    print("⚠️ Consider paper trading first")
else:
    print("\n⚠️ CONSERVATIVE APPROACH RECOMMENDED")
    print("📚 Model needs more training on current market conditions")

print(f"\n💡 Recommendation: {'READY FOR PAPER TRADING' if profitable_sessions >= 2 else 'CONTINUE TRAINING'}")
