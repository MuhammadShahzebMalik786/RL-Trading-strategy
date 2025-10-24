import numpy as np
from advanced_trading_env import AdvancedTradingEnv

# Create environment and test position opening
env = AdvancedTradingEnv()
obs, _ = env.reset()

print(f"Initial balance: ${env.balance}")
print(f"Margin per lot: {env.margin_per_lot}")
print(f"Min lot: {env.min_lot}")

# Test position sizing
regime = {'volatility': 1.0, 'trend': 0.0}
base_size = env._calculate_position_size(0.7, regime['volatility'])
print(f"Base position size: {base_size}")

# Test if we can open positions
current_price = 1.1000
for action in [1, 2, 3]:  # Buy actions
    size_multipliers = [0.5, 1.0, 2.0]
    size = base_size * size_multipliers[action - 1]
    can_open = env._can_open_position(size, current_price)
    required_margin = size * env.margin_per_lot
    
    print(f"Action {action}: size={size:.4f}, margin=${required_margin:.4f}, can_open={can_open}")

# Test actual step
print("\nTesting step with action 1 (buy small):")
obs, reward, done, _, info = env.step(1)
print(f"Reward: {reward}")
print(f"Positions: {len(env.positions)}")
print(f"Balance: ${env.balance}")
print(f"Margin used: ${env.margin_used}")
