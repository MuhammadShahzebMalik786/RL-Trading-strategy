from advanced_trading_env import AdvancedTradingEnv

env = AdvancedTradingEnv()
obs, _ = env.reset()

print("=== Testing Trading Sequence ===")
print(f"Initial: Balance=${env.balance}, Positions={len(env.positions)}")

# Take a buy action
obs, reward, done, _, info = env.step(2)  # Buy medium
print(f"After buy: Balance=${env.balance:.4f}, Positions={len(env.positions)}, Reward={reward}")

if len(env.positions) > 0:
    pos = env.positions[0]
    print(f"Position: {pos['type']} {pos['size']} @ {pos['entry_price']:.4f}")
    print(f"Stop: {pos['stop_loss']:.4f}, TP: {pos['take_profit']:.4f}")

# Run a few steps to see what happens
for i in range(10):
    obs, reward, done, _, info = env.step(0)  # Hold
    current_price = env.data.iloc[env.step_count + env.lookback - 1]['close']
    
    # Calculate real-time unrealized PnL
    if len(env.positions) > 0:
        pos = env.positions[0]
        unrealized_pnl = env._calculate_unrealized_pnl(pos, current_price)
        print(f"Step {i+1}: Price=${current_price:.2f}, Unrealized PnL=${unrealized_pnl:.2f}, Balance=${env.balance:.2f}")
    else:
        print(f"Step {i+1}: Price=${current_price:.2f}, No positions")
    
    if len(env.positions) == 0:
        print("Position closed!")
        break

print(f"\nFinal metrics: {env.get_metrics()}")
