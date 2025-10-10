from trading_env import TradingEnv
import numpy as np

def test_environment():
    """Test the trading environment with random actions"""
    print("🧪 Testing Trading Environment...")
    
    env = TradingEnv()
    obs, info = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    total_reward = 0
    step_count = 0
    
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        step_count += 1
        
        if step % 20 == 0:
            metrics = env.get_metrics()
            print(f"Step {step}: Action={action}, Reward={reward:.2f}, "
                  f"Balance=${env.balance:.2f}, Trades={metrics['total_trades']}")
        
        if done:
            print(f"Episode finished at step {step}")
            break
    
    final_metrics = env.get_metrics()
    print(f"\n📊 Final Results:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Balance: ${final_metrics['final_balance']:.2f}")
    print(f"Total Trades: {final_metrics['total_trades']}")
    print(f"Win Rate: {final_metrics['win_rate']:.2%}")
    print(f"Total PnL: ${final_metrics['total_pnl']:.2f}")

if __name__ == "__main__":
    test_environment()
