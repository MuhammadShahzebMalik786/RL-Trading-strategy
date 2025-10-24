#!/usr/bin/env python3
"""
Advanced RL Trading Demo
Demonstrates the enhanced trading environment capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from advanced_trading_env import AdvancedTradingEnv

def demo_environment():
    """Demonstrate the advanced trading environment"""
    
    print("🚀 Advanced RL Trading Environment Demo")
    print("=" * 50)
    
    # Create environment with correct settings
    env = AdvancedTradingEnv(
        initial_balance=10.0,
        leverage=10,  # Fixed leverage
        max_steps=500,
        lookback=50
    )
    
    print(f"📊 Environment Configuration:")
    print(f"  Initial Balance: ${env.initial_balance}")
    print(f"  Leverage: 1:{env.leverage}")
    print(f"  Action Space: {env.action_space.n} actions")
    print(f"  Observation Space: {env.observation_space.shape[0]} features")
    print(f"  Max Steps: {env.max_steps}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\n🔄 Environment Reset Complete")
    print(f"  Observation Shape: {obs.shape}")
    print(f"  Initial Balance: ${env.balance:.2f}")
    
    # Demo actions
    actions = [
        (0, "Hold"),
        (2, "Buy Medium"),
        (0, "Hold"), 
        (0, "Hold"),
        (5, "Sell Medium"),
        (7, "Close All"),
        (1, "Buy Small"),
        (9, "Close Losing")
    ]
    
    print(f"\n🎮 Action Demonstration:")
    print("-" * 40)
    
    step_data = []
    
    for i, (action, description) in enumerate(actions):
        if env.step_count >= env.max_steps - 1:
            break
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: {description}")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Balance: ${env.balance:.2f}")
        print(f"  Positions: {len(env.positions)}")
        print(f"  Terminated: {terminated}")
        print()
        
        step_data.append({
            'step': i+1,
            'action': description,
            'reward': reward,
            'balance': env.balance,
            'positions': len(env.positions)
        })
        
        if terminated:
            print("🛑 Episode terminated!")
            break
    
    # Final metrics
    metrics = env.get_metrics()
    print("📈 Final Performance Metrics:")
    print("-" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key or 'ratio' in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return step_data, metrics

def demo_random_agent():
    """Demo with random agent for comparison"""
    
    print("\n🎲 Random Agent Demo (100 steps)")
    print("=" * 40)
    
    env = AdvancedTradingEnv(max_steps=100)
    obs, info = env.reset()
    
    episode_rewards = []
    balances = []
    
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_rewards.append(reward)
        balances.append(env.balance)
        
        if terminated:
            break
    
    metrics = env.get_metrics()
    
    print(f"Random Agent Results:")
    print(f"  Total Steps: {len(episode_rewards)}")
    print(f"  Final Balance: ${metrics['final_balance']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
    
    return episode_rewards, balances, metrics

def plot_demo_results(step_data, random_rewards, random_balances):
    """Plot demonstration results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Step-by-step balance
    steps = [d['step'] for d in step_data]
    balances = [d['balance'] for d in step_data]
    
    axes[0, 0].plot(steps, balances, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=10, color='r', linestyle='--', label='Initial Balance')
    axes[0, 0].set_title('Balance Evolution (Demo Actions)', fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Balance ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rewards per step
    rewards = [d['reward'] for d in step_data]
    colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in rewards]
    
    axes[0, 1].bar(steps, rewards, color=colors, alpha=0.7)
    axes[0, 1].set_title('Rewards per Action', fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Random agent performance
    axes[1, 0].plot(random_rewards, 'r-', alpha=0.7, label='Rewards')
    axes[1, 0].set_title('Random Agent Rewards', fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(random_balances, 'purple', linewidth=2)
    axes[1, 1].axhline(y=10, color='r', linestyle='--', label='Initial Balance')
    axes[1, 1].set_title('Random Agent Balance', fontweight='bold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Balance ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main demo function"""
    
    # Demo 1: Manual actions
    step_data, demo_metrics = demo_environment()
    
    # Demo 2: Random agent
    random_rewards, random_balances, random_metrics = demo_random_agent()
    
    # Comparison
    print("\n📊 Performance Comparison:")
    print("=" * 50)
    print(f"{'Metric':<20} {'Demo Actions':<15} {'Random Agent':<15}")
    print("-" * 50)
    print(f"{'Final Balance':<20} ${demo_metrics['final_balance']:<14.2f} ${random_metrics['final_balance']:<14.2f}")
    print(f"{'Win Rate':<20} {demo_metrics['win_rate']:<14.2%} {random_metrics['win_rate']:<14.2%}")
    print(f"{'Total PnL':<20} ${demo_metrics['total_pnl']:<14.2f} ${random_metrics['total_pnl']:<14.2f}")
    print(f"{'Sharpe Ratio':<20} {demo_metrics['sharpe_ratio']:<14.2f} {random_metrics['sharpe_ratio']:<14.2f}")
    
    # Plot results
    plot_demo_results(step_data, random_rewards, random_balances)
    
    print("\n✅ Demo completed! Check 'plots/demo_results.png' for visualizations.")
    print("\n🚀 Ready to train advanced models with 'python advanced_train.py'")

if __name__ == "__main__":
    main()
