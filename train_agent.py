import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os
import time
from trading_env import TradingEnv

class TradingCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, verbose=1):
        super(TradingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_metrics = []
        self.step_rewards = []
        self.start_time = None
    
    def _on_training_start(self) -> None:
        import time
        self.start_time = time.time()
        print("🎯 Training Progress:")
        print("=" * 80)
        print(f"{'Step':<8} {'Reward':<8} {'Win%':<6} {'PnL':<8} {'Balance':<8} {'Trades':<7} {'Time':<8}")
        print("-" * 80)
    
    def _on_step(self) -> bool:
        # Track step rewards
        if hasattr(self.locals, 'rewards'):
            self.step_rewards.extend(self.locals['rewards'])
        
        if self.n_calls % self.eval_freq == 0:
            # Evaluate model
            obs, info = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            metrics = self.eval_env.get_metrics()
            self.episode_rewards.append(episode_reward)
            self.episode_metrics.append(metrics)
            
            # Calculate elapsed time
            elapsed = (time.time() - self.start_time) / 60 if self.start_time else 0
            
            # Safe metric extraction
            win_rate = metrics.get('win_rate', 0) * 100
            total_pnl = metrics.get('total_pnl', 0)
            balance = metrics.get('final_balance', self.eval_env.balance)
            trades = metrics.get('total_trades', 0)
            
            # Progress display
            print(f"{self.n_calls:<8} {episode_reward:<8.1f} {win_rate:<6.1f} "
                  f"${total_pnl:<7.1f} ${balance:<7.1f} "
                  f"{trades:<7} {elapsed:<7.1f}m")
            
            # Check targets
            accuracy_met = win_rate >= 70.0
            profit_met = total_pnl >= 20.0
            
            if accuracy_met or profit_met:
                print(f"🎉 TARGET ACHIEVED! {'Accuracy' if accuracy_met else 'Profit'} target met!")
            
            # Save best model
            if episode_reward > self.best_mean_reward:
                self.best_mean_reward = episode_reward
                self.model.save("best_model")
                print(f"💾 New best model saved! (Reward: {episode_reward:.1f})")
            
            # Show recent performance trend
            if len(self.episode_rewards) >= 5:
                recent_avg = np.mean(self.episode_rewards[-5:])
                trend = "📈" if recent_avg > np.mean(self.episode_rewards[-10:-5]) else "📉"
                print(f"{trend} Recent 5-episode avg: {recent_avg:.1f}")
        
        return True

def train_trading_agent():
    """Train the RL trading agent"""
    print("🚀 Starting RL Trading Agent Training...")
    
    # Create environment
    env = TradingEnv()
    eval_env = TradingEnv()
    
    # Create vectorized environment for training
    vec_env = make_vec_env(lambda: env, n_envs=1)
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Create callback
    callback = TradingCallback(eval_env, eval_freq=1000)
    
    # Train the model
    print("📈 Training started...")
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save final model
    model.save("final_model")
    print("💾 Final model saved!")
    
    return model, callback

def evaluate_agent(model_path="best_model"):
    """Evaluate the trained agent"""
    print(f"🔍 Evaluating agent from {model_path}...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create evaluation environment
    env = TradingEnv()
    
    # Run evaluation episodes
    n_episodes = 10
    all_rewards = []
    all_metrics = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        metrics = env.get_metrics()
        all_rewards.append(episode_reward)
        all_metrics.append(metrics)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Win Rate={metrics['win_rate']:.2%}, "
              f"PnL=${metrics['total_pnl']:.2f}")
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print("\n📊 Average Performance:")
    print(f"Win Rate: {avg_metrics['win_rate']:.2%}")
    print(f"Total PnL: ${avg_metrics['total_pnl']:.2f}")
    print(f"Final Balance: ${avg_metrics['final_balance']:.2f}")
    print(f"Max Drawdown: {avg_metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {avg_metrics['sharpe_ratio']:.2f}")
    
    return avg_metrics

def plot_training_progress(callback):
    """Plot training progress"""
    if not callback.episode_rewards:
        print("No training data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(callback.episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Win rate
    win_rates = [m['win_rate'] for m in callback.episode_metrics]
    axes[0, 1].plot(win_rates)
    axes[0, 1].axhline(y=0.7, color='r', linestyle='--', label='Target 70%')
    axes[0, 1].set_title('Win Rate Over Time')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Win Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # PnL
    pnls = [m['total_pnl'] for m in callback.episode_metrics]
    axes[1, 0].plot(pnls)
    axes[1, 0].axhline(y=20, color='r', linestyle='--', label='Target $20')
    axes[1, 0].set_title('Profit/Loss Over Time')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('PnL ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Drawdown
    drawdowns = [m['max_drawdown'] for m in callback.episode_metrics]
    axes[1, 1].plot(drawdowns)
    axes[1, 1].axhline(y=0.2, color='r', linestyle='--', label='Max 20%')
    axes[1, 1].set_title('Maximum Drawdown')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Drawdown')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_evaluation_results(metrics_list):
    """Plot evaluation results"""
    if not metrics_list:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Win rate histogram
    win_rates = [m['win_rate'] for m in metrics_list]
    axes[0, 0].hist(win_rates, bins=10, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0.7, color='r', linestyle='--', label='Target 70%')
    axes[0, 0].set_title('Win Rate Distribution')
    axes[0, 0].set_xlabel('Win Rate')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # PnL histogram
    pnls = [m['total_pnl'] for m in metrics_list]
    axes[0, 1].hist(pnls, bins=10, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=20, color='r', linestyle='--', label='Target $20')
    axes[0, 1].set_title('Daily PnL Distribution')
    axes[0, 1].set_xlabel('PnL ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Drawdown vs PnL
    drawdowns = [m['max_drawdown'] for m in metrics_list]
    axes[1, 0].scatter(drawdowns, pnls, alpha=0.7)
    axes[1, 0].set_title('Drawdown vs PnL')
    axes[1, 0].set_xlabel('Max Drawdown')
    axes[1, 0].set_ylabel('PnL ($)')
    axes[1, 0].grid(True)
    
    # Balance progression
    balances = [m['final_balance'] for m in metrics_list]
    axes[1, 1].plot(balances, marker='o')
    axes[1, 1].axhline(y=30, color='g', linestyle='--', label='Success $30')
    axes[1, 1].axhline(y=5, color='r', linestyle='--', label='Margin Call $5')
    axes[1, 1].set_title('Final Balance by Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Balance ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def check_targets(metrics):
    """Check if agent meets targets"""
    print("\n🎯 TARGET ACHIEVEMENT CHECK:")
    print("=" * 40)
    
    accuracy_target = metrics['win_rate'] >= 0.70
    profit_target = metrics['total_pnl'] >= 20.0
    
    print(f"Accuracy ≥ 70%: {metrics['win_rate']:.2%} {'✅' if accuracy_target else '❌'}")
    print(f"Daily profit ≥ $20: ${metrics['total_pnl']:.2f} {'✅' if profit_target else '❌'}")
    
    if accuracy_target or profit_target:
        print("\n🎉 SUCCESS! Agent meets at least one target!")
    else:
        print("\n⚠️  Agent needs more training to meet targets.")
    
    return accuracy_target, profit_target

if __name__ == "__main__":
    # Train the agent
    model, callback = train_trading_agent()
    
    # Plot training progress
    plot_training_progress(callback)
    
    # Evaluate the best model
    print("\n" + "="*50)
    print("EVALUATING BEST MODEL")
    print("="*50)
    
    # Run multiple evaluation episodes
    env = TradingEnv()
    model = PPO.load("best_model")
    
    evaluation_metrics = []
    for i in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        metrics = env.get_metrics()
        evaluation_metrics.append(metrics)
    
    # Calculate average performance
    avg_metrics = {}
    for key in evaluation_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in evaluation_metrics])
    
    # Plot evaluation results
    plot_evaluation_results(evaluation_metrics)
    
    # Check targets
    check_targets(avg_metrics)
    
    print(f"\n📈 Final Results:")
    print(f"Average Win Rate: {avg_metrics['win_rate']:.2%}")
    print(f"Average Daily PnL: ${avg_metrics['total_pnl']:.2f}")
    print(f"Average Final Balance: ${avg_metrics['final_balance']:.2f}")
    print(f"Average Max Drawdown: {avg_metrics['max_drawdown']:.2%}")
    print(f"Average Sharpe Ratio: {avg_metrics['sharpe_ratio']:.2f}")
