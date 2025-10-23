import numpy as np
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import matplotlib.pyplot as plt
from enhanced_trading_env import EnhancedTradingEnv

def train_enhanced_agent():
    """Train RL agent with enhanced market analysis"""
    
    print("🚀 Starting Enhanced RL Trading Agent Training")
    print("=" * 60)
    
    # Create environment
    env = EnhancedTradingEnv(initial_balance=10.0, leverage=2000, max_steps=1000)
    env = DummyVecEnv([lambda: env])
    
    # Evaluation environment
    eval_env = EnhancedTradingEnv(initial_balance=10.0, leverage=2000, max_steps=1000)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Training parameters
    models_config = {
        'PPO': {
            'model_class': PPO,
            'policy': 'MlpPolicy',
            'total_timesteps': 100000,
            'params': {
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'verbose': 1
            }
        },
        'SAC': {
            'model_class': SAC,
            'policy': 'MlpPolicy',
            'total_timesteps': 80000,
            'params': {
                'learning_rate': 0.0003,
                'buffer_size': 100000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'verbose': 1
            }
        },
        'A2C': {
            'model_class': A2C,
            'policy': 'MlpPolicy',
            'total_timesteps': 80000,
            'params': {
                'learning_rate': 0.0007,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'verbose': 1
            }
        }
    }
    
    trained_models = {}
    training_results = {}
    
    # Train each model
    for model_name, config in models_config.items():
        print(f"\n🤖 Training {model_name} with Enhanced Market Analysis...")
        print("-" * 40)
        
        # Create model
        model = config['model_class'](
            config['policy'],
            env,
            **config['params']
        )
        
        # Setup callbacks
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f'./models/enhanced_{model_name.lower()}_best',
            log_path=f'./logs/enhanced_{model_name.lower()}',
            eval_freq=5000,
            deterministic=True,
            render=False,
            callback_on_new_best=callback_on_best
        )
        
        # Train model
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=eval_callback
        )
        
        # Save final model
        model.save(f'./models/enhanced_{model_name.lower()}_final')
        trained_models[model_name] = model
        
        # Evaluate model
        print(f"\n📊 Evaluating {model_name}...")
        results = evaluate_model(model, eval_env, episodes=10)
        training_results[model_name] = results
        
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Average PnL: ${results['avg_pnl']:.2f}")
    
    # Create ensemble model
    print("\n🎯 Creating Ensemble Model...")
    ensemble_results = evaluate_ensemble(trained_models, eval_env, episodes=10)
    training_results['Ensemble'] = ensemble_results
    
    # Display final results
    print("\n" + "=" * 60)
    print("🏆 FINAL TRAINING RESULTS")
    print("=" * 60)
    
    for model_name, results in training_results.items():
        print(f"\n{model_name}:")
        print(f"  Average Reward: {results['avg_reward']:.2f}")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Average PnL: ${results['avg_pnl']:.2f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.1f}%")
    
    # Plot results
    plot_training_results(training_results)
    
    return trained_models, training_results

def evaluate_model(model, env, episodes: int = 10):
    """Evaluate a single model"""
    rewards = []
    pnls = []
    wins = 0
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        initial_balance = 10.0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        final_equity = info[0]['equity']
        pnl = final_equity - initial_balance
        
        rewards.append(episode_reward)
        pnls.append(pnl)
        
        if pnl > 0:
            wins += 1
    
    # Calculate metrics
    avg_reward = np.mean(rewards)
    avg_pnl = np.mean(pnls)
    win_rate = (wins / episodes) * 100
    
    # Calculate Sharpe ratio (simplified)
    returns = np.array(pnls) / initial_balance
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    # Calculate max drawdown
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
    max_drawdown = abs(np.min(drawdown)) * 100
    
    return {
        'avg_reward': avg_reward,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def evaluate_ensemble(models, env, episodes: int = 10):
    """Evaluate ensemble of models using majority voting"""
    rewards = []
    pnls = []
    wins = 0
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        initial_balance = 10.0
        
        done = False
        while not done:
            # Get predictions from all models
            actions = []
            for model in models.values():
                action, _ = model.predict(obs, deterministic=True)
                actions.append(action[0])
            
            # Majority voting
            ensemble_action = max(set(actions), key=actions.count)
            
            obs, reward, done, info = env.step([ensemble_action])
            episode_reward += reward
        
        final_equity = info[0]['equity']
        pnl = final_equity - initial_balance
        
        rewards.append(episode_reward)
        pnls.append(pnl)
        
        if pnl > 0:
            wins += 1
    
    # Calculate metrics
    avg_reward = np.mean(rewards)
    avg_pnl = np.mean(pnls)
    win_rate = (wins / episodes) * 100
    
    returns = np.array(pnls) / initial_balance
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
    max_drawdown = abs(np.min(drawdown)) * 100
    
    return {
        'avg_reward': avg_reward,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def plot_training_results(results):
    """Plot training results comparison"""
    models = list(results.keys())
    metrics = ['avg_reward', 'win_rate', 'avg_pnl', 'sharpe_ratio']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        bars = axes[i].bar(models, values, color=['blue', 'green', 'red', 'purple'])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./plots/enhanced_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create directories
    import os
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    # Train enhanced agent
    models, results = train_enhanced_agent()
    
    print("\n✅ Enhanced training completed!")
    print("Models saved in ./models/")
    print("Logs saved in ./logs/")
    print("Plots saved in ./plots/")
