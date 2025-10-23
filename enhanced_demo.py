import numpy as np
import matplotlib.pyplot as plt
from enhanced_trading_env import EnhancedTradingEnv
from stable_baselines3 import PPO
import os

def demo_enhanced_system():
    """Demonstrate the enhanced trading system"""
    
    print("🚀 Enhanced RL Trading System Demo")
    print("=" * 50)
    print("Features:")
    print("✅ Smart Money Concepts (SMC) Analysis")
    print("✅ Support/Resistance Detection")
    print("✅ Deep Learning Ensemble (LSTM, ANN, CNN, RNN)")
    print("✅ Intelligent Trade Filtering")
    print("✅ Advanced Risk Management")
    print("=" * 50)
    
    # Create environment
    env = EnhancedTradingEnv(initial_balance=10.0, leverage=2000, max_steps=500)
    
    print("\n🧠 Initializing Deep Learning Models...")
    obs, _ = env.reset()
    print("✅ Models trained and ready!")
    
    # Try to load trained model, otherwise use random actions
    model_path = './models/enhanced_ppo_best.zip'
    if os.path.exists(model_path):
        print(f"\n📦 Loading trained model: {model_path}")
        model = PPO.load(model_path)
        use_model = True
    else:
        print("\n⚠️  No trained model found. Using random actions for demo.")
        print("Run 'python enhanced_train.py' first to train models.")
        model = None
        use_model = False
    
    # Run demo episode
    print("\n🎮 Running Enhanced Trading Demo...")
    print("-" * 30)
    
    # Track performance
    balance_history = [env.balance]
    equity_history = [env.equity]
    actions_taken = []
    market_signals = []
    
    step = 0
    done = False
    
    while not done and step < 200:  # Shorter demo
        # Get action
        if use_model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        # Record data
        balance_history.append(env.balance)
        equity_history.append(env.equity)
        actions_taken.append(action)
        
        # Get market analysis from info
        if 'market_analysis' in info:
            market_signals.append(info['market_analysis']['trade_signal'])
        
        # Print periodic updates
        if step % 50 == 0:
            print(f"Step {step:3d} | Action: {action} | Balance: ${env.balance:.2f} | "
                  f"Equity: ${env.equity:.2f} | Positions: {len(env.positions)}")
            
            if 'market_analysis' in info:
                signal = info['market_analysis']['trade_signal']
                print(f"         | Market Signal: {signal['action']} "
                      f"(confidence: {signal['confidence']:.2f})")
        
        step += 1
    
    # Final results
    print("\n" + "=" * 50)
    print("📊 DEMO RESULTS")
    print("=" * 50)
    
    initial_balance = 10.0
    final_balance = env.balance
    final_equity = env.equity
    total_return = ((final_equity - initial_balance) / initial_balance) * 100
    
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance:   ${final_balance:.2f}")
    print(f"Final Equity:    ${final_equity:.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Total Steps:     {step}")
    print(f"Positions Taken: {len([a for a in actions_taken if a != 0])}")
    
    # Action distribution
    action_names = ['Hold', 'Buy Small', 'Buy Large', 'Sell Small', 'Sell Large', 'Close All']
    action_counts = [actions_taken.count(i) for i in range(6)]
    
    print(f"\nAction Distribution:")
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        percentage = (count / len(actions_taken)) * 100
        print(f"  {name}: {count} ({percentage:.1f}%)")
    
    # Market signal analysis
    if market_signals:
        signal_actions = [s['action'] for s in market_signals]
        buy_signals = signal_actions.count('buy')
        sell_signals = signal_actions.count('sell')
        hold_signals = signal_actions.count('hold')
        
        print(f"\nMarket Signals Generated:")
        print(f"  Buy Signals:  {buy_signals}")
        print(f"  Sell Signals: {sell_signals}")
        print(f"  Hold Signals: {hold_signals}")
        
        avg_confidence = np.mean([s['confidence'] for s in market_signals])
        print(f"  Average Confidence: {avg_confidence:.2f}")
    
    # Plot results
    plot_demo_results(balance_history, equity_history, actions_taken, step)
    
    print(f"\n✅ Demo completed! Check './plots/enhanced_demo_results.png' for visualization.")

def plot_demo_results(balance_history, equity_history, actions_taken, total_steps):
    """Plot demo results"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot balance and equity
    steps = range(len(balance_history))
    ax1.plot(steps, balance_history, label='Balance', color='blue', linewidth=2)
    ax1.plot(steps, equity_history, label='Equity', color='green', linewidth=2)
    ax1.set_title('Enhanced RL Trading System - Account Performance')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Account Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add action markers
    action_colors = ['gray', 'lightgreen', 'green', 'lightcoral', 'red', 'orange']
    action_names = ['Hold', 'Buy Small', 'Buy Large', 'Sell Small', 'Sell Large', 'Close All']
    
    for i, action in enumerate(actions_taken):
        if action != 0:  # Don't plot hold actions
            ax1.axvline(x=i, color=action_colors[action], alpha=0.6, linestyle='--', linewidth=1)
    
    # Plot action distribution
    action_counts = [actions_taken.count(i) for i in range(6)]
    ax2.bar(action_names, action_counts, color=action_colors)
    ax2.set_title('Action Distribution')
    ax2.set_xlabel('Actions')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, count in enumerate(action_counts):
        if count > 0:
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/enhanced_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_market_analysis_example():
    """Show an example of market analysis"""
    print("\n🔍 Market Analysis Example")
    print("-" * 30)
    
    env = EnhancedTradingEnv()
    obs, _ = env.reset()
    
    # Get market analysis for current state
    market_analysis = env._analyze_market()
    
    print("SMC Analysis:")
    smc = market_analysis['smc']
    print(f"  Swing Highs: {len(smc['swing_highs'])}")
    print(f"  Swing Lows: {len(smc['swing_lows'])}")
    print(f"  Bullish BOS: {len(smc['bos_signals']['bullish'])}")
    print(f"  Bearish BOS: {len(smc['bos_signals']['bearish'])}")
    print(f"  Order Blocks: {len(smc['order_blocks'])}")
    
    print("\nSupport/Resistance:")
    sr = market_analysis['support_resistance']
    print(f"  Support Levels: {len(sr['support'])}")
    print(f"  Resistance Levels: {len(sr['resistance'])}")
    
    if sr['support']:
        strongest_support = sr['support'][0]
        print(f"  Strongest Support: {strongest_support['price']:.5f} (strength: {strongest_support['strength']:.2f})")
    
    if sr['resistance']:
        strongest_resistance = sr['resistance'][0]
        print(f"  Strongest Resistance: {strongest_resistance['price']:.5f} (strength: {strongest_resistance['strength']:.2f})")
    
    print("\nDeep Learning Predictions:")
    dl = market_analysis['deep_learning']
    for model_name, prediction in dl['predictions'].items():
        sell_prob, hold_prob, buy_prob = prediction
        print(f"  {model_name.upper()}: Sell={sell_prob:.2f}, Hold={hold_prob:.2f}, Buy={buy_prob:.2f}")
    
    signal_names = ['Sell', 'Hold', 'Buy']
    print(f"  Ensemble Signal: {signal_names[dl['signal']]}")
    
    print("\nFinal Trade Signal:")
    signal = market_analysis['trade_signal']
    print(f"  Action: {signal['action'].upper()}")
    print(f"  Confidence: {signal['confidence']:.2f}")
    print(f"  Size: {signal['size'].upper()}")

if __name__ == "__main__":
    # Run demo
    demo_enhanced_system()
    
    # Show market analysis example
    show_market_analysis_example()
