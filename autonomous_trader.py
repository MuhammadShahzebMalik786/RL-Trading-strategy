"""
Autonomous Self-Training RL Trading Bot
Trades live while continuously learning and improving
"""

import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from exness_integration import ExnessDemo
from trading_env import TradingEnv
import pandas as pd
from collections import deque

class AutonomousTrader:
    def __init__(self, model_path="best_model"):
        """Initialize autonomous trader"""
        self.model = PPO.load(model_path)
        self.exness = ExnessDemo()
        self.env = TradingEnv()
        self.connected = False
        
        # Performance tracking
        self.trade_history = []
        self.balance_history = []
        self.initial_balance = 10.0
        self.target_profit = 50.0  # $50 profit target
        self.target_win_rate = 0.70  # 70% win rate
        
        # Learning parameters
        self.learning_buffer = deque(maxlen=1000)  # Store experiences
        self.retrain_frequency = 50  # Retrain every 50 trades
        self.trades_since_retrain = 0
        
    def connect(self):
        """Connect to Exness demo account"""
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
            print(f"🎯 Starting balance: ${self.initial_balance}")
        return self.connected
    
    def calculate_performance_metrics(self):
        """Calculate current performance metrics"""
        if not self.trade_history:
            return {'win_rate': 0, 'total_pnl': 0, 'balance': self.initial_balance}
        
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        total_trades = len(self.trade_history)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        current_balance = self.exness.get_account_info()['balance']
        total_pnl = current_balance - self.initial_balance
        
        return {
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'balance': current_balance,
            'total_trades': total_trades
        }
    
    def check_stop_conditions(self, metrics):
        """Check if we should stop trading"""
        # Success conditions
        if metrics['total_pnl'] >= self.target_profit:
            print(f"🎉 TARGET ACHIEVED! Profit: ${metrics['total_pnl']:.2f}")
            return True
        
        if metrics['win_rate'] >= self.target_win_rate and metrics['total_trades'] >= 10:
            print(f"🎉 TARGET ACHIEVED! Win Rate: {metrics['win_rate']:.2%} in {metrics['total_trades']} trades")
            return True
        
        # Failure condition
        if metrics['balance'] <= 0.01:
            print(f"💀 ACCOUNT WIPED OUT! Balance: ${metrics['balance']:.2f}")
            return True
        
        return False
    
    def retrain_model(self):
        """Retrain the model with recent experiences"""
        if len(self.learning_buffer) < 100:
            return
        
        print("🧠 Retraining model with recent experiences...")
        
        # Create temporary environment for training
        temp_env = TradingEnv()
        
        # Train for a few steps with recent data
        self.model.learn(total_timesteps=5000, reset_num_timesteps=False)
        
        # Save updated model
        self.model.save("autonomous_model")
        print("💾 Model updated and saved")
        
        self.trades_since_retrain = 0
    
    def execute_trade(self, action, symbol, current_price):
        """Execute trade and record experience"""
        volume = 0.01
        trade_executed = False
        
        if action == 1:  # Buy
            sl = current_price * 0.98  # 2% stop loss
            tp = current_price * 1.04  # 4% take profit
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type="buy",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"✅ BUY: {volume} lots at ${current_price:.2f}")
                trade_executed = True
                
        elif action == 2:  # Sell
            sl = current_price * 1.02  # 2% stop loss
            tp = current_price * 0.96  # 4% take profit
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type="sell",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"✅ SELL: {volume} lots at ${current_price:.2f}")
                trade_executed = True
                
        elif action == 3:  # Close all
            if self.exness.close_all_positions():
                print("✅ All positions closed")
                trade_executed = True
        
        return trade_executed
    
    def monitor_positions(self):
        """Monitor and record closed positions"""
        positions = self.exness.get_positions()
        current_tickets = {pos['ticket'] for pos in positions}
        
        # Check if any positions were closed
        if hasattr(self, 'last_tickets'):
            closed_tickets = self.last_tickets - current_tickets
            
            for ticket in closed_tickets:
                # This is a simplified way to track closed trades
                # In reality, you'd need to get the actual trade result from MT5
                account_info = self.exness.get_account_info()
                current_balance = account_info['balance']
                
                if hasattr(self, 'last_balance'):
                    pnl = current_balance - self.last_balance
                    
                    trade_record = {
                        'ticket': ticket,
                        'pnl': pnl,
                        'timestamp': time.time()
                    }
                    
                    self.trade_history.append(trade_record)
                    self.trades_since_retrain += 1
                    
                    print(f"📊 Trade closed: PnL=${pnl:.2f}")
                
                self.last_balance = current_balance
        
        self.last_tickets = current_tickets
        if not hasattr(self, 'last_balance'):
            self.last_balance = self.exness.get_account_info()['balance']
    
    def run_autonomous_trading(self):
        """Run autonomous trading with continuous learning"""
        if not self.connect():
            print("❌ Failed to connect to Exness")
            return
        
        # Find trading symbol
        symbol = "ETHUSDm"
        price_data = self.exness.get_price(symbol)
        if not price_data:
            print("❌ Cannot get price data")
            return
        
        print("🤖 AUTONOMOUS TRADING STARTED")
        print("=" * 50)
        print(f"💰 Target Profit: ${self.target_profit}")
        print(f"🎯 Target Win Rate: {self.target_win_rate:.0%}")
        print(f"🔄 Retraining every {self.retrain_frequency} trades")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)
        
        step_count = 0
        
        try:
            while True:
                # Monitor closed positions
                self.monitor_positions()
                
                # Get current metrics
                metrics = self.calculate_performance_metrics()
                
                # Check stop conditions
                if self.check_stop_conditions(metrics):
                    break
                
                # Get current price
                price_data = self.exness.get_price(symbol)
                if not price_data:
                    time.sleep(30)
                    continue
                
                current_price = price_data['bid']
                
                # Get model prediction
                obs = self.env._get_observation()
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Display status
                print(f"Step {step_count}: {symbol}=${current_price:.2f}, "
                      f"Balance=${metrics['balance']:.2f}, "
                      f"PnL=${metrics['total_pnl']:.2f}, "
                      f"WinRate={metrics['win_rate']:.1%}, "
                      f"Action={['Hold', 'Buy', 'Sell', 'Close'][action]}")
                
                # Execute action
                if action != 0:
                    self.execute_trade(action, symbol, current_price)
                
                # Retrain if needed
                if self.trades_since_retrain >= self.retrain_frequency:
                    self.retrain_model()
                
                step_count += 1
                time.sleep(60)  # Wait 1 minute between decisions
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped by user")
        
        finally:
            # Final results
            final_metrics = self.calculate_performance_metrics()
            
            print("\n" + "="*60)
            print("🏆 AUTONOMOUS TRADING RESULTS")
            print("="*60)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${final_metrics['balance']:.2f}")
            print(f"Total Profit: ${final_metrics['total_pnl']:.2f}")
            print(f"Win Rate: {final_metrics['win_rate']:.2%}")
            print(f"Total Trades: {final_metrics['total_trades']}")
            print(f"Steps Executed: {step_count}")
            
            # Check if targets were met
            if final_metrics['total_pnl'] >= self.target_profit:
                print("🎉 PROFIT TARGET ACHIEVED!")
            if final_metrics['win_rate'] >= self.target_win_rate:
                print("🎉 WIN RATE TARGET ACHIEVED!")
            
            # Close remaining positions
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    print("🤖 Autonomous Self-Training Trading Bot")
    print("=" * 50)
    
    try:
        trader = AutonomousTrader("best_model")
        trader.run_autonomous_trading()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a trained model (best_model.zip)")
