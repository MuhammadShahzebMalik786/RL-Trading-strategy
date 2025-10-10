"""
Live Trading with Trained RL Model on Exness Demo Account
"""

import time
import numpy as np
import torch
from stable_baselines3 import PPO
from exness_integration import ExnessDemo
from trading_env import TradingEnv
import pandas as pd

class LiveTrader:
    def __init__(self, model_path="best_model"):
        """Initialize live trader with trained model and Exness connection"""
        self.model = PPO.load(model_path)
        self.exness = ExnessDemo()
        self.env = TradingEnv()
        self.connected = False
        
    def connect(self):
        """Connect to Exness demo account"""
        self.connected = self.exness.connect()
        return self.connected
    
    def sync_account_state(self):
        """Sync environment with real account state"""
        if not self.connected:
            return False
        
        # Get real account info
        account_info = self.exness.get_account_info()
        positions = self.exness.get_positions()
        
        # Update environment to match real account
        self.env.balance = account_info['balance']
        self.env.equity = account_info['equity']
        self.env.margin_used = account_info['margin']
        
        # Convert real positions to env format
        self.env.positions = []
        for pos in positions:
            env_pos = {
                'direction': pos['type'],
                'entry_price': pos['price_open'],
                'lot_size': pos['volume'],
                'margin': pos['volume'] * self.env.margin_per_lot,
                'ticket': pos['ticket'],
                'stop_loss': pos['sl'] if pos['sl'] > 0 else None,
                'take_profit': pos['tp'] if pos['tp'] > 0 else None
            }
            self.env.positions.append(env_pos)
        
        print(f"📊 Synced - Balance: ${self.env.balance}, Positions: {len(self.env.positions)}")
        return True
    
    def execute_model_action(self, action, current_price, symbol):
        """Execute model's action on real Exness account"""
        if not self.connected:
            return False
        
        volume = 0.01  # Start with micro lots for safety
        
        if action == 1:  # Buy
            # Calculate SL/TP
            sl = current_price * (1 - self.env.stop_loss_pct)
            tp = current_price * (1 + self.env.take_profit_pct)
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type="buy",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"✅ BUY executed: {volume} lots at {current_price}")
                return True
            else:
                print(f"❌ BUY failed: {result.get('message', 'Unknown error')}")
                
        elif action == 2:  # Sell
            # Calculate SL/TP
            sl = current_price * (1 + self.env.stop_loss_pct)
            tp = current_price * (1 - self.env.take_profit_pct)
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type="sell",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"✅ SELL executed: {volume} lots at {current_price}")
                return True
            else:
                print(f"❌ SELL failed: {result.get('message', 'Unknown error')}")
                
        elif action == 3:  # Close all
            if self.exness.close_all_positions():
                print("✅ All positions closed")
                return True
            else:
                print("❌ Failed to close positions")
        
        return False
    
    def find_available_symbol(self):
        """Find available forex symbols on the account"""
        if not self.connected:
            return None
        
        # Try ETHUSDm first (your available symbol), then common forex symbols
        symbols_to_try = ["ETHUSDm", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]
        
        for symbol in symbols_to_try:
            price_data = self.exness.get_price(symbol)
            if price_data and 'bid' in price_data:
                print(f"✅ Found available symbol: {symbol}")
                return symbol
        
        print("❌ No common forex symbols found")
        return None
    
    def run_live_trading(self, duration_minutes=60):
        """Run live trading for specified duration"""
        if not self.connect():
            print("❌ Failed to connect to Exness")
            return
        
        # Find available trading symbol
        trading_symbol = self.find_available_symbol()
        if not trading_symbol:
            print("❌ No trading symbols available")
            return
        
        print(f"🚀 Starting live trading for {duration_minutes} minutes...")
        print(f"📈 Trading symbol: {trading_symbol}")
        print("🤖 Model will analyze market and make trading decisions")
        print("⚠️  Trading with micro lots (0.01) for safety")
        print("-" * 60)
        
        start_time = time.time()
        step_count = 0
        
        try:
            while time.time() - start_time < duration_minutes * 60:
                # Sync with real account
                self.sync_account_state()
                
                # Get current price and positions first
                price_data = self.exness.get_price(trading_symbol)
                if not price_data:
                    print(f"❌ Failed to get {trading_symbol} price data")
                    time.sleep(30)
                    continue
                
                current_price = price_data['bid']
                positions = self.exness.get_positions()
                
                # Get current observation
                obs = self.env._get_observation()
                
                # Get model prediction
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Debug: show raw action
                print(f"🧠 Model chose action: {action} ({['Hold', 'Buy', 'Sell', 'Close'][action]})")
                
                # Force more active trading - override if always choosing close/hold
                if step_count < 3 and action in [0, 3]:  # First few steps, encourage trading
                    if len(positions) == 0:  # No positions, encourage entry
                        # Simple momentum strategy as fallback
                        if step_count > 0:
                            price_change = current_price - self.last_price if hasattr(self, 'last_price') else 0
                            if price_change > 0:
                                action = 1  # Buy on upward movement
                                print("🔄 Overriding to BUY (momentum)")
                            elif price_change < 0:
                                action = 2  # Sell on downward movement  
                                print("🔄 Overriding to SELL (momentum)")
                            else:
                                action = 1  # Default to buy if no change
                                print("🔄 Overriding to BUY (default)")
                
                self.last_price = current_price
                
                # Get current price
                price_data = self.exness.get_price(trading_symbol)
                if not price_data:
                    print(f"❌ Failed to get {trading_symbol} price data")
                    time.sleep(30)
                    continue
                
                current_price = price_data['bid']
                
                # Display current status
                account_info = self.exness.get_account_info()
                positions = self.exness.get_positions()
                
                print(f"Step {step_count}: {trading_symbol}={current_price:.5f}, "
                      f"Balance=${account_info['balance']:.2f}, "
                      f"Positions={len(positions)}, "
                      f"Action={['Hold', 'Buy', 'Sell', 'Close'][action]}")
                
                # Execute action if not hold
                if action != 0:
                    self.execute_model_action(action, current_price, trading_symbol)
                
                step_count += 1
                
                # Wait before next decision (30 seconds)
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped by user")
        
        finally:
            # Final account status
            final_account = self.exness.get_account_info()
            initial_balance = 10.0
            profit = final_account['balance'] - initial_balance
            
            print("\n" + "="*60)
            print("📊 LIVE TRADING RESULTS")
            print("="*60)
            print(f"Initial Balance: ${initial_balance}")
            print(f"Final Balance: ${final_account['balance']:.2f}")
            print(f"Profit/Loss: ${profit:.2f}")
            print(f"Return: {(profit/initial_balance)*100:.2f}%")
            print(f"Steps Executed: {step_count}")
            
            # Close any remaining positions
            if self.exness.get_positions():
                print("🔄 Closing remaining positions...")
                self.exness.close_all_positions()
            
            self.exness.disconnect()

if __name__ == "__main__":
    print("🤖 RL Live Trading Bot")
    print("=" * 40)
    
    # Try to load best model first, then final model
    model_loaded = False
    for model_name in ["best_model", "final_model"]:
        try:
            trader = LiveTrader(model_name)
            print(f"✅ Model loaded successfully: {model_name}")
            model_loaded = True
            break
        except:
            print(f"⚠️  {model_name} not found, trying next...")
            continue
    
    if not model_loaded:
        print("❌ No trained model found. Train the model first with: python train_agent.py")
        exit()
    
    # Start live trading
    duration = int(input("Enter trading duration in minutes (default 60): ") or 60)
    trader.run_live_trading(duration)
