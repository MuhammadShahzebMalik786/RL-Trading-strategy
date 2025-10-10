"""
Aggressive Bull Market Trading Bot
Forces trades in trending markets
"""

import time
import numpy as np
from stable_baselines3 import PPO
from exness_integration import ExnessDemo
from trading_env import TradingEnv

class AggressiveTrader:
    def __init__(self, model_path="best_model"):
        self.model = PPO.load(model_path)
        self.exness = ExnessDemo()
        self.connected = False
        self.price_history = []
        self.trade_count = 0
        self.initial_balance = 10.0
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
        return self.connected
    
    def detect_trend(self, prices, period=5):
        """Detect market trend from recent prices"""
        if len(prices) < period:
            return "sideways"
        
        recent_prices = prices[-period:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if price_change > 0.002:  # 0.2% up
            return "bullish"
        elif price_change < -0.002:  # 0.2% down
            return "bearish"
        else:
            return "sideways"
    
    def force_trade_decision(self, trend, positions, step_count):
        """Force trading decisions based on trend"""
        
        # If no positions and clear trend, enter trade
        if len(positions) == 0:
            if trend == "bullish":
                return 1  # BUY
            elif trend == "bearish":
                return 2  # SELL
            elif step_count % 10 == 0:  # Force trade every 10 steps in sideways
                return 1  # Default to BUY in bull market
        
        # If have positions, let them run unless trend reverses
        if len(positions) > 0:
            current_pos_type = positions[0]['type']
            if trend == "bullish" and current_pos_type == "sell":
                return 3  # Close conflicting position
            elif trend == "bearish" and current_pos_type == "buy":
                return 3  # Close conflicting position
        
        return 0  # Hold
    
    def execute_trade(self, action, symbol, current_price):
        """Execute trade with aggressive settings"""
        volume = 0.1
        
        if action == 1:  # BUY
            sl = current_price * 0.97  # 3% stop loss (wider)
            tp = current_price * 1.06  # 6% take profit (higher)
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type="buy",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"🚀 AGGRESSIVE BUY: {volume} lots at ${current_price:.2f}")
                self.trade_count += 1
                return True
                
        elif action == 2:  # SELL
            sl = current_price * 1.03  # 3% stop loss
            tp = current_price * 0.94  # 6% take profit
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type="sell",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"🔥 AGGRESSIVE SELL: {volume} lots at ${current_price:.2f}")
                self.trade_count += 1
                return True
                
        elif action == 3:  # Close all
            if self.exness.close_all_positions():
                print("⚡ Positions closed for trend change")
                return True
        
        return False
    
    def run_aggressive_trading(self):
        """Run aggressive trading strategy"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        
        print("🔥 AGGRESSIVE BULL MARKET TRADER")
        print("=" * 50)
        print("🎯 Target: +$50 profit OR 70% win rate")
        print("⚡ Strategy: Force trades in trending markets")
        print("🚀 Bull market bias: Prefers BUY orders")
        print("-" * 50)
        
        try:
            while True:
                # Get current price
                price_data = self.exness.get_price(symbol)
                if not price_data:
                    time.sleep(30)
                    continue
                
                current_price = price_data['bid']
                self.price_history.append(current_price)
                
                # Keep only last 20 prices
                if len(self.price_history) > 20:
                    self.price_history.pop(0)
                
                # Detect trend
                trend = self.detect_trend(self.price_history)
                
                # Get positions
                positions = self.exness.get_positions()
                account_info = self.exness.get_account_info()
                
                # Calculate performance
                current_pnl = account_info['balance'] - self.initial_balance
                
                # Check stop conditions
                if current_pnl >= 50.0:
                    print(f"🎉 TARGET ACHIEVED! Profit: ${current_pnl:.2f}")
                    break
                
                if account_info['balance'] <= 0.5:
                    print(f"💀 Account depleted: ${account_info['balance']:.2f}")
                    break
                
                # Force trading decision
                action = self.force_trade_decision(trend, positions, step_count)
                
                # Display status
                print(f"Step {step_count}: ${current_price:.2f} | Trend: {trend.upper()} | "
                      f"Balance: ${account_info['balance']:.2f} | PnL: ${current_pnl:.2f} | "
                      f"Positions: {len(positions)} | Action: {['Hold', 'BUY', 'SELL', 'Close'][action]}")
                
                # Execute action
                if action != 0:
                    self.execute_trade(action, symbol, current_price)
                
                step_count += 1
                time.sleep(30)  # 30 second intervals
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped")
        
        finally:
            final_account = self.exness.get_account_info()
            final_pnl = final_account['balance'] - self.initial_balance
            
            print("\n" + "="*50)
            print("🏆 AGGRESSIVE TRADING RESULTS")
            print("="*50)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${final_account['balance']:.2f}")
            print(f"Total Profit: ${final_pnl:.2f}")
            print(f"Total Trades: {self.trade_count}")
            print(f"Steps: {step_count}")
            
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    trader = AggressiveTrader("best_model")
    trader.run_aggressive_trading()
