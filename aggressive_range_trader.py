"""
Aggressive Range Trader - Catches Obvious Trades
Lower thresholds, more sensitive to opportunities
"""

import time
import numpy as np
from stable_baselines3 import PPO
from exness_integration import ExnessDemo

class AggressiveRangeTrader:
    def __init__(self, model_path="best_model"):
        self.model = PPO.load(model_path)
        self.exness = ExnessDemo()
        self.connected = False
        self.price_history = []
        self.initial_balance = 10.0
        
        # More aggressive parameters
        self.range_period = 15  # Shorter period for faster detection
        self.min_range_size = 5.0  # Lower minimum range ($5 vs $8)
        self.spread_buffer = 1.0  # Smaller buffer ($1 vs $2)
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
        return self.connected
    
    def detect_obvious_signals(self, prices, current_price):
        """Detect obvious trading opportunities"""
        if len(prices) < 10:
            return 0, None, None
        
        recent_prices = prices[-10:]
        
        # Find recent high/low
        recent_high = max(recent_prices)
        recent_low = min(recent_prices)
        range_size = recent_high - recent_low
        
        # Price position in recent range
        if range_size > 3.0:  # Minimum $3 range
            position_pct = (current_price - recent_low) / range_size
            
            # OBVIOUS BUY signals
            if position_pct <= 0.15:  # In bottom 15% of range
                sl = recent_low - 2.0
                tp = recent_low + (range_size * 0.7)  # Target 70% of range
                return 1, sl, tp
            
            # OBVIOUS SELL signals  
            elif position_pct >= 0.85:  # In top 15% of range
                sl = recent_high + 2.0
                tp = recent_high - (range_size * 0.7)  # Target 70% of range
                return 2, sl, tp
        
        # Momentum signals (price bouncing off levels)
        if len(prices) >= 5:
            last_5 = prices[-5:]
            
            # Strong bounce up from low
            if (current_price > last_5[-2] > last_5[-3] and 
                current_price - min(last_5) > 2.0):
                sl = min(last_5) - 2.0
                tp = current_price + 8.0  # $8 target
                return 1, sl, tp
            
            # Strong bounce down from high
            elif (current_price < last_5[-2] < last_5[-3] and 
                  max(last_5) - current_price > 2.0):
                sl = max(last_5) + 2.0
                tp = current_price - 8.0  # $8 target
                return 2, sl, tp
        
        return 0, None, None
    
    def force_trade_on_pattern(self, prices, current_price, step_count):
        """Force trades on clear patterns"""
        if len(prices) < 8:
            return 0, None, None
        
        # Every 5 steps, look for any reasonable trade
        if step_count % 5 == 0:
            recent_8 = prices[-8:]
            avg_price = np.mean(recent_8)
            
            # If price is significantly below average - BUY
            if current_price < avg_price - 3.0:
                sl = current_price - 5.0
                tp = avg_price + 2.0
                print("🔄 FORCED BUY - Price below average")
                return 1, sl, tp
            
            # If price is significantly above average - SELL
            elif current_price > avg_price + 3.0:
                sl = current_price + 5.0
                tp = avg_price - 2.0
                print("🔄 FORCED SELL - Price above average")
                return 2, sl, tp
        
        return 0, None, None
    
    def execute_aggressive_trade(self, action, symbol, current_price, sl, tp):
        """Execute trade with relaxed requirements"""
        volume = 0.1
        
        # Relax profit requirements - take any positive expectancy
        if action == 1:  # BUY
            potential_profit = tp - current_price
            potential_loss = current_price - sl
            
            # Accept even 1:1 risk reward (vs 1.5:1)
            if potential_profit > 0 and potential_loss > 0:
                result = self.exness.place_order(
                    symbol=symbol,
                    order_type="buy",
                    volume=volume,
                    sl=sl,
                    tp=tp
                )
                
                if result['status'] == 'success':
                    rr_ratio = potential_profit / potential_loss
                    print(f"🚀 AGGRESSIVE BUY: ${current_price:.2f}")
                    print(f"   SL: ${sl:.2f} | TP: ${tp:.2f} | RR: {rr_ratio:.2f}")
                    return True
                    
        elif action == 2:  # SELL
            potential_profit = current_price - tp
            potential_loss = sl - current_price
            
            # Accept even 1:1 risk reward
            if potential_profit > 0 and potential_loss > 0:
                result = self.exness.place_order(
                    symbol=symbol,
                    order_type="sell",
                    volume=volume,
                    sl=sl,
                    tp=tp
                )
                
                if result['status'] == 'success':
                    rr_ratio = potential_profit / potential_loss
                    print(f"🔥 AGGRESSIVE SELL: ${current_price:.2f}")
                    print(f"   SL: ${sl:.2f} | TP: ${tp:.2f} | RR: {rr_ratio:.2f}")
                    return True
        
        return False
    
    def run_aggressive_trading(self):
        """Run aggressive range trading"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        trades_taken = 0
        missed_opportunities = 0
        
        print("⚡ AGGRESSIVE RANGE TRADER")
        print("=" * 50)
        print("🎯 Strategy: Catch ALL obvious trades")
        print("💥 Lower thresholds, relaxed requirements")
        print("🔄 Force trades every 5 steps if needed")
        print("-" * 50)
        
        try:
            while True:
                # Get current price
                price_data = self.exness.get_price(symbol)
                if not price_data:
                    time.sleep(15)
                    continue
                
                current_price = price_data['bid']
                self.price_history.append(current_price)
                
                if len(self.price_history) > 30:
                    self.price_history.pop(0)
                
                # Get account info
                positions = self.exness.get_positions()
                account_info = self.exness.get_account_info()
                current_pnl = account_info['balance'] - self.initial_balance
                
                # Check stop conditions
                if current_pnl >= 50.0:
                    print(f"🎉 TARGET ACHIEVED! Profit: ${current_pnl:.2f}")
                    break
                
                if account_info['balance'] <= 2.0:
                    print(f"💀 Account protection: ${account_info['balance']:.2f}")
                    break
                
                # Multiple signal detection methods
                action, sl, tp = 0, None, None
                signal_type = "NONE"
                
                # Method 1: Obvious signals
                action, sl, tp = self.detect_obvious_signals(self.price_history, current_price)
                if action != 0:
                    signal_type = "OBVIOUS"
                
                # Method 2: Force trade if no positions and time passed
                if action == 0 and len(positions) == 0:
                    action, sl, tp = self.force_trade_on_pattern(self.price_history, current_price, step_count)
                    if action != 0:
                        signal_type = "FORCED"
                
                # Display status
                recent_range = max(self.price_history[-5:]) - min(self.price_history[-5:]) if len(self.price_history) >= 5 else 0
                print(f"Step {step_count}: ${current_price:.2f} | "
                      f"Range(5): ${recent_range:.2f} | "
                      f"Balance: ${account_info['balance']:.2f} | "
                      f"Positions: {len(positions)} | "
                      f"Signal: {signal_type}")
                
                # Execute trade with relaxed requirements
                if action != 0 and len(positions) < 3:  # Allow up to 3 positions
                    if self.execute_aggressive_trade(action, symbol, current_price, sl, tp):
                        trades_taken += 1
                    else:
                        missed_opportunities += 1
                        print(f"❌ Missed opportunity #{missed_opportunities}")
                
                step_count += 1
                time.sleep(15)  # Faster checks - every 15 seconds
                
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
            print(f"Trades Taken: {trades_taken}")
            print(f"Missed Opportunities: {missed_opportunities}")
            print(f"Steps: {step_count}")
            
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    trader = AggressiveRangeTrader("best_model")
    trader.run_aggressive_trading()
