"""
Range/Scalping Trader for Sideways Markets
Optimized for low volatility, tight ranges with spread consideration
"""

import time
import numpy as np
from stable_baselines3 import PPO
from exness_integration import ExnessDemo

class RangeTrader:
    def __init__(self, model_path="best_model"):
        self.model = PPO.load(model_path)
        self.exness = ExnessDemo()
        self.connected = False
        self.price_history = []
        self.initial_balance = 10.0
        
        # Range trading parameters
        self.range_period = 20  # Look at last 20 candles
        self.min_range_size = 8.0  # Minimum $8 range to trade
        self.spread_buffer = 2.0  # $2 buffer for spread
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
        return self.connected
    
    def detect_range(self, prices):
        """Detect support/resistance levels in ranging market"""
        if len(prices) < self.range_period:
            return None, None, False
        
        recent_prices = prices[-self.range_period:]
        
        # Find support (lowest low) and resistance (highest high)
        support = min(recent_prices)
        resistance = max(recent_prices)
        range_size = resistance - support
        
        # Check if it's a valid range (not too tight due to spread)
        if range_size < self.min_range_size:
            return support, resistance, False
        
        # Check if price is actually ranging (not trending)
        price_volatility = np.std(recent_prices)
        avg_price = np.mean(recent_prices)
        volatility_pct = price_volatility / avg_price
        
        # Low volatility = ranging market
        is_ranging = volatility_pct < 0.003  # Less than 0.3% volatility
        
        return support, resistance, is_ranging
    
    def calculate_range_entry(self, current_price, support, resistance):
        """Calculate optimal entry points in range"""
        range_size = resistance - support
        
        # Entry zones (avoid exact support/resistance due to spread)
        buy_zone_high = support + (range_size * 0.25)  # Buy in lower 25% of range
        sell_zone_low = resistance - (range_size * 0.25)  # Sell in upper 25% of range
        
        # Account for spread
        spread_cost = 2.0  # Approximate spread cost
        
        if current_price <= buy_zone_high:
            # Near support - consider BUY
            entry_price = current_price + spread_cost  # Account for spread
            sl = support - self.spread_buffer  # SL below support
            tp = resistance - self.spread_buffer  # TP near resistance
            
            # Check if trade is profitable after spread
            potential_profit = tp - entry_price
            potential_loss = entry_price - sl
            
            if potential_profit > potential_loss * 1.5:  # 1.5:1 RR minimum
                return 1, sl, tp  # BUY signal
        
        elif current_price >= sell_zone_low:
            # Near resistance - consider SELL
            entry_price = current_price - spread_cost  # Account for spread
            sl = resistance + self.spread_buffer  # SL above resistance
            tp = support + self.spread_buffer  # TP near support
            
            # Check if trade is profitable after spread
            potential_profit = entry_price - tp
            potential_loss = sl - entry_price
            
            if potential_profit > potential_loss * 1.5:  # 1.5:1 RR minimum
                return 2, sl, tp  # SELL signal
        
        return 0, None, None  # No trade
    
    def execute_range_trade(self, action, symbol, current_price, sl, tp):
        """Execute range trade with tight SL/TP"""
        volume = 0.1
        
        if action == 1:  # BUY
            result = self.exness.place_order(
                symbol=symbol,
                order_type="buy",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                profit_potential = (tp - current_price) * volume * 100  # Rough calculation
                print(f"📈 RANGE BUY: {volume} lots at ${current_price:.2f}")
                print(f"   SL: ${sl:.2f} | TP: ${tp:.2f} | Potential: ${profit_potential:.2f}")
                return True
                
        elif action == 2:  # SELL
            result = self.exness.place_order(
                symbol=symbol,
                order_type="sell",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                profit_potential = (current_price - tp) * volume * 100  # Rough calculation
                print(f"📉 RANGE SELL: {volume} lots at ${current_price:.2f}")
                print(f"   SL: ${sl:.2f} | TP: ${tp:.2f} | Potential: ${profit_potential:.2f}")
                return True
        
        return False
    
    def run_range_trading(self):
        """Run range trading strategy"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        trades_taken = 0
        
        print("📊 RANGE/SCALPING TRADER")
        print("=" * 50)
        print("🎯 Strategy: Buy support, Sell resistance")
        print("💰 Spread-aware entries with 1.5:1 RR minimum")
        print("⚡ Optimized for sideways markets")
        print("-" * 50)
        
        try:
            while True:
                # Get current price
                price_data = self.exness.get_price(symbol)
                if not price_data:
                    time.sleep(20)
                    continue
                
                current_price = price_data['bid']
                self.price_history.append(current_price)
                
                if len(self.price_history) > 50:
                    self.price_history.pop(0)
                
                # Detect range
                support, resistance, is_ranging = self.detect_range(self.price_history)
                
                if not is_ranging or not support or not resistance:
                    print(f"Step {step_count}: ${current_price:.2f} | NOT RANGING - Waiting...")
                    step_count += 1
                    time.sleep(30)
                    continue
                
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
                
                # Range analysis
                range_size = resistance - support
                price_position = (current_price - support) / range_size * 100  # % position in range
                
                # Calculate entry signal
                action, sl, tp = self.calculate_range_entry(current_price, support, resistance)
                
                # Display status
                print(f"Step {step_count}: ${current_price:.2f} | "
                      f"Range: ${support:.2f}-${resistance:.2f} (${range_size:.2f}) | "
                      f"Position: {price_position:.1f}% | "
                      f"Balance: ${account_info['balance']:.2f} | "
                      f"Positions: {len(positions)}")
                
                # Execute trade if signal and no conflicting positions
                if action != 0 and len(positions) < 2:  # Max 2 positions
                    if self.execute_range_trade(action, symbol, current_price, sl, tp):
                        trades_taken += 1
                
                # Show range levels
                if step_count % 10 == 0:
                    print(f"📊 Range Analysis: Support=${support:.2f}, Resistance=${resistance:.2f}")
                    print(f"   Buy Zone: <${support + range_size*0.25:.2f} | Sell Zone: >${resistance - range_size*0.25:.2f}")
                
                step_count += 1
                time.sleep(20)  # Check every 20 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped")
        
        finally:
            final_account = self.exness.get_account_info()
            final_pnl = final_account['balance'] - self.initial_balance
            
            print("\n" + "="*50)
            print("🏆 RANGE TRADING RESULTS")
            print("="*50)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${final_account['balance']:.2f}")
            print(f"Total Profit: ${final_pnl:.2f}")
            print(f"Trades Taken: {trades_taken}")
            print(f"Steps: {step_count}")
            
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    trader = RangeTrader("best_model")
    trader.run_range_trading()
