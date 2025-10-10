"""
AGGRESSIVE OPPORTUNITY TRADER
Forces trades, catches every opportunity, very low thresholds
"""

import time
import numpy as np
from exness_integration import ExnessDemo
from collections import deque

class AggressiveOpportunityTrader:
    def __init__(self):
        self.exness = ExnessDemo()
        self.connected = False
        self.initial_balance = 10.0
        self.price_history = deque(maxlen=20)
        self.trade_count = 0
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
        return self.connected
    
    def detect_any_opportunity(self, prices, current_price, step_count):
        """Detect ANY trading opportunity with very low thresholds"""
        if len(prices) < 3:
            return 1, current_price - 5.0, current_price + 8.0  # Default BUY
        
        recent_prices = list(prices)[-10:]
        
        # Method 1: Price position in recent range
        if len(recent_prices) >= 5:
            recent_high = max(recent_prices)
            recent_low = min(recent_prices)
            range_size = recent_high - recent_low
            
            if range_size > 2.0:  # Very small minimum range
                position = (current_price - recent_low) / range_size
                
                # FIXED: Correct buy/sell logic
                if position <= 0.3:  # Bottom 30% - BUY (price is low)
                    sl = current_price - 5.0  # SL below entry
                    tp = current_price + 8.0  # TP above entry
                    return 1, sl, tp  # BUY
                elif position >= 0.7:  # Top 30% - SELL (price is high)
                    sl = current_price + 5.0  # SL above entry
                    tp = current_price - 8.0  # TP below entry
                    return 2, sl, tp  # SELL
        
        # Method 2: Any price movement
        if len(recent_prices) >= 3:
            price_change = current_price - recent_prices[-3]
            
            # FIXED: Buy on dips, sell on peaks
            if price_change < -2.0:  # Price dropped - BUY the dip
                sl = current_price - 6.0
                tp = current_price + 10.0
                return 1, sl, tp  # BUY
            elif price_change > 2.0:  # Price rose - SELL the peak
                sl = current_price + 6.0
                tp = current_price - 10.0
                return 2, sl, tp  # SELL
        
        # Method 3: Force trade every few steps
        if step_count % 3 == 0:  # Force trade every 3 steps
            avg_price = np.mean(recent_prices) if recent_prices else current_price
            
            # FIXED: Buy below average, sell above average
            if current_price < avg_price - 1.0:  # Below average - BUY
                sl = current_price - 5.0
                tp = current_price + 8.0
                return 1, sl, tp  # BUY
            elif current_price > avg_price + 1.0:  # Above average - SELL
                sl = current_price + 5.0
                tp = current_price - 8.0
                return 2, sl, tp  # SELL
        
        # Method 4: Momentum-based
        if len(recent_prices) >= 5:
            momentum = current_price - recent_prices[-5]
            
            # FIXED: Buy on negative momentum (oversold), sell on positive momentum (overbought)
            if momentum < -3.0:  # Strong downward momentum - BUY (oversold)
                sl = current_price - 5.0
                tp = current_price + 10.0
                return 1, sl, tp  # BUY
            elif momentum > 3.0:  # Strong upward momentum - SELL (overbought)
                sl = current_price + 5.0
                tp = current_price - 10.0
                return 2, sl, tp  # SELL
        
        return 0, None, None
    
    def execute_aggressive_trade(self, action, symbol, current_price, sl, tp):
        """Execute trade with minimal validation"""
        volume = 0.1
        
        # Very loose validation - accept almost anything
        if action == 1:  # BUY
            potential_profit = tp - current_price
            potential_loss = current_price - sl
        elif action == 2:  # SELL
            potential_profit = current_price - tp
            potential_loss = sl - current_price
        else:
            return False
        
        # Accept even negative expectancy if needed
        if potential_profit > -5.0 and potential_loss < 20.0:  # Very loose limits
            order_type = "buy" if action == 1 else "sell"
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                action_name = "BUY" if action == 1 else "SELL"
                rr = potential_profit / potential_loss if potential_loss > 0 else 0
                print(f"⚡ AGGRESSIVE {action_name}: ${current_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} | RR: {rr:.2f}")
                self.trade_count += 1
                return True
            else:
                print(f"❌ Trade failed: {result.get('message', 'Unknown error')}")
        
        return False
    
    def run_aggressive_trading(self):
        """Run extremely aggressive trading"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        missed_count = 0
        
        print("⚡ AGGRESSIVE OPPORTUNITY TRADER")
        print("=" * 50)
        print("🎯 Strategy: CATCH EVERY OPPORTUNITY")
        print("💥 Very low thresholds, force trades")
        print("🚀 Accept almost any risk/reward")
        print("-" * 50)
        
        try:
            while True:
                # Get current price
                price_data = self.exness.get_price(symbol)
                if not price_data:
                    time.sleep(10)
                    continue
                
                current_price = price_data['bid']
                self.price_history.append(current_price)
                
                # Get account info
                positions = self.exness.get_positions()
                account_info = self.exness.get_account_info()
                current_pnl = account_info['balance'] - self.initial_balance
                
                # Check stop conditions
                if current_pnl >= 50.0:
                    print(f"🎉 PROFIT TARGET ACHIEVED! ${current_pnl:.2f}")
                    break
                
                if account_info['balance'] <= 1.0:
                    print(f"💀 Account protection: ${account_info['balance']:.2f}")
                    break
                
                # ALWAYS look for opportunities
                action, sl, tp = self.detect_any_opportunity(self.price_history, current_price, step_count)
                
                # Display status
                recent_range = 0
                if len(self.price_history) >= 5:
                    recent_range = max(list(self.price_history)[-5:]) - min(list(self.price_history)[-5:])
                
                print(f"Step {step_count}: ${current_price:.2f} | Range: ${recent_range:.2f} | "
                      f"Balance: ${account_info['balance']:.2f} | PnL: ${current_pnl:.2f} | "
                      f"Positions: {len(positions)} | Trades: {self.trade_count} | "
                      f"Signal: {['Hold', 'BUY', 'SELL'][action] if action < 3 else 'Hold'}")
                
                # Execute trades - allow up to 5 positions
                if action != 0 and len(positions) < 5:
                    if self.execute_aggressive_trade(action, symbol, current_price, sl, tp):
                        print(f"✅ Trade #{self.trade_count} executed successfully")
                    else:
                        missed_count += 1
                        print(f"❌ Missed opportunity #{missed_count}")
                
                # If no trades for 10 steps, force one
                if step_count > 0 and step_count % 10 == 0 and len(positions) == 0:
                    print("🔄 FORCING TRADE - No activity detected")
                    # Force a simple buy
                    if self.execute_aggressive_trade(1, symbol, current_price, 
                                                   current_price - 8.0, current_price + 12.0):
                        print("🚀 FORCED BUY executed")
                
                step_count += 1
                time.sleep(10)  # Very fast - 10 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped")
        
        finally:
            # Final results
            final_account = self.exness.get_account_info()
            final_pnl = final_account['balance'] - self.initial_balance
            
            print("\n" + "="*50)
            print("🏆 AGGRESSIVE TRADING RESULTS")
            print("="*50)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${final_account['balance']:.2f}")
            print(f"Total Profit: ${final_pnl:.2f}")
            print(f"Trades Executed: {self.trade_count}")
            print(f"Missed Opportunities: {missed_count}")
            print(f"Steps: {step_count}")
            print(f"Trade Frequency: {self.trade_count/max(step_count,1)*100:.1f}% of steps")
            
            if self.trade_count > 0:
                avg_profit_per_trade = final_pnl / self.trade_count
                print(f"Avg Profit per Trade: ${avg_profit_per_trade:.2f}")
            
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    trader = AggressiveOpportunityTrader()
    trader.run_aggressive_trading()
