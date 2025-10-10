"""
Smart Trading Bot with Dynamic Position Management
Constantly monitors and adjusts SL/TP based on market conditions
"""

import time
import numpy as np
from stable_baselines3 import PPO
from exness_integration import ExnessDemo
import MetaTrader5 as mt5

class SmartTrader:
    def __init__(self, model_path="best_model"):
        self.model = PPO.load(model_path)
        self.exness = ExnessDemo()
        self.connected = False
        self.price_history = []
        self.initial_balance = 10.0
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
        return self.connected
    
    def modify_position(self, ticket, new_sl=None, new_tp=None):
        """Modify existing position's SL/TP"""
        if not self.connected:
            return False
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        
        # Prepare modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": new_sl if new_sl else position.sl,
            "tp": new_tp if new_tp else position.tp,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Modified position {ticket}: SL=${new_sl:.2f}, TP=${new_tp:.2f}")
            return True
        else:
            print(f"❌ Failed to modify {ticket}: {result.comment}")
            return False
    
    def calculate_dynamic_sltp(self, position, current_price, trend):
        """Calculate dynamic SL/TP based on market conditions"""
        entry_price = position['price_open']
        position_type = position['type']
        
        # Calculate current profit in pips
        if position_type == 'buy':
            current_profit_pct = (current_price - entry_price) / entry_price
        else:
            current_profit_pct = (entry_price - current_price) / entry_price
        
        # Dynamic SL/TP based on profit and trend
        if position_type == 'buy':
            # BUY position management
            if current_profit_pct > 0.02:  # 2% profit - trail stop
                new_sl = current_price * 0.99  # Trail 1% below current price
                new_tp = current_price * 1.04  # Extend TP to 4% above current
            elif current_profit_pct > 0.01:  # 1% profit - move to breakeven
                new_sl = entry_price * 1.001  # Breakeven + small buffer
                new_tp = current_price * 1.03  # 3% above current
            else:  # Still at risk
                new_sl = entry_price * 0.97  # 3% stop loss
                new_tp = entry_price * 1.06  # 6% take profit
                
            # Trend-based adjustments
            if trend == "bullish":
                new_tp = new_tp * 1.02  # Extend TP in strong bull trend
            elif trend == "bearish":
                new_sl = max(new_sl, current_price * 0.985)  # Tighter SL in bear trend
                
        else:  # SELL position
            if current_profit_pct > 0.02:  # 2% profit - trail stop
                new_sl = current_price * 1.01  # Trail 1% above current price
                new_tp = current_price * 0.96  # Extend TP to 4% below current
            elif current_profit_pct > 0.01:  # 1% profit - move to breakeven
                new_sl = entry_price * 0.999  # Breakeven + small buffer
                new_tp = current_price * 0.97  # 3% below current
            else:  # Still at risk
                new_sl = entry_price * 1.03  # 3% stop loss
                new_tp = entry_price * 0.94  # 6% take profit
                
            # Trend-based adjustments
            if trend == "bearish":
                new_tp = new_tp * 0.98  # Extend TP in strong bear trend
            elif trend == "bullish":
                new_sl = min(new_sl, current_price * 1.015)  # Tighter SL in bull trend
        
        return new_sl, new_tp
    
    def monitor_positions(self, current_price, trend):
        """Monitor and adjust all open positions"""
        positions = self.exness.get_positions()
        
        for pos in positions:
            ticket = pos['ticket']
            current_sl = pos['sl']
            current_tp = pos['tp']
            
            # Calculate new SL/TP
            new_sl, new_tp = self.calculate_dynamic_sltp(pos, current_price, trend)
            
            # Only modify if significant change (avoid too frequent modifications)
            sl_change = abs(new_sl - current_sl) / current_sl if current_sl > 0 else 1
            tp_change = abs(new_tp - current_tp) / current_tp if current_tp > 0 else 1
            
            if sl_change > 0.005 or tp_change > 0.005:  # 0.5% minimum change
                self.modify_position(ticket, new_sl, new_tp)
                
                # Show position status
                profit = pos['profit']
                print(f"📊 Position {ticket}: Profit=${profit:.2f}, "
                      f"SL: ${current_sl:.2f}→${new_sl:.2f}, "
                      f"TP: ${current_tp:.2f}→${new_tp:.2f}")
    
    def detect_trend(self, prices, period=10):
        """Enhanced trend detection"""
        if len(prices) < period:
            return "sideways"
        
        recent_prices = prices[-period:]
        
        # Calculate multiple trend indicators
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        momentum = (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]
        
        # Strong trend thresholds
        if price_change > 0.005 and momentum > 0.002:
            return "strong_bullish"
        elif price_change > 0.002:
            return "bullish"
        elif price_change < -0.005 and momentum < -0.002:
            return "strong_bearish"
        elif price_change < -0.002:
            return "bearish"
        else:
            return "sideways"
    
    def should_enter_trade(self, trend, positions, step_count):
        """Smart entry decision"""
        if len(positions) >= 2:  # Max 2 positions
            return 0
        
        # Entry based on strong trends
        if trend == "strong_bullish" and not any(p['type'] == 'buy' for p in positions):
            return 1  # BUY
        elif trend == "strong_bearish" and not any(p['type'] == 'sell' for p in positions):
            return 2  # SELL
        elif trend in ["bullish", "bearish"] and len(positions) == 0 and step_count % 5 == 0:
            return 1 if trend == "bullish" else 2
        
        return 0  # Hold
    
    def execute_trade(self, action, symbol, current_price):
        """Execute trade with smart initial SL/TP"""
        volume = 0.1
        
        if action == 1:  # BUY
            sl = current_price * 0.97  # 3% initial SL
            tp = current_price * 1.06  # 6% initial TP
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type="buy",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"🚀 SMART BUY: {volume} lots at ${current_price:.2f}")
                return True
                
        elif action == 2:  # SELL
            sl = current_price * 1.03  # 3% initial SL
            tp = current_price * 0.94  # 6% initial TP
            
            result = self.exness.place_order(
                symbol=symbol,
                order_type="sell",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"🔥 SMART SELL: {volume} lots at ${current_price:.2f}")
                return True
        
        return False
    
    def run_smart_trading(self):
        """Run smart trading with constant position monitoring"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        
        print("🧠 SMART POSITION MANAGEMENT TRADER")
        print("=" * 60)
        print("🎯 Target: +$50 profit OR account preservation")
        print("⚡ Features: Dynamic SL/TP, Trailing stops, Breakeven moves")
        print("📊 Monitoring: Constant position adjustment")
        print("-" * 60)
        
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
                
                # Detect trend
                trend = self.detect_trend(self.price_history)
                
                # Get account info and positions
                positions = self.exness.get_positions()
                account_info = self.exness.get_account_info()
                current_pnl = account_info['balance'] - self.initial_balance
                
                # CONSTANT POSITION MONITORING
                if positions:
                    self.monitor_positions(current_price, trend)
                
                # Check stop conditions
                if current_pnl >= 50.0:
                    print(f"🎉 TARGET ACHIEVED! Profit: ${current_pnl:.2f}")
                    break
                
                if account_info['balance'] <= 1.0:
                    print(f"💀 Account protection: ${account_info['balance']:.2f}")
                    break
                
                # Entry decision
                action = self.should_enter_trade(trend, positions, step_count)
                
                # Display status
                total_profit = sum(p['profit'] for p in positions)
                print(f"Step {step_count}: ${current_price:.2f} | {trend.upper()} | "
                      f"Balance: ${account_info['balance']:.2f} | PnL: ${current_pnl:.2f} | "
                      f"Positions: {len(positions)} | Unrealized: ${total_profit:.2f}")
                
                # Execute new trades
                if action != 0:
                    self.execute_trade(action, symbol, current_price)
                
                step_count += 1
                time.sleep(15)  # Check every 15 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped")
        
        finally:
            final_account = self.exness.get_account_info()
            final_pnl = final_account['balance'] - self.initial_balance
            
            print("\n" + "="*60)
            print("🏆 SMART TRADING RESULTS")
            print("="*60)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${final_account['balance']:.2f}")
            print(f"Total Profit: ${final_pnl:.2f}")
            print(f"Steps: {step_count}")
            
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    trader = SmartTrader("best_model")
    trader.run_smart_trading()
