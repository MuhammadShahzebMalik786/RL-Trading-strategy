"""
ULTIMATE TRADING BOT - All Strategies Combined
Live training, smart position management, aggressive entries, range detection
"""

import time
import numpy as np
from stable_baselines3 import PPO
from exness_integration import ExnessDemo
import MetaTrader5 as mt5
from collections import deque

class UltimateTrader:
    def __init__(self, model_path="best_model"):
        self.model = PPO.load(model_path)
        self.exness = ExnessDemo()
        self.connected = False
        self.price_history = deque(maxlen=50)
        self.initial_balance = 10.0
        
        # Performance tracking
        self.trade_history = []
        self.learning_buffer = deque(maxlen=1000)
        self.trades_since_retrain = 0
        
        # Combined parameters
        self.target_profit = 50.0
        self.target_win_rate = 0.70
        self.retrain_frequency = 20  # Retrain every 20 trades
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
            print(f"🎯 Starting balance: ${self.initial_balance}")
        return self.connected
    
    def detect_market_condition(self, prices):
        """Detect current market condition"""
        if len(prices) < 20:
            return "unknown"
        
        recent_20 = list(prices)[-20:]
        recent_10 = list(prices)[-10:]
        recent_5 = list(prices)[-5:]
        
        # Trend detection
        trend_20 = (recent_20[-1] - recent_20[0]) / recent_20[0]
        trend_10 = (recent_10[-1] - recent_10[0]) / recent_10[0]
        
        # Volatility
        volatility = np.std(recent_10) / np.mean(recent_10)
        
        # Range detection
        range_size = max(recent_20) - min(recent_20)
        
        if abs(trend_20) < 0.003 and volatility < 0.004 and range_size > 5.0:
            return "ranging"
        elif trend_20 > 0.005 and trend_10 > 0.002:
            return "strong_bullish"
        elif trend_20 > 0.002:
            return "bullish"
        elif trend_20 < -0.005 and trend_10 < -0.002:
            return "strong_bearish"
        elif trend_20 < -0.002:
            return "bearish"
        else:
            return "sideways"
    
    def get_trading_signal(self, prices, current_price, market_condition, positions, step_count):
        """Combined signal generation"""
        if len(prices) < 10:
            return 0, None, None
        
        recent_prices = list(prices)[-15:]
        
        # RANGING MARKET STRATEGY
        if market_condition == "ranging":
            recent_high = max(recent_prices)
            recent_low = min(recent_prices)
            range_size = recent_high - recent_low
            
            if range_size > 5.0:
                position_pct = (current_price - recent_low) / range_size
                
                # Aggressive range entries
                if position_pct <= 0.2 and not any(p['type'] == 'buy' for p in positions):
                    sl = recent_low - 3.0
                    tp = recent_low + (range_size * 0.8)
                    return 1, sl, tp
                elif position_pct >= 0.8 and not any(p['type'] == 'sell' for p in positions):
                    sl = recent_high + 3.0
                    tp = recent_high - (range_size * 0.8)
                    return 2, sl, tp
        
        # TRENDING MARKET STRATEGY
        elif market_condition in ["strong_bullish", "bullish"]:
            if len(positions) == 0 or step_count % 8 == 0:
                # Pullback entries in uptrend
                recent_low = min(recent_prices[-5:])
                if current_price <= recent_low + 2.0:
                    sl = current_price - 8.0
                    tp = current_price + 15.0
                    return 1, sl, tp
        
        elif market_condition in ["strong_bearish", "bearish"]:
            if len(positions) == 0 or step_count % 8 == 0:
                # Pullback entries in downtrend
                recent_high = max(recent_prices[-5:])
                if current_price >= recent_high - 2.0:
                    sl = current_price + 8.0
                    tp = current_price - 15.0
                    return 2, sl, tp
        
        # MOMENTUM STRATEGY (any market)
        if len(recent_prices) >= 5:
            last_5 = recent_prices[-5:]
            
            # Strong momentum up
            if (current_price > last_5[-2] > last_5[-3] and 
                current_price - min(last_5) > 3.0):
                sl = min(last_5) - 3.0
                tp = current_price + 10.0
                return 1, sl, tp
            
            # Strong momentum down
            elif (current_price < last_5[-2] < last_5[-3] and 
                  max(last_5) - current_price > 3.0):
                sl = max(last_5) + 3.0
                tp = current_price - 10.0
                return 2, sl, tp
        
        # FORCED ENTRIES (prevent inactivity)
        if len(positions) == 0 and step_count % 10 == 0:
            avg_price = np.mean(recent_prices)
            
            if current_price < avg_price - 2.0:
                sl = current_price - 6.0
                tp = avg_price + 3.0
                return 1, sl, tp
            elif current_price > avg_price + 2.0:
                sl = current_price + 6.0
                tp = avg_price - 3.0
                return 2, sl, tp
        
        return 0, None, None
    
    def modify_position(self, ticket, new_sl=None, new_tp=None):
        """Modify position SL/TP"""
        if not self.connected:
            return False
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": new_sl if new_sl else position.sl,
            "tp": new_tp if new_tp else position.tp,
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
    
    def manage_positions(self, current_price, market_condition):
        """Dynamic position management"""
        positions = self.exness.get_positions()
        
        for pos in positions:
            ticket = pos['ticket']
            entry_price = pos['price_open']
            position_type = pos['type']
            current_sl = pos['sl']
            current_tp = pos['tp']
            
            # Calculate profit percentage
            if position_type == 'buy':
                profit_pct = (current_price - entry_price) / entry_price
            else:
                profit_pct = (entry_price - current_price) / entry_price
            
            # Dynamic management based on profit
            if profit_pct > 0.015:  # 1.5% profit - trail stop
                if position_type == 'buy':
                    new_sl = current_price * 0.992  # Trail 0.8% below
                    new_tp = current_price * 1.025  # Extend TP
                else:
                    new_sl = current_price * 1.008  # Trail 0.8% above
                    new_tp = current_price * 0.975  # Extend TP
                
                # Only modify if significant change
                if abs(new_sl - current_sl) / current_sl > 0.003:
                    self.modify_position(ticket, new_sl, new_tp)
            
            elif profit_pct > 0.008:  # 0.8% profit - move to breakeven
                if position_type == 'buy':
                    new_sl = entry_price * 1.002  # Small buffer above entry
                else:
                    new_sl = entry_price * 0.998  # Small buffer below entry
                
                if abs(new_sl - current_sl) / current_sl > 0.005:
                    self.modify_position(ticket, new_sl, None)
    
    def execute_trade(self, action, symbol, current_price, sl, tp):
        """Execute trade with validation"""
        volume = 0.1
        
        # Validate trade parameters
        if action == 1:  # BUY
            potential_profit = tp - current_price
            potential_loss = current_price - sl
        elif action == 2:  # SELL
            potential_profit = current_price - tp
            potential_loss = sl - current_price
        else:
            return False
        
        # Accept any positive expectancy
        if potential_profit <= 0 or potential_loss <= 0:
            return False
        
        rr_ratio = potential_profit / potential_loss
        
        if action == 1:  # BUY
            result = self.exness.place_order(
                symbol=symbol,
                order_type="buy",
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result['status'] == 'success':
                print(f"🚀 BUY: ${current_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} | RR: {rr_ratio:.2f}")
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
                print(f"🔥 SELL: ${current_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} | RR: {rr_ratio:.2f}")
                return True
        
        return False
    
    def retrain_model(self):
        """Retrain model with recent performance"""
        if len(self.trade_history) < 10:
            return
        
        print("🧠 Retraining model with recent performance...")
        
        # Quick retraining
        self.model.learn(total_timesteps=3000, reset_num_timesteps=False)
        self.model.save("ultimate_model")
        
        print("💾 Model updated")
        self.trades_since_retrain = 0
    
    def calculate_performance(self):
        """Calculate current performance metrics"""
        if not self.trade_history:
            account_info = self.exness.get_account_info()
            return {
                'win_rate': 0,
                'total_pnl': account_info['balance'] - self.initial_balance,
                'balance': account_info['balance'],
                'total_trades': 0
            }
        
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        total_trades = len(self.trade_history)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        account_info = self.exness.get_account_info()
        total_pnl = account_info['balance'] - self.initial_balance
        
        return {
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'balance': account_info['balance'],
            'total_trades': total_trades
        }
    
    def monitor_closed_trades(self):
        """Monitor for closed trades"""
        positions = self.exness.get_positions()
        current_tickets = {pos['ticket'] for pos in positions}
        
        if hasattr(self, 'last_tickets'):
            closed_tickets = self.last_tickets - current_tickets
            
            for ticket in closed_tickets:
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
    
    def run_ultimate_trading(self):
        """Run ultimate combined trading strategy"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        
        print("🚀 ULTIMATE TRADING BOT - ALL STRATEGIES COMBINED")
        print("=" * 70)
        print("🎯 Target: +$50 profit OR 70% win rate (10+ trades)")
        print("🧠 Features: Live learning, Smart position management")
        print("⚡ Strategies: Range, Trend, Momentum, Forced entries")
        print("📊 Management: Dynamic SL/TP, Trailing stops")
        print("-" * 70)
        
        try:
            while True:
                # Get current price
                price_data = self.exness.get_price(symbol)
                if not price_data:
                    time.sleep(10)
                    continue
                
                current_price = price_data['bid']
                self.price_history.append(current_price)
                
                # Monitor closed trades
                self.monitor_closed_trades()
                
                # Get current state
                positions = self.exness.get_positions()
                metrics = self.calculate_performance()
                
                # Check stop conditions
                if metrics['total_pnl'] >= self.target_profit:
                    print(f"🎉 PROFIT TARGET ACHIEVED! ${metrics['total_pnl']:.2f}")
                    break
                
                if (metrics['win_rate'] >= self.target_win_rate and 
                    metrics['total_trades'] >= 10):
                    print(f"🎉 WIN RATE TARGET ACHIEVED! {metrics['win_rate']:.2%} in {metrics['total_trades']} trades")
                    break
                
                if metrics['balance'] <= 1.0:
                    print(f"💀 Account protection: ${metrics['balance']:.2f}")
                    break
                
                # Detect market condition
                market_condition = self.detect_market_condition(self.price_history)
                
                # Manage existing positions
                if positions:
                    self.manage_positions(current_price, market_condition)
                
                # Get trading signal
                action, sl, tp = self.get_trading_signal(
                    self.price_history, current_price, market_condition, positions, step_count
                )
                
                # Display status
                print(f"Step {step_count}: ${current_price:.2f} | {market_condition.upper()} | "
                      f"Balance: ${metrics['balance']:.2f} | PnL: ${metrics['total_pnl']:.2f} | "
                      f"WinRate: {metrics['win_rate']:.1%} | Positions: {len(positions)} | "
                      f"Signal: {['Hold', 'BUY', 'SELL'][action] if action < 3 else 'Hold'}")
                
                # Execute trades (max 3 positions)
                if action != 0 and len(positions) < 3:
                    self.execute_trade(action, symbol, current_price, sl, tp)
                
                # Retrain model periodically
                if self.trades_since_retrain >= self.retrain_frequency:
                    self.retrain_model()
                
                step_count += 1
                time.sleep(10)  # Fast 10-second intervals
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped")
        
        finally:
            final_metrics = self.calculate_performance()
            
            print("\n" + "="*70)
            print("🏆 ULTIMATE TRADING RESULTS")
            print("="*70)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${final_metrics['balance']:.2f}")
            print(f"Total Profit: ${final_metrics['total_pnl']:.2f}")
            print(f"Win Rate: {final_metrics['win_rate']:.2%}")
            print(f"Total Trades: {final_metrics['total_trades']}")
            print(f"Steps Executed: {step_count}")
            
            # Check achievements
            if final_metrics['total_pnl'] >= self.target_profit:
                print("🎉 PROFIT TARGET ACHIEVED!")
            if final_metrics['win_rate'] >= self.target_win_rate and final_metrics['total_trades'] >= 10:
                print("🎉 WIN RATE TARGET ACHIEVED!")
            
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    print("🚀 Ultimate Trading Bot - All Features Combined")
    print("=" * 60)
    
    try:
        trader = UltimateTrader("best_model")
        trader.run_ultimate_trading()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Ensure you have a trained model and MT5 connection")
