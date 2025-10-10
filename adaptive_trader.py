"""
ADAPTIVE TRADING BOT - Smart Parameter Tuning
Adjusts strategy parameters based on market performance, doesn't fight the market
"""

import time
import numpy as np
from stable_baselines3 import PPO
from exness_integration import ExnessDemo
from collections import deque

class AdaptiveTrader:
    def __init__(self, model_path="best_model"):
        self.model = PPO.load(model_path)
        self.exness = ExnessDemo()
        self.connected = False
        self.price_history = deque(maxlen=100)
        self.initial_balance = 10.0
        
        # Adaptive parameters that change based on performance
        self.params = {
            'sl_multiplier': 3.0,      # Stop loss distance multiplier
            'tp_multiplier': 6.0,      # Take profit distance multiplier
            'entry_threshold': 0.2,    # Entry threshold (0.1-0.4)
            'position_size': 0.1,      # Position size
            'max_positions': 2,        # Max concurrent positions
            'trend_sensitivity': 0.003, # Trend detection sensitivity
            'range_min_size': 5.0      # Minimum range size to trade
        }
        
        # Performance tracking for adaptation
        self.recent_trades = deque(maxlen=20)
        self.performance_window = deque(maxlen=10)
        self.last_adaptation = 0
        self.adaptation_frequency = 10  # Adapt every 10 trades
        
        # Market condition memory
        self.market_memory = {
            'ranging': {'trades': 0, 'wins': 0, 'avg_profit': 0},
            'trending': {'trades': 0, 'wins': 0, 'avg_profit': 0},
            'volatile': {'trades': 0, 'wins': 0, 'avg_profit': 0}
        }
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
        return self.connected
    
    def analyze_market_condition(self, prices):
        """Analyze current market condition for parameter adaptation"""
        if len(prices) < 30:
            return "unknown"
        
        recent_30 = list(prices)[-30:]
        recent_10 = list(prices)[-10:]
        
        # Calculate market metrics
        volatility = np.std(recent_10) / np.mean(recent_10)
        trend_strength = abs((recent_30[-1] - recent_30[0]) / recent_30[0])
        range_size = max(recent_30) - min(recent_30)
        
        # Classify market condition
        if volatility > 0.008:  # High volatility
            return "volatile"
        elif trend_strength > 0.01:  # Strong trend
            return "trending"
        elif range_size > self.params['range_min_size']:  # Clear range
            return "ranging"
        else:
            return "choppy"
    
    def adapt_parameters(self, market_condition):
        """Adapt trading parameters based on recent performance"""
        if len(self.recent_trades) < 5:
            return
        
        recent_performance = self.recent_trades[-10:] if len(self.recent_trades) >= 10 else self.recent_trades
        win_rate = sum(1 for t in recent_performance if t['pnl'] > 0) / len(recent_performance)
        avg_profit = np.mean([t['pnl'] for t in recent_performance])
        
        print(f"🔧 ADAPTING PARAMETERS - Market: {market_condition}, WinRate: {win_rate:.1%}, AvgProfit: ${avg_profit:.2f}")
        
        # Update market memory
        if market_condition in self.market_memory:
            mem = self.market_memory[market_condition]
            mem['trades'] += len(recent_performance)
            mem['wins'] += sum(1 for t in recent_performance if t['pnl'] > 0)
            mem['avg_profit'] = (mem['avg_profit'] + avg_profit) / 2
        
        # PARAMETER ADAPTATION LOGIC
        
        # 1. If losing money, be more conservative
        if avg_profit < -1.0:
            self.params['sl_multiplier'] = max(2.0, self.params['sl_multiplier'] - 0.5)  # Tighter SL
            self.params['tp_multiplier'] = min(8.0, self.params['tp_multiplier'] + 1.0)  # Wider TP
            self.params['entry_threshold'] = min(0.4, self.params['entry_threshold'] + 0.05)  # More selective
            self.params['max_positions'] = max(1, self.params['max_positions'] - 1)  # Fewer positions
            print("   📉 Conservative adaptation: Tighter SL, wider TP, fewer trades")
        
        # 2. If winning but low win rate, adjust risk/reward
        elif win_rate < 0.4 and avg_profit > 0:
            self.params['tp_multiplier'] = min(10.0, self.params['tp_multiplier'] + 1.5)  # Much wider TP
            self.params['sl_multiplier'] = max(2.0, self.params['sl_multiplier'] - 0.3)  # Slightly tighter SL
            print("   ⚖️ Risk/Reward adaptation: Wider TP for better RR")
        
        # 3. If high win rate but small profits, be more aggressive
        elif win_rate > 0.7 and avg_profit < 2.0:
            self.params['position_size'] = min(0.2, self.params['position_size'] + 0.02)  # Larger size
            self.params['max_positions'] = min(3, self.params['max_positions'] + 1)  # More positions
            self.params['entry_threshold'] = max(0.1, self.params['entry_threshold'] - 0.03)  # More trades
            print("   🚀 Aggressive adaptation: Larger size, more trades")
        
        # 4. Market-specific adaptations
        if market_condition == "ranging":
            # In ranging markets, use tighter SL/TP
            self.params['sl_multiplier'] = 2.5
            self.params['tp_multiplier'] = 4.0
            self.params['entry_threshold'] = 0.15  # More aggressive in ranges
            print("   📊 Range adaptation: Tighter SL/TP, more entries")
            
        elif market_condition == "trending":
            # In trending markets, use wider SL, let profits run
            self.params['sl_multiplier'] = 4.0
            self.params['tp_multiplier'] = 8.0
            self.params['entry_threshold'] = 0.25  # More selective in trends
            print("   📈 Trend adaptation: Wider SL/TP, selective entries")
            
        elif market_condition == "volatile":
            # In volatile markets, reduce position size and be very selective
            self.params['position_size'] = 0.05  # Smaller size
            self.params['max_positions'] = 1  # Only one position
            self.params['entry_threshold'] = 0.35  # Very selective
            self.params['sl_multiplier'] = 2.0  # Tight SL
            print("   ⚡ Volatility adaptation: Small size, tight SL, very selective")
        
        # 5. Learn from best performing market condition
        best_condition = max(self.market_memory.keys(), 
                           key=lambda k: self.market_memory[k]['avg_profit'] if self.market_memory[k]['trades'] > 0 else -999)
        
        if self.market_memory[best_condition]['trades'] > 5:
            print(f"   🏆 Best performing condition: {best_condition}")
            # Slightly bias parameters toward what works best
            if best_condition == "ranging":
                self.params['range_min_size'] = max(3.0, self.params['range_min_size'] - 0.5)
            elif best_condition == "trending":
                self.params['trend_sensitivity'] = max(0.002, self.params['trend_sensitivity'] - 0.0005)
        
        print(f"   📋 New params: SL={self.params['sl_multiplier']:.1f}x, TP={self.params['tp_multiplier']:.1f}x, "
              f"Threshold={self.params['entry_threshold']:.2f}, Size={self.params['position_size']:.2f}")
    
    def get_adaptive_signal(self, prices, current_price, market_condition, positions, step_count):
        """Generate trading signals using adaptive parameters"""
        if len(prices) < 15:
            return 0, None, None
        
        recent_prices = list(prices)[-15:]
        
        # Use adaptive thresholds
        entry_threshold = self.params['entry_threshold']
        sl_mult = self.params['sl_multiplier']
        tp_mult = self.params['tp_multiplier']
        
        # ADAPTIVE RANGING STRATEGY
        if market_condition == "ranging":
            recent_high = max(recent_prices)
            recent_low = min(recent_prices)
            range_size = recent_high - recent_low
            
            if range_size >= self.params['range_min_size']:
                position_pct = (current_price - recent_low) / range_size
                
                # Use adaptive entry threshold
                if position_pct <= entry_threshold and not any(p['type'] == 'buy' for p in positions):
                    sl = current_price - (range_size * sl_mult / 10)
                    tp = current_price + (range_size * tp_mult / 10)
                    return 1, sl, tp
                elif position_pct >= (1 - entry_threshold) and not any(p['type'] == 'sell' for p in positions):
                    sl = current_price + (range_size * sl_mult / 10)
                    tp = current_price - (range_size * tp_mult / 10)
                    return 2, sl, tp
        
        # ADAPTIVE TRENDING STRATEGY
        elif market_condition == "trending":
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(trend) > self.params['trend_sensitivity']:
                if trend > 0 and len([p for p in positions if p['type'] == 'buy']) == 0:
                    # Uptrend - buy on pullbacks
                    recent_low = min(recent_prices[-5:])
                    if current_price <= recent_low + (recent_low * entry_threshold / 10):
                        sl = current_price - (current_price * sl_mult / 100)
                        tp = current_price + (current_price * tp_mult / 100)
                        return 1, sl, tp
                        
                elif trend < 0 and len([p for p in positions if p['type'] == 'sell']) == 0:
                    # Downtrend - sell on pullbacks
                    recent_high = max(recent_prices[-5:])
                    if current_price >= recent_high - (recent_high * entry_threshold / 10):
                        sl = current_price + (current_price * sl_mult / 100)
                        tp = current_price - (current_price * tp_mult / 100)
                        return 2, sl, tp
        
        # ADAPTIVE MOMENTUM STRATEGY (any condition)
        if len(recent_prices) >= 5:
            momentum = (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]
            
            if momentum > entry_threshold / 20:  # Adaptive momentum threshold
                sl = current_price - (current_price * sl_mult / 100)
                tp = current_price + (current_price * tp_mult / 100)
                return 1, sl, tp
            elif momentum < -entry_threshold / 20:
                sl = current_price + (current_price * sl_mult / 100)
                tp = current_price - (current_price * tp_mult / 100)
                return 2, sl, tp
        
        return 0, None, None
    
    def execute_adaptive_trade(self, action, symbol, current_price, sl, tp, market_condition):
        """Execute trade with adaptive position sizing"""
        volume = self.params['position_size']
        
        # Validate trade
        if action == 1:  # BUY
            potential_profit = tp - current_price
            potential_loss = current_price - sl
        elif action == 2:  # SELL
            potential_profit = current_price - tp
            potential_loss = sl - current_price
        else:
            return False
        
        if potential_profit <= 0 or potential_loss <= 0:
            return False
        
        rr_ratio = potential_profit / potential_loss
        
        # Execute trade
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
            print(f"🎯 ADAPTIVE {action_name}: ${current_price:.2f} | SL: ${sl:.2f} | TP: ${tp:.2f} | RR: {rr_ratio:.2f} | Size: {volume}")
            
            # Record trade for adaptation
            trade_record = {
                'action': action,
                'entry_price': current_price,
                'sl': sl,
                'tp': tp,
                'volume': volume,
                'market_condition': market_condition,
                'timestamp': time.time(),
                'pnl': 0  # Will be updated when closed
            }
            
            return True
        
        return False
    
    def monitor_trades_for_adaptation(self):
        """Monitor closed trades and update records for adaptation"""
        positions = self.exness.get_positions()
        current_tickets = {pos['ticket'] for pos in positions}
        
        if hasattr(self, 'last_tickets'):
            closed_tickets = self.last_tickets - current_tickets
            
            for ticket in closed_tickets:
                account_info = self.exness.get_account_info()
                current_balance = account_info['balance']
                
                if hasattr(self, 'last_balance'):
                    pnl = current_balance - self.last_balance
                    
                    # Add to recent trades for adaptation
                    trade_record = {
                        'ticket': ticket,
                        'pnl': pnl,
                        'timestamp': time.time()
                    }
                    
                    self.recent_trades.append(trade_record)
                    print(f"📊 Trade closed: PnL=${pnl:.2f} | Recent trades: {len(self.recent_trades)}")
                
                self.last_balance = current_balance
        
        self.last_tickets = current_tickets
        if not hasattr(self, 'last_balance'):
            self.last_balance = self.exness.get_account_info()['balance']
    
    def run_adaptive_trading(self):
        """Run adaptive trading with parameter tuning"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        
        print("🧠 ADAPTIVE TRADING BOT - SMART PARAMETER TUNING")
        print("=" * 70)
        print("🎯 Target: +$50 profit OR 70% win rate (10+ trades)")
        print("🔧 Features: Parameter adaptation based on market performance")
        print("📊 Strategy: Doesn't fight market, adapts to conditions")
        print("-" * 70)
        
        try:
            while True:
                # Get current price
                price_data = self.exness.get_price(symbol)
                if not price_data:
                    time.sleep(15)
                    continue
                
                current_price = price_data['bid']
                self.price_history.append(current_price)
                
                # Monitor trades for adaptation
                self.monitor_trades_for_adaptation()
                
                # Get current state
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
                
                # Analyze market condition
                market_condition = self.analyze_market_condition(self.price_history)
                
                # Adapt parameters periodically
                if len(self.recent_trades) >= self.adaptation_frequency and len(self.recent_trades) % self.adaptation_frequency == 0:
                    if len(self.recent_trades) > self.last_adaptation:
                        self.adapt_parameters(market_condition)
                        self.last_adaptation = len(self.recent_trades)
                
                # Get adaptive trading signal
                action, sl, tp = self.get_adaptive_signal(
                    self.price_history, current_price, market_condition, positions, step_count
                )
                
                # Calculate performance metrics
                win_rate = 0
                if len(self.recent_trades) > 0:
                    wins = sum(1 for t in self.recent_trades if t['pnl'] > 0)
                    win_rate = wins / len(self.recent_trades)
                
                # Display status
                print(f"Step {step_count}: ${current_price:.2f} | {market_condition.upper()} | "
                      f"Balance: ${account_info['balance']:.2f} | PnL: ${current_pnl:.2f} | "
                      f"WinRate: {win_rate:.1%} | Positions: {len(positions)} | "
                      f"Params: SL={self.params['sl_multiplier']:.1f}x TP={self.params['tp_multiplier']:.1f}x")
                
                # Execute trades with adaptive parameters
                if action != 0 and len(positions) < self.params['max_positions']:
                    self.execute_adaptive_trade(action, symbol, current_price, sl, tp, market_condition)
                
                step_count += 1
                time.sleep(15)  # 15-second intervals
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped")
        
        finally:
            # Final results
            final_account = self.exness.get_account_info()
            final_pnl = final_account['balance'] - self.initial_balance
            final_win_rate = 0
            
            if len(self.recent_trades) > 0:
                wins = sum(1 for t in self.recent_trades if t['pnl'] > 0)
                final_win_rate = wins / len(self.recent_trades)
            
            print("\n" + "="*70)
            print("🏆 ADAPTIVE TRADING RESULTS")
            print("="*70)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${final_account['balance']:.2f}")
            print(f"Total Profit: ${final_pnl:.2f}")
            print(f"Win Rate: {final_win_rate:.2%}")
            print(f"Total Trades: {len(self.recent_trades)}")
            print(f"Steps: {step_count}")
            
            # Show final adapted parameters
            print(f"\n📋 Final Adapted Parameters:")
            for key, value in self.params.items():
                print(f"   {key}: {value}")
            
            # Show market condition performance
            print(f"\n📊 Market Condition Performance:")
            for condition, stats in self.market_memory.items():
                if stats['trades'] > 0:
                    win_rate = stats['wins'] / stats['trades']
                    print(f"   {condition}: {stats['trades']} trades, {win_rate:.1%} win rate, ${stats['avg_profit']:.2f} avg profit")
            
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    trader = AdaptiveTrader("best_model")
    trader.run_adaptive_trading()
