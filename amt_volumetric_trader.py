"""
AMT Volumetric Strategy - Live Trading with Auto Parameter Tuning
Converted from Pine Script to Python for MT5 execution
"""

import time
import numpy as np
import pandas as pd
from exness_integration import ExnessDemo
from collections import deque
import MetaTrader5 as mt5

class AMTVolumetricTrader:
    def __init__(self):
        self.exness = ExnessDemo()
        self.connected = False
        self.initial_balance = 10.0
        
        # Adaptive parameters (will be tuned automatically)
        self.params = {
            'bins': 30,                    # Profile bins per session
            'va_percent': 70.0,            # Value Area percentage
            'vol_sma_len': 20,             # Volume SMA length
            'atr_len': 14,                 # ATR length
            'risk_mult': 1.0,              # Stop distance multiplier
            'take_mult': 1.5,              # Target distance multiplier
            'position_size': 0.1           # Position size
        }
        
        # Performance tracking for parameter tuning
        self.trade_history = deque(maxlen=50)
        self.parameter_performance = {}
        self.tuning_iteration = 0
        self.trades_per_iteration = 10
        
        # Market data storage
        self.session_data = deque(maxlen=1000)
        self.volume_bins = []
        self.session_high = None
        self.session_low = None
        self.last_session_reset = None
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
        return self.connected
    
    def get_market_data(self, symbol="ETHUSDm", timeframe=mt5.TIMEFRAME_M5, count=100):
        """Get OHLCV data with volume from MT5"""
        if not self.connected:
            return None
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def detect_session_reset(self, current_time):
        """Detect new trading session (daily reset)"""
        if self.last_session_reset is None:
            self.last_session_reset = current_time.date()
            return True
        
        if current_time.date() != self.last_session_reset:
            self.last_session_reset = current_time.date()
            return True
        
        return False
    
    def calculate_volume_profile(self, data):
        """Calculate volume profile with POC and Value Area"""
        if len(data) < 10:
            return None, None, None, None
        
        # Session range
        session_high = data['high'].max()
        session_low = data['low'].min()
        session_range = session_high - session_low
        
        if session_range <= 0:
            return None, None, None, None
        
        # Initialize volume bins
        bins = self.params['bins']
        volume_bins = [0.0] * bins
        bin_width = session_range / bins
        
        # Distribute volume to bins
        for _, row in data.iterrows():
            price = row['close']
            volume = row['tick_volume']  # Use tick_volume as proxy
            
            # Calculate bin index
            bin_idx = int((price - session_low) / session_range * (bins - 1))
            bin_idx = max(0, min(bin_idx, bins - 1))  # Clamp to valid range
            
            volume_bins[bin_idx] += volume
        
        # Find POC (Point of Control)
        poc_idx = volume_bins.index(max(volume_bins))
        poc_price = session_low + (poc_idx + 0.5) * bin_width
        
        # Calculate Value Area
        total_volume = sum(volume_bins)
        va_target = total_volume * (self.params['va_percent'] / 100.0)
        
        va_volume = volume_bins[poc_idx]
        va_min_idx = poc_idx
        va_max_idx = poc_idx
        
        left = poc_idx - 1
        right = poc_idx + 1
        
        while va_volume < va_target and (left >= 0 or right < bins):
            left_vol = volume_bins[left] if left >= 0 else 0
            right_vol = volume_bins[right] if right < bins else 0
            
            if left_vol > right_vol and left >= 0:
                va_volume += left_vol
                va_min_idx = left
                left -= 1
            elif right < bins:
                va_volume += right_vol
                va_max_idx = right
                right += 1
            else:
                break
        
        val_price = session_low + (va_min_idx + 0.5) * bin_width  # Value Area Low
        vah_price = session_low + (va_max_idx + 0.5) * bin_width  # Value Area High
        
        return poc_price, val_price, vah_price, volume_bins
    
    def calculate_cvd(self, data):
        """Calculate Cumulative Volume Delta (Order Flow Proxy)"""
        cvd = 0
        cvd_values = []
        
        for _, row in data.iterrows():
            if row['close'] > row['open']:
                up_volume = row['tick_volume']
                dn_volume = 0
            elif row['close'] < row['open']:
                up_volume = 0
                dn_volume = row['tick_volume']
            else:
                up_volume = dn_volume = row['tick_volume'] / 2
            
            cvd += (up_volume - dn_volume)
            cvd_values.append(cvd)
        
        return cvd_values
    
    def detect_market_phase(self, data, current_price, val_price, vah_price):
        """Detect market phase: Balance or Inefficiency"""
        if len(data) < self.params['vol_sma_len']:
            return "unknown"
        
        # Volume analysis
        recent_volumes = data['tick_volume'].tail(self.params['vol_sma_len'])
        vol_sma = recent_volumes.mean()
        current_volume = data['tick_volume'].iloc[-1]
        
        # Price position relative to Value Area
        is_in_value = val_price <= current_price <= vah_price
        
        if is_in_value and current_volume < vol_sma:
            return "balance"
        elif (current_price > vah_price or current_price < val_price) and current_volume > vol_sma:
            return "inefficiency"
        else:
            return "neutral"
    
    def calculate_atr(self, data):
        """Calculate Average True Range"""
        if len(data) < self.params['atr_len']:
            return 0
        
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.tail(self.params['atr_len']).mean()
        
        return atr
    
    def generate_signals(self, data, poc_price, val_price, vah_price):
        """Generate trading signals based on AMT strategy"""
        if len(data) < 10:
            return None
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        
        # Market phase
        phase = self.detect_market_phase(data, current_price, val_price, vah_price)
        
        # CVD analysis
        cvd_values = self.calculate_cvd(data.tail(10))
        cvd_rising = len(cvd_values) >= 3 and cvd_values[-1] > cvd_values[-3]
        cvd_falling = len(cvd_values) >= 3 and cvd_values[-1] < cvd_values[-3]
        
        # Price crosses
        cross_over_val = prev_price <= val_price < current_price
        cross_under_vah = prev_price >= vah_price > current_price
        
        # ATR for stops and targets
        atr = self.calculate_atr(data)
        stop_dist = atr * self.params['risk_mult']
        take_dist = atr * self.params['take_mult']
        
        # Signal generation
        signals = []
        
        # Balance Phase Signals
        if phase == "balance":
            if cross_over_val:
                signals.append({
                    'type': 'long_balance',
                    'entry': current_price,
                    'stop': current_price - stop_dist,
                    'target': current_price + take_dist,
                    'reason': 'Balance phase - cross over VAL'
                })
            
            if cross_under_vah:
                signals.append({
                    'type': 'short_balance',
                    'entry': current_price,
                    'stop': current_price + stop_dist,
                    'target': current_price - take_dist,
                    'reason': 'Balance phase - cross under VAH'
                })
        
        # Inefficiency Phase Signals
        elif phase == "inefficiency":
            if current_price > vah_price and cvd_rising:
                signals.append({
                    'type': 'long_trend',
                    'entry': current_price,
                    'stop': current_price - stop_dist,
                    'target': current_price + take_dist,
                    'reason': 'Inefficiency phase - above VAH with rising CVD'
                })
            
            if current_price < val_price and cvd_falling:
                signals.append({
                    'type': 'short_trend',
                    'entry': current_price,
                    'stop': current_price + stop_dist,
                    'target': current_price - take_dist,
                    'reason': 'Inefficiency phase - below VAL with falling CVD'
                })
        
        return signals
    
    def execute_signal(self, signal, symbol):
        """Execute trading signal"""
        volume = self.params['position_size']
        
        if 'long' in signal['type']:
            result = self.exness.place_order(
                symbol=symbol,
                order_type="buy",
                volume=volume,
                sl=signal['stop'],
                tp=signal['target']
            )
            
            if result['status'] == 'success':
                print(f"🚀 LONG: {signal['type']} | Entry: ${signal['entry']:.2f} | "
                      f"SL: ${signal['stop']:.2f} | TP: ${signal['target']:.2f}")
                print(f"   Reason: {signal['reason']}")
                return True
        
        elif 'short' in signal['type']:
            result = self.exness.place_order(
                symbol=symbol,
                order_type="sell",
                volume=volume,
                sl=signal['stop'],
                tp=signal['target']
            )
            
            if result['status'] == 'success':
                print(f"🔥 SHORT: {signal['type']} | Entry: ${signal['entry']:.2f} | "
                      f"SL: ${signal['stop']:.2f} | TP: ${signal['target']:.2f}")
                print(f"   Reason: {signal['reason']}")
                return True
        
        return False
    
    def tune_parameters(self):
        """Automatically tune parameters based on performance"""
        if len(self.trade_history) < self.trades_per_iteration:
            return
        
        recent_trades = list(self.trade_history)[-self.trades_per_iteration:]
        win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)
        avg_profit = np.mean([t['pnl'] for t in recent_trades])
        
        print(f"🔧 PARAMETER TUNING - Iteration {self.tuning_iteration}")
        print(f"   Recent Performance: {win_rate:.1%} win rate, ${avg_profit:.2f} avg profit")
        
        # Store current parameter performance
        param_key = f"iter_{self.tuning_iteration}"
        self.parameter_performance[param_key] = {
            'params': self.params.copy(),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'score': win_rate * avg_profit  # Combined score
        }
        
        # Parameter tuning logic
        if avg_profit < 0:  # Losing money
            # More conservative
            self.params['risk_mult'] = max(0.5, self.params['risk_mult'] - 0.1)
            self.params['take_mult'] = min(3.0, self.params['take_mult'] + 0.2)
            self.params['va_percent'] = min(80.0, self.params['va_percent'] + 2.0)
            print("   📉 Conservative tuning: Tighter stops, wider targets")
            
        elif win_rate < 0.4:  # Low win rate
            # Adjust risk/reward
            self.params['take_mult'] = min(3.0, self.params['take_mult'] + 0.3)
            self.params['vol_sma_len'] = max(10, self.params['vol_sma_len'] - 2)
            print("   ⚖️ Risk/Reward tuning: Wider targets, faster signals")
            
        elif win_rate > 0.7 and avg_profit < 1.0:  # High win rate, small profits
            # More aggressive
            self.params['position_size'] = min(0.2, self.params['position_size'] + 0.02)
            self.params['risk_mult'] = min(2.0, self.params['risk_mult'] + 0.1)
            print("   🚀 Aggressive tuning: Larger size, wider stops")
        
        # Adaptive parameter adjustments
        if self.tuning_iteration > 2:
            # Find best performing parameters
            best_iteration = max(self.parameter_performance.keys(), 
                               key=lambda k: self.parameter_performance[k]['score'])
            best_params = self.parameter_performance[best_iteration]['params']
            
            # Gradually move toward best parameters
            for key in ['bins', 'va_percent', 'vol_sma_len', 'atr_len']:
                current = self.params[key]
                best = best_params[key]
                self.params[key] = current + (best - current) * 0.3  # 30% adjustment
            
            print(f"   🏆 Learning from best iteration: {best_iteration}")
        
        print(f"   📋 New parameters: Risk={self.params['risk_mult']:.1f}x, "
              f"Take={self.params['take_mult']:.1f}x, VA={self.params['va_percent']:.0f}%")
        
        self.tuning_iteration += 1
    
    def monitor_trades(self):
        """Monitor closed trades for parameter tuning"""
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
                        'timestamp': time.time(),
                        'params': self.params.copy()
                    }
                    
                    self.trade_history.append(trade_record)
                    print(f"📊 Trade closed: PnL=${pnl:.2f} | Total trades: {len(self.trade_history)}")
                
                self.last_balance = current_balance
        
        self.last_tickets = current_tickets
        if not hasattr(self, 'last_balance'):
            self.last_balance = self.exness.get_account_info()['balance']
    
    def run_amt_strategy(self):
        """Run AMT Volumetric Strategy with auto parameter tuning"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        
        print("📊 AMT VOLUMETRIC STRATEGY - LIVE TRADING")
        print("=" * 60)
        print("🎯 Strategy: Volume Profile + Order Flow + Auto Tuning")
        print("📈 Features: POC, Value Area, CVD, Phase Detection")
        print("🔧 Auto-tuning: Parameters adapt every 10 trades")
        print("-" * 60)
        
        try:
            while True:
                # Get market data
                data = self.get_market_data(symbol, mt5.TIMEFRAME_M5, 100)
                if data is None:
                    time.sleep(30)
                    continue
                
                # Monitor trades
                self.monitor_trades()
                
                # Check for session reset
                current_time = pd.Timestamp.now()
                if self.detect_session_reset(current_time):
                    print(f"🔄 New session detected: {current_time.date()}")
                
                # Calculate volume profile
                poc_price, val_price, vah_price, volume_bins = self.calculate_volume_profile(data)
                
                if poc_price is None:
                    time.sleep(30)
                    continue
                
                # Generate signals
                signals = self.generate_signals(data, poc_price, val_price, vah_price)
                
                # Get current state
                positions = self.exness.get_positions()
                account_info = self.exness.get_account_info()
                current_pnl = account_info['balance'] - self.initial_balance
                current_price = data['close'].iloc[-1]
                
                # Check stop conditions
                if current_pnl >= 50.0:
                    print(f"🎉 PROFIT TARGET ACHIEVED! ${current_pnl:.2f}")
                    break
                
                if account_info['balance'] <= 1.0:
                    print(f"💀 Account protection: ${account_info['balance']:.2f}")
                    break
                
                # Display status
                phase = self.detect_market_phase(data, current_price, val_price, vah_price)
                print(f"Step {step_count}: ${current_price:.2f} | {phase.upper()} | "
                      f"POC: ${poc_price:.2f} | VAL: ${val_price:.2f} | VAH: ${vah_price:.2f} | "
                      f"Balance: ${account_info['balance']:.2f} | Positions: {len(positions)}")
                
                # Execute signals
                if signals and len(positions) < 3:
                    for signal in signals:
                        if self.execute_signal(signal, symbol):
                            break  # Only take one signal per iteration
                
                # Parameter tuning
                if len(self.trade_history) >= self.trades_per_iteration and len(self.trade_history) % self.trades_per_iteration == 0:
                    self.tune_parameters()
                
                step_count += 1
                time.sleep(60)  # 1-minute intervals for M5 strategy
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped")
        
        finally:
            # Final results
            final_account = self.exness.get_account_info()
            final_pnl = final_account['balance'] - self.initial_balance
            
            if len(self.trade_history) > 0:
                wins = sum(1 for t in self.trade_history if t['pnl'] > 0)
                win_rate = wins / len(self.trade_history)
            else:
                win_rate = 0
            
            print("\n" + "="*60)
            print("🏆 AMT VOLUMETRIC STRATEGY RESULTS")
            print("="*60)
            print(f"Initial Balance: ${self.initial_balance:.2f}")
            print(f"Final Balance: ${final_account['balance']:.2f}")
            print(f"Total Profit: ${final_pnl:.2f}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Total Trades: {len(self.trade_history)}")
            print(f"Tuning Iterations: {self.tuning_iteration}")
            
            # Show final optimized parameters
            print(f"\n📋 Final Optimized Parameters:")
            for key, value in self.params.items():
                print(f"   {key}: {value}")
            
            self.exness.close_all_positions()
            self.exness.disconnect()

if __name__ == "__main__":
    trader = AMTVolumetricTrader()
    trader.run_amt_strategy()
