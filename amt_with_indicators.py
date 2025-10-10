"""
AMT Volumetric Strategy with Real-time Indicator Plotting
Shows POC, Value Area, CVD, and Volume Profile on live charts
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from exness_integration import ExnessDemo
from collections import deque
import MetaTrader5 as mt5
from datetime import datetime

class AMTIndicatorTrader:
    def __init__(self):
        self.exness = ExnessDemo()
        self.connected = False
        self.initial_balance = 10.0
        
        # Parameters
        self.params = {
            'bins': 30,
            'va_percent': 70.0,
            'vol_sma_len': 20,
            'atr_len': 14,
            'risk_mult': 1.0,
            'take_mult': 1.5,
            'position_size': 0.1
        }
        
        # Data storage for plotting
        self.price_data = deque(maxlen=100)
        self.volume_data = deque(maxlen=100)
        self.poc_data = deque(maxlen=100)
        self.val_data = deque(maxlen=100)
        self.vah_data = deque(maxlen=100)
        self.cvd_data = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        
        # Trading data
        self.trade_history = deque(maxlen=50)
        self.signals = deque(maxlen=20)
        
        # Setup plotting
        self.setup_plots()
        
    def connect(self):
        self.connected = self.exness.connect()
        if self.connected:
            account_info = self.exness.get_account_info()
            self.initial_balance = account_info['balance']
        return self.connected
    
    def setup_plots(self):
        """Setup real-time plotting"""
        plt.style.use('dark_background')
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main price chart with indicators
        self.ax1.set_title('AMT Strategy - Price & Value Area', color='white', fontsize=14)
        self.ax1.set_ylabel('Price', color='white')
        self.ax1.grid(True, alpha=0.3)
        
        # Volume Profile
        self.ax2.set_title('Volume Profile', color='white', fontsize=14)
        self.ax2.set_xlabel('Volume')
        self.ax2.set_ylabel('Price')
        self.ax2.grid(True, alpha=0.3)
        
        # CVD (Order Flow)
        self.ax3.set_title('Cumulative Volume Delta (CVD)', color='white', fontsize=14)
        self.ax3.set_ylabel('CVD')
        self.ax3.grid(True, alpha=0.3)
        
        # Performance & Signals
        self.ax4.set_title('Trading Signals & Performance', color='white', fontsize=14)
        self.ax4.set_ylabel('Balance')
        self.ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
    
    def get_market_data(self, symbol="ETHUSDm", timeframe=mt5.TIMEFRAME_M5, count=100):
        """Get OHLCV data from MT5"""
        if not self.connected:
            return None
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_volume_profile(self, data):
        """Calculate volume profile with POC and Value Area"""
        if len(data) < 10:
            return None, None, None, None
        
        session_high = data['high'].max()
        session_low = data['low'].min()
        session_range = session_high - session_low
        
        if session_range <= 0:
            return None, None, None, None
        
        bins = self.params['bins']
        volume_bins = [0.0] * bins
        bin_width = session_range / bins
        
        # Distribute volume to bins
        for _, row in data.iterrows():
            price = row['close']
            volume = row['tick_volume']
            
            bin_idx = int((price - session_low) / session_range * (bins - 1))
            bin_idx = max(0, min(bin_idx, bins - 1))
            
            volume_bins[bin_idx] += volume
        
        # Find POC
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
        
        val_price = session_low + (va_min_idx + 0.5) * bin_width
        vah_price = session_low + (va_max_idx + 0.5) * bin_width
        
        # Return price levels and volume profile for plotting
        price_levels = [session_low + (i + 0.5) * bin_width for i in range(bins)]
        
        return poc_price, val_price, vah_price, list(zip(price_levels, volume_bins))
    
    def calculate_cvd(self, data):
        """Calculate Cumulative Volume Delta"""
        cvd = 0
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
        
        return cvd
    
    def detect_market_phase(self, data, current_price, val_price, vah_price):
        """Detect market phase"""
        if len(data) < self.params['vol_sma_len']:
            return "unknown"
        
        recent_volumes = data['tick_volume'].tail(self.params['vol_sma_len'])
        vol_sma = recent_volumes.mean()
        current_volume = data['tick_volume'].iloc[-1]
        
        is_in_value = val_price <= current_price <= vah_price
        
        if is_in_value and current_volume < vol_sma:
            return "balance"
        elif (current_price > vah_price or current_price < val_price) and current_volume > vol_sma:
            return "inefficiency"
        else:
            return "neutral"
    
    def generate_signals(self, data, poc_price, val_price, vah_price):
        """Generate trading signals"""
        if len(data) < 10:
            return []
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        
        phase = self.detect_market_phase(data, current_price, val_price, vah_price)
        
        # CVD analysis
        cvd_current = self.calculate_cvd(data.tail(1))
        cvd_prev = self.calculate_cvd(data.tail(4).head(3))
        cvd_rising = cvd_current > cvd_prev
        
        # Price crosses
        cross_over_val = prev_price <= val_price < current_price
        cross_under_vah = prev_price >= vah_price > current_price
        
        # ATR for stops
        high_low = data['high'] - data['low']
        atr = high_low.tail(self.params['atr_len']).mean()
        
        signals = []
        
        if phase == "balance":
            if cross_over_val:
                signals.append({
                    'type': 'LONG',
                    'entry': current_price,
                    'stop': current_price - (atr * self.params['risk_mult']),
                    'target': current_price + (atr * self.params['take_mult']),
                    'reason': 'Balance: Cross over VAL'
                })
            
            if cross_under_vah:
                signals.append({
                    'type': 'SHORT',
                    'entry': current_price,
                    'stop': current_price + (atr * self.params['risk_mult']),
                    'target': current_price - (atr * self.params['take_mult']),
                    'reason': 'Balance: Cross under VAH'
                })
        
        elif phase == "inefficiency":
            if current_price > vah_price and cvd_rising:
                signals.append({
                    'type': 'LONG',
                    'entry': current_price,
                    'stop': current_price - (atr * self.params['risk_mult']),
                    'target': current_price + (atr * self.params['take_mult']),
                    'reason': 'Inefficiency: Above VAH + Rising CVD'
                })
            
            if current_price < val_price and not cvd_rising:
                signals.append({
                    'type': 'SHORT',
                    'entry': current_price,
                    'stop': current_price + (atr * self.params['risk_mult']),
                    'target': current_price - (atr * self.params['take_mult']),
                    'reason': 'Inefficiency: Below VAL + Falling CVD'
                })
        
        return signals
    
    def update_plots(self, data, poc_price, val_price, vah_price, volume_profile, cvd, phase):
        """Update all plots with current data"""
        if len(self.timestamps) == 0:
            return
        
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot 1: Price chart with indicators
        self.ax1.set_title('AMT Strategy - Price & Value Area', color='white', fontsize=14)
        
        if len(self.price_data) > 1:
            # Price line
            self.ax1.plot(list(self.timestamps), list(self.price_data), 'white', linewidth=2, label='Price')
            
            # POC line
            if len(self.poc_data) > 0:
                self.ax1.plot(list(self.timestamps), list(self.poc_data), 'orange', linewidth=2, label='POC')
            
            # Value Area
            if len(self.val_data) > 0 and len(self.vah_data) > 0:
                self.ax1.plot(list(self.timestamps), list(self.val_data), 'green', linewidth=1, label='VAL')
                self.ax1.plot(list(self.timestamps), list(self.vah_data), 'red', linewidth=1, label='VAH')
                
                # Fill Value Area
                self.ax1.fill_between(list(self.timestamps), list(self.val_data), list(self.vah_data), 
                                     alpha=0.2, color='blue', label='Value Area')
        
        # Phase background color
        if phase == "balance":
            self.ax1.axhspan(min(self.price_data) if self.price_data else 0, 
                           max(self.price_data) if self.price_data else 1, 
                           alpha=0.1, color='teal')
        elif phase == "inefficiency":
            self.ax1.axhspan(min(self.price_data) if self.price_data else 0, 
                           max(self.price_data) if self.price_data else 1, 
                           alpha=0.1, color='red')
        
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylabel('Price')
        
        # Plot 2: Volume Profile
        self.ax2.set_title('Volume Profile', color='white', fontsize=14)
        
        if volume_profile:
            prices, volumes = zip(*volume_profile)
            self.ax2.barh(prices, volumes, height=(max(prices) - min(prices)) / len(prices), 
                         color='cyan', alpha=0.7)
            
            # Highlight POC
            if poc_price:
                self.ax2.axhline(y=poc_price, color='orange', linewidth=3, label='POC')
            if val_price:
                self.ax2.axhline(y=val_price, color='green', linewidth=2, label='VAL')
            if vah_price:
                self.ax2.axhline(y=vah_price, color='red', linewidth=2, label='VAH')
        
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlabel('Volume')
        self.ax2.set_ylabel('Price')
        
        # Plot 3: CVD
        self.ax3.set_title('Cumulative Volume Delta (CVD)', color='white', fontsize=14)
        
        if len(self.cvd_data) > 1:
            self.ax3.plot(list(self.timestamps), list(self.cvd_data), 'yellow', linewidth=2)
            
            # CVD trend
            if len(self.cvd_data) >= 3:
                if self.cvd_data[-1] > self.cvd_data[-3]:
                    self.ax3.text(0.02, 0.95, 'CVD: RISING ↗', transform=self.ax3.transAxes, 
                                color='green', fontsize=12, fontweight='bold')
                else:
                    self.ax3.text(0.02, 0.95, 'CVD: FALLING ↘', transform=self.ax3.transAxes, 
                                color='red', fontsize=12, fontweight='bold')
        
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylabel('CVD')
        
        # Plot 4: Performance & Signals
        self.ax4.set_title('Trading Performance & Signals', color='white', fontsize=14)
        
        # Account balance
        account_info = self.exness.get_account_info()
        current_balance = account_info['balance']
        
        # Show current stats
        pnl = current_balance - self.initial_balance
        positions = self.exness.get_positions()
        
        stats_text = f"Balance: ${current_balance:.2f}\n"
        stats_text += f"PnL: ${pnl:.2f}\n"
        stats_text += f"Positions: {len(positions)}\n"
        stats_text += f"Phase: {phase.upper()}"
        
        self.ax4.text(0.02, 0.95, stats_text, transform=self.ax4.transAxes, 
                     color='white', fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Recent signals
        if len(self.signals) > 0:
            signal_text = "Recent Signals:\n"
            for i, signal in enumerate(list(self.signals)[-3:]):  # Last 3 signals
                signal_text += f"{signal['type']}: {signal['reason'][:30]}...\n"
            
            self.ax4.text(0.02, 0.5, signal_text, transform=self.ax4.transAxes, 
                         color='cyan', fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        self.ax4.grid(True, alpha=0.3)
        
        # Refresh display
        plt.tight_layout()
        plt.pause(0.1)
    
    def execute_signal(self, signal, symbol):
        """Execute trading signal"""
        volume = self.params['position_size']
        
        if signal['type'] == 'LONG':
            result = self.exness.place_order(
                symbol=symbol,
                order_type="buy",
                volume=volume,
                sl=signal['stop'],
                tp=signal['target']
            )
        else:
            result = self.exness.place_order(
                symbol=symbol,
                order_type="sell",
                volume=volume,
                sl=signal['stop'],
                tp=signal['target']
            )
        
        if result['status'] == 'success':
            print(f"🎯 {signal['type']}: {signal['reason']}")
            print(f"   Entry: ${signal['entry']:.2f} | SL: ${signal['stop']:.2f} | TP: ${signal['target']:.2f}")
            return True
        
        return False
    
    def run_amt_with_indicators(self):
        """Run AMT strategy with live indicator plotting"""
        if not self.connect():
            print("❌ Failed to connect")
            return
        
        symbol = "ETHUSDm"
        step_count = 0
        
        print("📊 AMT STRATEGY WITH LIVE INDICATORS")
        print("=" * 50)
        print("📈 Real-time plotting: Price, Volume Profile, CVD")
        print("🎯 Visual signals and performance tracking")
        print("-" * 50)
        
        try:
            while True:
                # Get market data
                data = self.get_market_data(symbol, mt5.TIMEFRAME_M5, 50)
                if data is None:
                    time.sleep(30)
                    continue
                
                # Calculate indicators
                poc_price, val_price, vah_price, volume_profile = self.calculate_volume_profile(data)
                
                if poc_price is None:
                    time.sleep(30)
                    continue
                
                cvd = self.calculate_cvd(data)
                current_price = data['close'].iloc[-1]
                current_time = datetime.now()
                
                # Store data for plotting
                self.price_data.append(current_price)
                self.poc_data.append(poc_price)
                self.val_data.append(val_price)
                self.vah_data.append(vah_price)
                self.cvd_data.append(cvd)
                self.timestamps.append(current_time)
                
                # Detect phase
                phase = self.detect_market_phase(data, current_price, val_price, vah_price)
                
                # Generate signals
                signals = self.generate_signals(data, poc_price, val_price, vah_price)
                
                # Store signals
                for signal in signals:
                    self.signals.append(signal)
                
                # Update plots
                self.update_plots(data, poc_price, val_price, vah_price, volume_profile, cvd, phase)
                
                # Execute signals
                positions = self.exness.get_positions()
                if signals and len(positions) < 2:
                    for signal in signals:
                        if self.execute_signal(signal, symbol):
                            break
                
                # Check stop conditions
                account_info = self.exness.get_account_info()
                current_pnl = account_info['balance'] - self.initial_balance
                
                if current_pnl >= 50.0:
                    print(f"🎉 PROFIT TARGET ACHIEVED! ${current_pnl:.2f}")
                    break
                
                if account_info['balance'] <= 1.0:
                    print(f"💀 Account protection: ${account_info['balance']:.2f}")
                    break
                
                print(f"Step {step_count}: ${current_price:.2f} | {phase.upper()} | "
                      f"POC: ${poc_price:.2f} | Balance: ${account_info['balance']:.2f}")
                
                step_count += 1
                time.sleep(30)  # 30-second updates
                
        except KeyboardInterrupt:
            print("\n⏹️  Trading stopped")
        
        finally:
            self.exness.close_all_positions()
            self.exness.disconnect()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    trader = AMTIndicatorTrader()
    trader.run_amt_with_indicators()
