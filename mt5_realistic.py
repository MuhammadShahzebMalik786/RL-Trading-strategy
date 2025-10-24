import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib
import time
import json
from datetime import datetime

class RealisticTrader:
    def __init__(self):
        self.symbol = "ETHUSDm"
        self.lot_size = 0.1
        self.min_rr_ratio = 2.0  # Realistic 1:2 R:R
        self.lookback_period = 100
        
        # Realistic thresholds
        self.rsi_oversold = 35  # Less extreme
        self.rsi_overbought = 65  # Less extreme
        self.min_confidence = 0.5  # Lower threshold
        
        self.current_trade = None
        self.trade_history = []
        
    def get_market_data(self):
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, self.lookback_period)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        for col in ['open', 'high', 'low', 'close', 'tick_volume']:
            df[col] = df[col].astype(np.float64)
        
        return df
    
    def calculate_indicators(self, df):
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        
        try:
            rsi = talib.RSI(close, 14)
            macd, macd_signal, macd_hist = talib.MACD(close)
            atr = talib.ATR(high, low, close, 14)
            
            # Simple moving averages
            sma_20 = talib.SMA(close, 20)
            sma_50 = talib.SMA(close, 50)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20)
            
            # Simple patterns
            hammer = talib.CDLHAMMER(open_price, high, low, close)
            engulfing = talib.CDLENGULFING(open_price, high, low, close)
            
            return {
                'rsi': rsi, 'macd_hist': macd_hist, 'atr': atr,
                'sma_20': sma_20, 'sma_50': sma_50,
                'bb_upper': bb_upper, 'bb_middle': bb_middle, 'bb_lower': bb_lower,
                'hammer': hammer, 'engulfing': engulfing,
                'close': close, 'high': high, 'low': low
            }
        except Exception as e:
            print(f"❌ Indicator error: {e}")
            return None
    
    def find_simple_levels(self, df):
        """Simple support/resistance"""
        current_price = df['close'].iloc[-1]
        
        # Recent 20-period high/low
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        resistance = recent_high if recent_high > current_price else None
        support = recent_low if recent_low < current_price else None
        
        return support, resistance
    
    def analyze_realistic_setup(self, df, indicators):
        """Realistic setup with achievable conditions"""
        if indicators is None or len(df) < 50:
            return None
        
        current_price = df['close'].iloc[-1]
        rsi = indicators['rsi'][-1] if not np.isnan(indicators['rsi'][-1]) else 50
        macd_hist = indicators['macd_hist'][-1] if not np.isnan(indicators['macd_hist'][-1]) else 0
        atr = indicators['atr'][-1] if not np.isnan(indicators['atr'][-1]) else current_price * 0.02
        
        # Trend context
        sma_20 = indicators['sma_20'][-1] if not np.isnan(indicators['sma_20'][-1]) else current_price
        sma_50 = indicators['sma_50'][-1] if not np.isnan(indicators['sma_50'][-1]) else current_price
        
        # Support/Resistance
        support, resistance = self.find_simple_levels(df)
        
        setup = {
            'signal': 0,
            'confidence': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': 0,
            'reasons': []
        }
        
        # BULLISH CONDITIONS (easier to trigger)
        bullish_score = 0
        
        # RSI oversold (relaxed threshold)
        if rsi < self.rsi_oversold:
            bullish_score += 2
            setup['reasons'].append(f"RSI oversold ({rsi:.1f})")
        
        # Price below BB lower band (oversold)
        if current_price < indicators['bb_lower'][-1]:
            bullish_score += 2
            setup['reasons'].append("Below Bollinger lower band")
        
        # MACD turning positive
        if macd_hist > 0:
            bullish_score += 1
            setup['reasons'].append("MACD bullish")
        
        # Price above short MA (trend)
        if current_price > sma_20:
            bullish_score += 1
            setup['reasons'].append("Above SMA20")
        
        # Hammer pattern
        if indicators['hammer'][-1] > 0:
            bullish_score += 1
            setup['reasons'].append("Hammer pattern")
        
        # BEARISH CONDITIONS
        bearish_score = 0
        
        # RSI overbought
        if rsi > self.rsi_overbought:
            bearish_score += 2
            setup['reasons'].append(f"RSI overbought ({rsi:.1f})")
        
        # Price above BB upper band
        if current_price > indicators['bb_upper'][-1]:
            bearish_score += 2
            setup['reasons'].append("Above Bollinger upper band")
        
        # MACD turning negative
        if macd_hist < 0:
            bearish_score += 1
            setup['reasons'].append("MACD bearish")
        
        # Price below short MA
        if current_price < sma_20:
            bearish_score += 1
            setup['reasons'].append("Below SMA20")
        
        # Bearish engulfing
        if indicators['engulfing'][-1] < 0:
            bearish_score += 1
            setup['reasons'].append("Bearish engulfing")
        
        # DETERMINE SIGNAL (lower threshold = more trades)
        if bullish_score >= 3:  # Only need 3 points
            setup['signal'] = 1
            setup['confidence'] = min(bullish_score / 5.0, 1.0)
            setup['stop_loss'] = current_price - (atr * 1.5)
            setup['take_profit'] = current_price + (atr * 3.0)  # 1:2 R:R
            
        elif bearish_score >= 3:
            setup['signal'] = -1
            setup['confidence'] = min(bearish_score / 5.0, 1.0)
            setup['stop_loss'] = current_price + (atr * 1.5)
            setup['take_profit'] = current_price - (atr * 3.0)
        
        # Calculate R:R
        if setup['signal'] != 0:
            risk = abs(current_price - setup['stop_loss'])
            reward = abs(setup['take_profit'] - current_price)
            setup['risk_reward'] = reward / risk if risk > 0 else 0
        
        return setup
    
    def has_open_position(self):
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions) > 0 if positions else False
    
    def execute_trade(self, setup):
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return False, "No price"
        
        if setup['signal'] == 1:  # Buy
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "sl": setup['stop_loss'],
                "tp": setup['take_profit'],
                "deviation": 20,
                "magic": 234000,
                "comment": "Realistic Buy",
            }
        else:  # Sell
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "sl": setup['stop_loss'],
                "tp": setup['take_profit'],
                "deviation": 20,
                "magic": 234000,
                "comment": "Realistic Sell",
            }
        
        result = mt5.order_send(request)
        success = result and result.retcode == 10009
        
        if success:
            self.current_trade = {
                'timestamp': datetime.now(),
                'signal': setup['signal'],
                'entry_price': tick.ask if setup['signal'] == 1 else tick.bid,
                'stop_loss': setup['stop_loss'],
                'take_profit': setup['take_profit']
            }
            direction = "BUY" if setup['signal'] == 1 else "SELL"
            return True, f"✅ {direction} | SL: ${setup['stop_loss']:.2f} | TP: ${setup['take_profit']:.2f} | RR: 1:{setup['risk_reward']:.1f}"
        else:
            return False, "❌ Trade failed"
    
    def update_trailing_stop(self):
        """Update trailing stop loss for open positions"""
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return
        
        current_price = tick.bid if positions[0].type == mt5.ORDER_TYPE_BUY else tick.ask
        
        for pos in positions:
            ticket = pos.ticket
            entry_price = pos.price_open
            current_sl = pos.sl
            
            # Calculate PnL
            if pos.type == mt5.ORDER_TYPE_BUY:
                pnl = current_price - entry_price
            else:
                pnl = entry_price - current_price
            
            # Trailing logic: if PnL >= +0.5, move SL to +0.1 and trail by 0.1
            if pnl >= 0.5:
                if pos.type == mt5.ORDER_TYPE_BUY:
                    new_sl = entry_price + max(0.1, pnl - 0.4)  # Trail 0.4 behind PnL
                else:
                    new_sl = entry_price - max(0.1, pnl - 0.4)
                
                # Only update if new SL is better
                if ((pos.type == mt5.ORDER_TYPE_BUY and new_sl > current_sl) or 
                    (pos.type == mt5.ORDER_TYPE_SELL and new_sl < current_sl)):
                    
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": self.symbol,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": pos.tp
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == 10009:
                        print(f"🔄 Trailing SL updated: ${new_sl:.2f} (PnL: +${pnl:.2f})")
    
    def monitor_position(self):
        if not self.current_trade:
            return
        
        # Update trailing stops
        self.update_trailing_stop()
        
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            print("🏁 Trade closed")
            self.current_trade = None
            return
        
        pos = positions[0]
        elapsed = (datetime.now() - self.current_trade['timestamp']).seconds // 60
        print(f"📊 Open: PnL ${pos.profit:.2f} | Time: {elapsed}m")
    
    def start_realistic_trading(self):
        if not mt5.initialize():
            print("❌ MT5 not available")
            return
        
        print("🎯 REALISTIC TRADING SYSTEM")
        print(f"📊 Min R:R: 1:{self.min_rr_ratio}")
        print(f"🎪 RSI levels: {self.rsi_oversold}/{self.rsi_overbought}")
        print(f"✅ WILL ACTUALLY TAKE TRADES!")
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                print(f"\n🔍 Scan #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Monitor existing position
                if self.has_open_position():
                    self.monitor_position()
                    time.sleep(60)
                    continue
                
                # Get data
                df = self.get_market_data()
                if df is None:
                    print("❌ No data")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                indicators = self.calculate_indicators(df)
                if indicators is None:
                    time.sleep(60)
                    continue
                
                # Analyze setup
                setup = self.analyze_realistic_setup(df, indicators)
                
                current_price = df['close'].iloc[-1]
                rsi = indicators['rsi'][-1] if not np.isnan(indicators['rsi'][-1]) else 50
                
                print(f"💰 ETH: ${current_price:.2f} | RSI: {rsi:.1f}")
                
                if setup and setup['signal'] != 0:
                    print(f"🎯 SETUP FOUND!")
                    print(f"   Direction: {'BUY' if setup['signal'] == 1 else 'SELL'}")
                    print(f"   Confidence: {setup['confidence']:.1%}")
                    print(f"   R:R: 1:{setup['risk_reward']:.1f}")
                    print(f"   Reasons: {', '.join(setup['reasons'])}")
                    
                    # More lenient criteria
                    if (setup['confidence'] >= self.min_confidence and 
                        setup['risk_reward'] >= self.min_rr_ratio):
                        
                        success, result = self.execute_trade(setup)
                        print(f"⚡ {result}")
                    else:
                        print(f"⏸️ R:R too low: {setup['risk_reward']:.1f}")
                else:
                    print(f"⏸️ No setup (RSI: {rsi:.1f})")
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n✅ Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    trader = RealisticTrader()
    trader.start_realistic_trading()
