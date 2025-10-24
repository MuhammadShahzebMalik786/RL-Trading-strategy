import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib
import time
from datetime import datetime

class SwingTrader:
    def __init__(self):
        self.symbol = "ETHUSDm"
        self.lot_size = 0.1
        self.min_rr_ratio = 3.0
        self.lookback_period = 100
        self.last_trade_time = None
        self.min_trade_interval = 300
        self.trailing_positions = {}  # Track positions for trailing SL
        
    def get_market_data(self):
        """Get market data with proper data types"""
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, self.lookback_period)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        # Convert to float64 for TA-Lib
        df['open'] = df['open'].astype(np.float64)
        df['high'] = df['high'].astype(np.float64)
        df['low'] = df['low'].astype(np.float64)
        df['close'] = df['close'].astype(np.float64)
        df['tick_volume'] = df['tick_volume'].astype(np.float64)
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate technical indicators with proper data types"""
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['tick_volume'].values.astype(np.float64)
        open_price = df['open'].values.astype(np.float64)
        
        try:
            # Basic indicators
            rsi = talib.RSI(close, 14)
            macd, macd_signal, macd_hist = talib.MACD(close)
            atr = talib.ATR(high, low, close, 14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            
            # Moving averages
            sma_20 = talib.SMA(close, 20)
            sma_50 = talib.SMA(close, 50)
            
            # Candlestick patterns
            doji = talib.CDLDOJI(open_price, high, low, close)
            hammer = talib.CDLHAMMER(open_price, high, low, close)
            engulfing = talib.CDLENGULFING(open_price, high, low, close)
            
            return {
                'rsi': rsi, 'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
                'atr': atr, 'bb_upper': bb_upper, 'bb_middle': bb_middle, 'bb_lower': bb_lower,
                'sma_20': sma_20, 'sma_50': sma_50,
                'doji': doji, 'hammer': hammer, 'engulfing': engulfing,
                'close': close, 'high': high, 'low': low
            }
        except Exception as e:
            print(f"❌ Indicator calculation error: {e}")
            return None
    
    def find_support_resistance(self, df):
        """Find key support and resistance levels"""
        current_price = df['close'].iloc[-1]
        
        # Simple S/R using recent highs/lows
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        resistance = recent_high if recent_high > current_price else None
        support = recent_low if recent_low < current_price else None
        
        return support, resistance
    
    def analyze_setup(self, df, indicators):
        """Analyze for swing trading setup"""
        if indicators is None or len(df) < 50:
            return None
        
        current_price = df['close'].iloc[-1]
        rsi = indicators['rsi'][-1] if not np.isnan(indicators['rsi'][-1]) else 50
        macd_hist = indicators['macd_hist'][-1] if not np.isnan(indicators['macd_hist'][-1]) else 0
        atr = indicators['atr'][-1] if not np.isnan(indicators['atr'][-1]) else current_price * 0.02
        
        support, resistance = self.find_support_resistance(df)
        
        setup = {
            'signal': 0,
            'confidence': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': 0,
            'reasons': []
        }
        
        # BULLISH SETUP
        bullish_score = 0
        
        if rsi < 30:
            bullish_score += 2
            setup['reasons'].append("RSI oversold")
        
        if support and abs(current_price - support) / current_price < 0.01:
            bullish_score += 2
            setup['reasons'].append("Near support")
        
        if indicators['hammer'][-1] > 0:
            bullish_score += 1
            setup['reasons'].append("Hammer pattern")
        
        if macd_hist > 0:
            bullish_score += 1
            setup['reasons'].append("MACD bullish")
        
        # BEARISH SETUP
        bearish_score = 0
        
        if rsi > 70:
            bearish_score += 2
            setup['reasons'].append("RSI overbought")
        
        if resistance and abs(current_price - resistance) / current_price < 0.01:
            bearish_score += 2
            setup['reasons'].append("Near resistance")
        
        if indicators['engulfing'][-1] < 0:
            bearish_score += 1
            setup['reasons'].append("Bearish engulfing")
        
        if macd_hist < 0:
            bearish_score += 1
            setup['reasons'].append("MACD bearish")
        
        # Set signal
        if bullish_score >= 3:
            setup['signal'] = 1
            setup['confidence'] = min(bullish_score / 5.0, 1.0)
            setup['stop_loss'] = current_price - (atr * 1.5)
            setup['take_profit'] = current_price + (atr * 4.5)
        elif bearish_score >= 3:
            setup['signal'] = -1
            setup['confidence'] = min(bearish_score / 5.0, 1.0)
            setup['stop_loss'] = current_price + (atr * 1.5)
            setup['take_profit'] = current_price - (atr * 4.5)
        
        # Calculate R:R
        if setup['signal'] != 0:
            risk = abs(current_price - setup['stop_loss'])
            reward = abs(setup['take_profit'] - current_price)
            setup['risk_reward'] = reward / risk if risk > 0 else 0
        
        return setup
    
    def should_trade(self, setup):
        """Check if setup meets criteria"""
        if not setup or setup['signal'] == 0:
            return False
        
        if self.last_trade_time:
            if time.time() - self.last_trade_time < self.min_trade_interval:
                return False
        
        if setup['risk_reward'] < self.min_rr_ratio:
            return False
        
        if setup['confidence'] < 0.6:
            return False
        
        return True
    
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
    
    def execute_trade(self, setup):
        """Execute swing trade"""
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
                "comment": "Swing Buy",
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
                "comment": "Swing Sell",
            }
        
        result = mt5.order_send(request)
        success = result and result.retcode == 10009
        
        if success:
            self.last_trade_time = time.time()
            direction = "BUY" if setup['signal'] == 1 else "SELL"
            return True, f"✅ {direction} | SL: ${setup['stop_loss']:.2f} | TP: ${setup['take_profit']:.2f} | RR: 1:{setup['risk_reward']:.1f}"
        else:
            return False, f"❌ Trade failed"
    
    def start_trading(self):
        if not mt5.initialize():
            print("❌ MT5 not available")
            return
        
        print("🎯 SWING TRADER STARTED")
        print(f"📊 Min R:R: 1:{self.min_rr_ratio}")
        
        step = 0
        while True:
            try:
                print(f"\n📊 Analysis #{step + 1} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Get data
                df = self.get_market_data()
                if df is None:
                    print("❌ No data")
                    time.sleep(60)
                    continue
                
                # Update trailing stops first
                self.update_trailing_stop()
                
                # Calculate indicators
                indicators = self.calculate_indicators(df)
                if indicators is None:
                    print("❌ Indicator error")
                    time.sleep(60)
                    continue
                
                # Analyze setup
                setup = self.analyze_setup(df, indicators)
                
                # Print analysis
                current_price = df['close'].iloc[-1]
                rsi = indicators['rsi'][-1] if not np.isnan(indicators['rsi'][-1]) else 50
                
                print(f"💰 ETH: ${current_price:.2f}")
                print(f"📈 RSI: {rsi:.1f}")
                
                if setup and setup['signal'] != 0:
                    print(f"🎯 SETUP: {'BUY' if setup['signal'] == 1 else 'SELL'}")
                    print(f"   Confidence: {setup['confidence']:.1%}")
                    print(f"   R:R: 1:{setup['risk_reward']:.1f}")
                    print(f"   Reasons: {', '.join(setup['reasons'])}")
                    
                    if self.should_trade(setup):
                        success, result = self.execute_trade(setup)
                        print(f"⚡ {result}")
                    else:
                        print(f"⏸️ Setup doesn't meet criteria")
                else:
                    print(f"⏸️ No setup found")
                
                step += 1
                print(f"⏳ Next check in 5 minutes...")
                time.sleep(300)
                
            except KeyboardInterrupt:
                print("\n✅ Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    trader = SwingTrader()
    trader.start_trading()
