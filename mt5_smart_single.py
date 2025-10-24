import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib
import time
import json
from datetime import datetime

class SmartSingleTrader:
    def __init__(self):
        self.symbol = "ETHUSDm"
        self.lot_size = 0.1
        self.min_rr_ratio = 4.0  # Higher R:R for quality
        self.lookback_period = 200
        
        # Single trade management
        self.current_trade = None
        self.trade_history = []
        self.learning_data = []
        
        # Strategy parameters
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.min_confidence = 0.8
        
    def get_market_data(self):
        """Get comprehensive market data"""
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, self.lookback_period)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        for col in ['open', 'high', 'low', 'close', 'tick_volume']:
            df[col] = df[col].astype(np.float64)
        
        return df
    
    def calculate_advanced_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values
        open_price = df['open'].values
        
        try:
            # Trend indicators
            rsi = talib.RSI(close, 14)
            macd, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
            adx = talib.ADX(high, low, close, 14)
            
            # Support/Resistance
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
            
            # Volatility
            atr = talib.ATR(high, low, close, 14)
            
            # Volume
            obv = talib.OBV(close, volume)
            
            # Moving averages
            ema_20 = talib.EMA(close, 20)
            ema_50 = talib.EMA(close, 50)
            sma_200 = talib.SMA(close, 200)
            
            # Candlestick patterns
            doji = talib.CDLDOJI(open_price, high, low, close)
            hammer = talib.CDLHAMMER(open_price, high, low, close)
            engulfing = talib.CDLENGULFING(open_price, high, low, close)
            shooting_star = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
            
            return {
                'rsi': rsi, 'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
                'adx': adx, 'atr': atr, 'obv': obv,
                'bb_upper': bb_upper, 'bb_middle': bb_middle, 'bb_lower': bb_lower,
                'ema_20': ema_20, 'ema_50': ema_50, 'sma_200': sma_200,
                'doji': doji, 'hammer': hammer, 'engulfing': engulfing, 'shooting_star': shooting_star,
                'close': close, 'high': high, 'low': low
            }
        except Exception as e:
            print(f"❌ Indicator error: {e}")
            return None
    
    def detect_divergence(self, price, indicator, periods=10):
        """Detect RSI/MACD divergence"""
        if len(price) < periods * 2:
            return 0
        
        # Recent price trend
        recent_high = np.max(price[-periods:])
        prev_high = np.max(price[-periods*2:-periods])
        price_trend = 1 if recent_high > prev_high else -1
        
        # Indicator trend
        recent_ind = np.max(indicator[-periods:])
        prev_ind = np.max(indicator[-periods*2:-periods])
        ind_trend = 1 if recent_ind > prev_ind else -1
        
        # Divergence
        if price_trend != ind_trend:
            return -price_trend  # Bearish = -1, Bullish = 1
        return 0
    
    def find_key_levels(self, df):
        """Find strong support/resistance levels"""
        highs = df['high'].values
        lows = df['low'].values
        current_price = df['close'].iloc[-1]
        
        # Find pivot points
        resistance_levels = []
        support_levels = []
        
        for i in range(10, len(highs) - 10):
            # Resistance: local high
            if highs[i] == np.max(highs[i-10:i+10]) and highs[i] > current_price:
                resistance_levels.append(highs[i])
            
            # Support: local low
            if lows[i] == np.min(lows[i-10:i+10]) and lows[i] < current_price:
                support_levels.append(lows[i])
        
        # Get nearest levels
        nearest_resistance = min(resistance_levels) if resistance_levels else None
        nearest_support = max(support_levels) if support_levels else None
        
        return nearest_support, nearest_resistance
    
    def analyze_master_setup(self, df, indicators):
        """Master setup analysis with multiple confirmations"""
        if indicators is None or len(df) < 100:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Get indicator values
        rsi = indicators['rsi'][-1] if not np.isnan(indicators['rsi'][-1]) else 50
        macd_hist = indicators['macd_hist'][-1] if not np.isnan(indicators['macd_hist'][-1]) else 0
        adx = indicators['adx'][-1] if not np.isnan(indicators['adx'][-1]) else 20
        atr = indicators['atr'][-1] if not np.isnan(indicators['atr'][-1]) else current_price * 0.02
        
        # Trend context
        ema_20 = indicators['ema_20'][-1]
        ema_50 = indicators['ema_50'][-1]
        sma_200 = indicators['sma_200'][-1]
        
        # Support/Resistance
        support, resistance = self.find_key_levels(df)
        
        # Divergences
        rsi_div = self.detect_divergence(indicators['close'], indicators['rsi'])
        macd_div = self.detect_divergence(indicators['close'], indicators['macd'])
        
        setup = {
            'signal': 0,
            'confidence': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': 0,
            'reasons': [],
            'strength': 0
        }
        
        # BULLISH MASTER SETUP
        bullish_strength = 0
        
        # 1. RSI oversold + divergence (strongest signal)
        if rsi < self.rsi_oversold and rsi_div > 0:
            bullish_strength += 4
            setup['reasons'].append("RSI oversold + bullish divergence")
        elif rsi < self.rsi_oversold:
            bullish_strength += 2
            setup['reasons'].append("RSI oversold")
        
        # 2. Price at strong support
        if support and abs(current_price - support) / current_price < 0.005:
            bullish_strength += 3
            setup['reasons'].append("Strong support level")
        
        # 3. MACD turning bullish + divergence
        if macd_hist > 0 and macd_div > 0:
            bullish_strength += 3
            setup['reasons'].append("MACD bullish + divergence")
        elif macd_hist > 0:
            bullish_strength += 1
            setup['reasons'].append("MACD bullish")
        
        # 4. Strong trend (ADX > 25)
        if adx > 25 and current_price > ema_20 > ema_50:
            bullish_strength += 2
            setup['reasons'].append("Strong uptrend")
        
        # 5. Bullish candlestick pattern
        if indicators['hammer'][-1] > 0:
            bullish_strength += 2
            setup['reasons'].append("Hammer pattern")
        
        # BEARISH MASTER SETUP
        bearish_strength = 0
        
        # 1. RSI overbought + divergence
        if rsi > self.rsi_overbought and rsi_div < 0:
            bearish_strength += 4
            setup['reasons'].append("RSI overbought + bearish divergence")
        elif rsi > self.rsi_overbought:
            bearish_strength += 2
            setup['reasons'].append("RSI overbought")
        
        # 2. Price at strong resistance
        if resistance and abs(current_price - resistance) / current_price < 0.005:
            bearish_strength += 3
            setup['reasons'].append("Strong resistance level")
        
        # 3. MACD turning bearish + divergence
        if macd_hist < 0 and macd_div < 0:
            bearish_strength += 3
            setup['reasons'].append("MACD bearish + divergence")
        elif macd_hist < 0:
            bearish_strength += 1
            setup['reasons'].append("MACD bearish")
        
        # 4. Strong downtrend
        if adx > 25 and current_price < ema_20 < ema_50:
            bearish_strength += 2
            setup['reasons'].append("Strong downtrend")
        
        # 5. Bearish patterns
        if indicators['shooting_star'][-1] > 0 or indicators['engulfing'][-1] < 0:
            bearish_strength += 2
            setup['reasons'].append("Bearish pattern")
        
        # Determine final signal
        if bullish_strength >= 6:  # High threshold
            setup['signal'] = 1
            setup['strength'] = bullish_strength
            setup['confidence'] = min(bullish_strength / 10.0, 1.0)
            setup['stop_loss'] = support if support else current_price - (atr * 2)
            setup['take_profit'] = resistance if resistance else current_price + (atr * 8)
            
        elif bearish_strength >= 6:
            setup['signal'] = -1
            setup['strength'] = bearish_strength
            setup['confidence'] = min(bearish_strength / 10.0, 1.0)
            setup['stop_loss'] = resistance if resistance else current_price + (atr * 2)
            setup['take_profit'] = support if support else current_price - (atr * 8)
        
        # Calculate R:R
        if setup['signal'] != 0:
            risk = abs(current_price - setup['stop_loss'])
            reward = abs(setup['take_profit'] - current_price)
            setup['risk_reward'] = reward / risk if risk > 0 else 0
        
        return setup
    
    def has_open_position(self):
        """Check if we have open position"""
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions) > 0 if positions else False
    
    def execute_single_trade(self, setup):
        """Execute single high-quality trade"""
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return False, "No price"
        
        trade_data = {
            'timestamp': datetime.now(),
            'signal': setup['signal'],
            'confidence': setup['confidence'],
            'strength': setup['strength'],
            'reasons': setup['reasons'],
            'entry_price': tick.ask if setup['signal'] == 1 else tick.bid,
            'stop_loss': setup['stop_loss'],
            'take_profit': setup['take_profit'],
            'risk_reward': setup['risk_reward']
        }
        
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
                "comment": "Master Buy",
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
                "comment": "Master Sell",
            }
        
        result = mt5.order_send(request)
        success = result and result.retcode == 10009
        
        if success:
            self.current_trade = trade_data
            direction = "BUY" if setup['signal'] == 1 else "SELL"
            return True, f"🎯 MASTER {direction} | Strength: {setup['strength']} | RR: 1:{setup['risk_reward']:.1f}"
        else:
            return False, "❌ Trade failed"
    
    def monitor_trade(self):
        """Monitor current trade and learn from outcome"""
        if not self.current_trade:
            return
        
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            # Trade closed - analyze outcome
            self.learn_from_trade()
            self.current_trade = None
            return
        
        # Trade still open
        pos = positions[0]
        print(f"📊 Monitoring: PnL ${pos.profit:.2f} | Time: {(datetime.now() - self.current_trade['timestamp']).seconds//60}m")
    
    def learn_from_trade(self):
        """Learn from completed trade"""
        if not self.current_trade:
            return
        
        # Get final outcome from history
        deals = mt5.history_deals_get(datetime.now().replace(hour=0), datetime.now())
        if deals:
            last_deal = deals[-1]
            profit = last_deal.profit
            
            outcome = {
                'trade': self.current_trade,
                'profit': profit,
                'success': profit > 0,
                'timestamp': datetime.now()
            }
            
            self.trade_history.append(outcome)
            
            # Save learning data
            with open('trade_learning.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            
            result = "✅ WIN" if profit > 0 else "❌ LOSS"
            print(f"🧠 TRADE COMPLETE: {result} | Profit: ${profit:.2f}")
            print(f"📚 Learning saved. Total trades: {len(self.trade_history)}")
    
    def start_master_trading(self):
        if not mt5.initialize():
            print("❌ MT5 not available")
            return
        
        print("🎯 MASTER SINGLE TRADE SYSTEM")
        print(f"📊 Min R:R: 1:{self.min_rr_ratio}")
        print(f"🎪 Min Confidence: {self.min_confidence:.0%}")
        print(f"🔥 Only trades MASTER setups")
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                print(f"\n🔍 Master Scan #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Check if we have open position
                if self.has_open_position():
                    self.monitor_trade()
                    time.sleep(60)
                    continue
                
                # Get market data
                df = self.get_market_data()
                if df is None:
                    print("❌ No data")
                    time.sleep(60)
                    continue
                
                # Calculate indicators
                indicators = self.calculate_advanced_indicators(df)
                if indicators is None:
                    time.sleep(60)
                    continue
                
                # Analyze master setup
                setup = self.analyze_master_setup(df, indicators)
                
                current_price = df['close'].iloc[-1]
                rsi = indicators['rsi'][-1] if not np.isnan(indicators['rsi'][-1]) else 50
                
                print(f"💰 ETH: ${current_price:.2f} | RSI: {rsi:.1f}")
                
                if setup and setup['signal'] != 0:
                    print(f"🎯 MASTER SETUP DETECTED!")
                    print(f"   Direction: {'BUY' if setup['signal'] == 1 else 'SELL'}")
                    print(f"   Strength: {setup['strength']}/10")
                    print(f"   Confidence: {setup['confidence']:.1%}")
                    print(f"   R:R: 1:{setup['risk_reward']:.1f}")
                    print(f"   Reasons: {', '.join(setup['reasons'])}")
                    
                    # Check quality criteria
                    if (setup['confidence'] >= self.min_confidence and 
                        setup['risk_reward'] >= self.min_rr_ratio):
                        
                        success, result = self.execute_single_trade(setup)
                        print(f"⚡ {result}")
                        
                        if success:
                            print(f"🎯 TRADE EXECUTED - Now monitoring...")
                    else:
                        print(f"⏸️ Setup doesn't meet master criteria")
                else:
                    print(f"⏸️ No master setup found")
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n✅ Master trading stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    trader = SmartSingleTrader()
    trader.start_master_trading()
