import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time
from datetime import datetime

class GreedyMoneyBot:
    def __init__(self):
        self.symbol = "ETHUSDm"
        self.base_lot = 0.1  # Match broker minimum
        self.max_lot = 1.0  # Aggressive position sizing
        self.strategies = {}
        self.performance = {}
        self.total_profit = 0
        
        # GREEDY PARAMETERS
        self.min_profit_target = 0.001  # 0.1% minimum
        self.max_risk_per_trade = 0.02  # 2% risk
        self.compound_profits = True
        self.use_martingale = True
        self.last_price = 0  # Track price for momentum forcing
        
        self.init_strategies()
        self.load_models()  # Load saved models
    
    def save_models(self):
        """Save all models and performance data"""
        os.makedirs('greedy_models', exist_ok=True)
        
        # Save models
        for name, strategy in self.strategies.items():
            with open(f'greedy_models/{name}_model.pkl', 'wb') as f:
                pickle.dump(strategy['model'], f)
        
        # Save performance
        with open('greedy_models/performance.pkl', 'wb') as f:
            pickle.dump(self.performance, f)
        
        # Save total profit
        with open('greedy_models/total_profit.txt', 'w') as f:
            f.write(str(self.total_profit))
        
        print(f"💾 Models saved! Total profit: ${self.total_profit:.2f}")
    
    def load_models(self):
        """Load saved models and performance"""
        if not os.path.exists('greedy_models'):
            return
        
        try:
            # Load models
            for name in self.strategies:
                model_path = f'greedy_models/{name}_model.pkl'
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.strategies[name]['model'] = pickle.load(f)
                    print(f"📂 Loaded {name} model")
            
            # Load performance
            if os.path.exists('greedy_models/performance.pkl'):
                with open('greedy_models/performance.pkl', 'rb') as f:
                    self.performance = pickle.load(f)
                print("📊 Performance data loaded")
            
            # Load total profit
            if os.path.exists('greedy_models/total_profit.txt'):
                with open('greedy_models/total_profit.txt', 'r') as f:
                    self.total_profit = float(f.read().strip())
                print(f"💰 Previous profit: ${self.total_profit:.2f}")
                
        except Exception as e:
            print(f"⚠️ Load error: {e}")
    
    def init_strategies(self):
        """Initialize multiple greedy strategies"""
        self.strategies = {
            'scalper': {'model': RandomForestClassifier(n_estimators=50, max_depth=5), 'timeframe': 1, 'profit': 0},
            'momentum': {'model': GradientBoostingClassifier(n_estimators=100), 'timeframe': 5, 'profit': 0},
            'reversal': {'model': LogisticRegression(), 'timeframe': 15, 'profit': 0},
            'breakout': {'model': RandomForestClassifier(n_estimators=200, max_depth=15), 'timeframe': 5, 'profit': 0}
        }
        
        for name in self.strategies:
            self.performance[name] = {'wins': 0, 'losses': 0, 'profit': 0}
    
    def get_aggressive_features(self, df, strategy_type):
        """Extract features optimized for each strategy"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values
        
        # Base indicators
        rsi = talib.RSI(close, 14)
        macd, macd_signal, macd_hist = talib.MACD(close)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20)
        atr = talib.ATR(high, low, close, 14)
        
        # Strategy-specific features
        if strategy_type == 'scalper':
            # Ultra-short term features
            features = np.column_stack([
                rsi, macd_hist,
                (close - np.roll(close, 1)) / close,  # 1-bar return
                (close - np.roll(close, 2)) / close,  # 2-bar return
                volume / np.roll(volume, 1),          # Volume ratio
                (high - low) / close,                 # Range
            ])
        
        elif strategy_type == 'momentum':
            # Trend following features
            sma_10 = talib.SMA(close, 10)
            sma_50 = talib.SMA(close, 50)
            features = np.column_stack([
                rsi, macd, macd_signal,
                (close - sma_10) / sma_10,
                (close - sma_50) / sma_50,
                (sma_10 - sma_50) / sma_50,
                np.roll(close, 5) / close - 1,        # 5-bar momentum
            ])
        
        elif strategy_type == 'reversal':
            # Mean reversion features
            features = np.column_stack([
                rsi,
                (close - bb_middle) / bb_middle,
                (close - bb_upper) / bb_upper,
                (close - bb_lower) / bb_lower,
                talib.STOCH(high, low, close)[0],     # Stochastic
                talib.CCI(high, low, close, 14),      # CCI
            ])
        
        else:  # breakout
            # Volatility breakout features
            features = np.column_stack([
                atr / close,
                (high - np.roll(high, 20).max()) / close,  # New highs
                (low - np.roll(low, 20).min()) / close,    # New lows
                volume / np.roll(volume, 20).mean(),       # Volume surge
                rsi, macd_hist,
            ])
        
        return features[50:]  # Remove NaN rows
    
    def create_greedy_labels(self, df, strategy_type):
        """Create labels optimized for maximum profit"""
        close = df['close'].values
        
        if strategy_type == 'scalper':
            # Even more aggressive - tiny profits
            future_1 = np.roll(close, -1) / close - 1
            labels = np.where(future_1 > 0.0002, 1,      # 0.02% quick profit
                     np.where(future_1 < -0.0002, -1, 
                     np.where(future_1 > 0, 1, -1)))     # Any positive = buy, negative = sell
        
        elif strategy_type == 'momentum':
            # More aggressive momentum
            future_5 = np.roll(close, -5) / close - 1
            labels = np.where(future_5 > 0.001, 1,       # 0.1% trend profit
                     np.where(future_5 < -0.001, -1, 
                     np.where(future_5 > 0, 1, -1)))
        
        elif strategy_type == 'reversal':
            # Aggressive mean reversion
            future_3 = np.roll(close, -3) / close - 1
            labels = np.where(future_3 > 0.0008, 1,      # 0.08% reversion
                     np.where(future_3 < -0.0008, -1, 
                     np.where(future_3 > 0, 1, -1)))
        
        else:  # breakout
            # Aggressive breakout
            future_10 = np.roll(close, -10) / close - 1
            labels = np.where(future_10 > 0.002, 1,      # 0.2% breakout
                     np.where(future_10 < -0.002, -1, 
                     np.where(future_10 > 0, 1, -1)))
        
        return labels[50:-20]
    
    def train_strategy(self, strategy_name):
        """Train individual strategy for maximum profit"""
        strategy = self.strategies[strategy_name]
        timeframe = mt5.TIMEFRAME_M1 if strategy['timeframe'] == 1 else mt5.TIMEFRAME_M5
        
        # Get data
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 500)
        if rates is None:
            return False
        
        df = pd.DataFrame(rates)
        for col in ['open', 'high', 'low', 'close', 'tick_volume']:
            df[col] = df[col].astype(np.float64)
        
        # Extract features and labels
        X = self.get_aggressive_features(df, strategy_name)
        y = self.create_greedy_labels(df, strategy_name)
        
        # Ensure same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        # Clean data
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 50:
            return False
        
        # Train model
        strategy['model'].fit(X, y)
        accuracy = strategy['model'].score(X, y)
        
        print(f"🧠 {strategy_name.upper()}: {accuracy:.1%} accuracy")
        return True
    
    def get_best_signal(self):
        """Get signal from best performing strategy"""
        signals = {}
        confidences = {}
        
        # Get current data for each strategy
        for name, strategy in self.strategies.items():
            timeframe = mt5.TIMEFRAME_M1 if strategy['timeframe'] == 1 else mt5.TIMEFRAME_M5
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 100)
            
            if rates is None:
                continue
            
            df = pd.DataFrame(rates)
            for col in ['open', 'high', 'low', 'close', 'tick_volume']:
                df[col] = df[col].astype(np.float64)
            
            # Get features
            X = self.get_aggressive_features(df, name)
            if len(X) == 0 or np.isnan(X[-1]).any():
                continue
            
            # Predict
            try:
                signal = strategy['model'].predict([X[-1]])[0]
                if hasattr(strategy['model'], 'predict_proba'):
                    proba = strategy['model'].predict_proba([X[-1]])[0]
                    confidence = np.max(proba)
                else:
                    confidence = 0.7  # Default for models without proba
                
                signals[name] = signal
                confidences[name] = confidence
                
            except:
                continue
        
        if not signals:
            return 0, 0, 'none'
        
        # Weight by recent performance
        best_strategy = max(self.performance.keys(), 
                          key=lambda x: self.performance[x]['profit'])
        
        if best_strategy in signals:
            return signals[best_strategy], confidences[best_strategy], best_strategy
        
        # Fallback to highest confidence
        best_conf_strategy = max(confidences.keys(), key=lambda x: confidences[x])
        return signals[best_conf_strategy], confidences[best_conf_strategy], best_conf_strategy
    
    def calculate_greedy_position_size(self, confidence, strategy_name):
        """Calculate position size based on greed and performance"""
        # Get symbol info for proper volume calculation
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            return 0.01  # Default minimum
        
        min_volume = symbol_info.volume_min
        max_volume = symbol_info.volume_max
        volume_step = symbol_info.volume_step
        
        base_size = max(min_volume, 0.1)  # Use broker minimum 0.1
        
        # Increase size based on confidence
        confidence_multiplier = 1 + (confidence - 0.5) * 2  # 1x to 2x
        
        # Increase size based on strategy performance
        strategy_profit = self.performance[strategy_name]['profit']
        if strategy_profit > 0:
            performance_multiplier = 1 + min(strategy_profit / 100, 1)  # Up to 2x
        else:
            performance_multiplier = 0.5  # Reduce if losing
        
        # Martingale on losses (GREEDY!)
        if self.use_martingale and strategy_profit < 0:
            martingale_multiplier = min(2.0, 1 + abs(strategy_profit) / 50)
        else:
            martingale_multiplier = 1.0
        
        final_size = base_size * confidence_multiplier * performance_multiplier * martingale_multiplier
        
        # Round to valid volume step
        final_size = round(final_size / volume_step) * volume_step
        final_size = max(min_volume, min(final_size, min(max_volume, self.max_lot)))
        
        return final_size
    
    def execute_greedy_trade(self, signal, confidence, strategy_name):
        """Execute trade with greedy parameters"""
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            print("❌ No tick data")
            return False
        
        # Check if market is open
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            print("❌ Symbol not found")
            return False
        
        if not symbol_info.visible:
            print("❌ Symbol not visible in Market Watch")
            if not mt5.symbol_select(self.symbol, True):
                print("❌ Failed to add symbol to Market Watch")
                return False
        
        # Calculate position size
        lot_size = self.calculate_greedy_position_size(confidence, strategy_name)
        
        current_price = tick.ask if signal == 1 else tick.bid
        atr_estimate = current_price * 0.008  # Tight stops for greed
        
        # Aggressive TP/SL ratios
        if strategy_name == 'scalper':
            tp_multiplier, sl_multiplier = 2, 1  # 2:1 RR
        elif strategy_name == 'momentum':
            tp_multiplier, sl_multiplier = 4, 1  # 4:1 RR
        else:
            tp_multiplier, sl_multiplier = 3, 1  # 3:1 RR
        
        if signal == 1:  # Buy
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "sl": current_price - (atr_estimate * sl_multiplier),
                "tp": current_price + (atr_estimate * tp_multiplier),
                "deviation": 20,
                "magic": 777000,
                "comment": f"GREEDY_{strategy_name}_{confidence:.0%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        else:  # Sell
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "sl": current_price + (atr_estimate * sl_multiplier),
                "tp": current_price - (atr_estimate * tp_multiplier),
                "deviation": 20,
                "magic": 777000,
                "comment": f"GREEDY_{strategy_name}_{confidence:.0%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        
        result = mt5.order_send(request)
        
        if result is None:
            print("❌ order_send() returned None")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Trade failed: {result.retcode} - {result.comment}")
            
            # Common error explanations
            if result.retcode == 10004:
                print("   → Requote error - price moved")
            elif result.retcode == 10006:
                print("   → Request rejected - invalid parameters")
            elif result.retcode == 10007:
                print("   → Request canceled by trader")
            elif result.retcode == 10008:
                print("   → Order placed but not confirmed")
            elif result.retcode == 10013:
                print("   → Invalid request - check symbol/volume")
            elif result.retcode == 10014:
                print("   → Invalid volume")
            elif result.retcode == 10015:
                print("   → Invalid price")
            elif result.retcode == 10016:
                print("   → Invalid stops")
            elif result.retcode == 10018:
                print("   → Market closed")
            elif result.retcode == 10019:
                print("   → No money")
            elif result.retcode == 10020:
                print("   → Prices changed")
            elif result.retcode == 10021:
                print("   → No quotes")
            else:
                print(f"   → Error code: {result.retcode}")
            
            return False
        
        direction = "BUY" if signal == 1 else "SELL"
        print(f"💰 {direction} {lot_size} lots | {strategy_name} | RR: 1:{tp_multiplier}")
        return True
    
    def start_greedy_bot(self):
        """Start the greedy money-making bot"""
        if not mt5.initialize():
            print("❌ MT5 failed")
            return
        
        print("💰 GREEDY MONEY BOT")
        print("🎯 MAXIMUM PROFIT MODE")
        print(f"💸 Max lot size: {self.max_lot}")
        print("⚠️  HIGH RISK = HIGH REWARD!")
        
        # Check symbol info
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            print(f"📊 {self.symbol} Info:")
            print(f"   Min volume: {symbol_info.volume_min}")
            print(f"   Max volume: {symbol_info.volume_max}")
            print(f"   Volume step: {symbol_info.volume_step}")
            print(f"   Spread: {symbol_info.spread}")
        else:
            print(f"❌ Symbol {self.symbol} not found!")
            return
        
        # Initial training
        print("\n🧠 Training all strategies...")
        for name in self.strategies:
            if self.train_strategy(name):
                print(f"✅ {name} trained")
            else:
                print(f"❌ {name} failed")
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                print(f"\n💰 MONEY SCAN #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Check positions
                positions = mt5.positions_get(symbol=self.symbol)
                if positions:
                    total_pnl = sum(pos.profit for pos in positions)
                    print(f"📊 {len(positions)} positions | Total PnL: ${total_pnl:.2f}")
                    
                    # Close profitable positions quickly (GREEDY!)
                    for pos in positions:
                        if pos.profit > 5:  # Close at $5 profit
                            close_request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": self.symbol,
                                "volume": pos.volume,
                                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                "position": pos.ticket,
                                "magic": 777000,
                            }
                            result = mt5.order_send(close_request)
                            if result and result.retcode == 10009:
                                self.total_profit += pos.profit
                                print(f"💰 Closed for ${pos.profit:.2f} profit!")
                    
                    time.sleep(10)
                    continue
                
                # Retrain frequently for overfitting
                if scan_count % 20 == 1:
                    best_strategy = max(self.performance.keys(), 
                                      key=lambda x: self.performance[x]['profit'])
                    self.train_strategy(best_strategy)
                    print(f"🔄 Retrained {best_strategy}")
                
                # Get best signal
                signal, confidence, strategy_name = self.get_best_signal()
                
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    current_price = tick.bid
                    print(f"💎 ETH: ${current_price:.2f}")
                    print(f"🎯 Best: {strategy_name} | Signal: {signal} | Conf: {confidence:.1%}")
                    
                    # Execute if confident (ULTRA GREEDY threshold)
                    if abs(signal) == 1 and confidence > 0.50:  # Even lower threshold
                        success = self.execute_greedy_trade(signal, confidence, strategy_name)
                        if success:
                            print("🚀 GREEDY TRADE EXECUTED!")
                        else:
                            print("❌ Trade failed")
                    elif confidence > 0.70:  # Force trade on high confidence even if signal=0
                        # Force a trade based on price momentum
                        tick = mt5.symbol_info_tick(self.symbol)
                        if tick:
                            prev_price = getattr(self, 'last_price', tick.bid)
                            price_change = (tick.bid - prev_price) / prev_price
                            
                            if price_change > 0.0001:  # Tiny upward momentum
                                forced_signal = 1
                                print(f"🔥 FORCING BUY on momentum! Change: {price_change:.4%}")
                                success = self.execute_greedy_trade(forced_signal, confidence, strategy_name)
                                if success:
                                    print("🚀 FORCED TRADE EXECUTED!")
                            elif price_change < -0.0001:  # Tiny downward momentum
                                forced_signal = -1
                                print(f"🔥 FORCING SELL on momentum! Change: {price_change:.4%}")
                                success = self.execute_greedy_trade(forced_signal, confidence, strategy_name)
                                if success:
                                    print("🚀 FORCED TRADE EXECUTED!")
                            
                            self.last_price = tick.bid
                    else:
                        print("⏸️ Waiting for better setup...")
                
                time.sleep(5)  # Very fast scanning for maximum opportunities
                
            except KeyboardInterrupt:
                print(f"\n💰 FINAL PROFIT: ${self.total_profit:.2f}")
                self.save_models()
                print("✅ Bot stopped and saved!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.save_models()  # Save on error too
                time.sleep(30)

if __name__ == "__main__":
    bot = GreedyMoneyBot()
    bot.start_greedy_bot()
