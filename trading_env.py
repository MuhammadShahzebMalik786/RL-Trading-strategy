import gymnasium as gym
import numpy as np
import pandas as pd
import requests
import time
from typing import Dict, List, Tuple, Optional
import ta

class TradingEnv(gym.Env):
    # Class variable to cache data across instances
    _cached_data = None
    _cache_timestamp = None
    
    def __init__(self, 
                 initial_balance: float = 10.0,
                 leverage: int = 2000,
                 spread_per_lot: float = 1.6,
                 min_lot: float = 0.1,
                 max_lot: float = 1.0,
                 margin_per_lot: float = 1.0,
                 max_steps: int = 500,
                 lookback: int = 20,
                 use_cache: bool = True):
        
        super(TradingEnv, self).__init__()
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.spread_per_lot = spread_per_lot
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.margin_per_lot = margin_per_lot
        self.max_steps = max_steps
        self.lookback = lookback
        self.use_cache = use_cache
        
        # Risk management
        self.max_drawdown_pct = 0.20
        self.max_concurrent_trades = 3
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        self.trade_cooldown = 3
        
        # Action space: [hold, buy, sell, close_all]
        self.action_space = gym.spaces.Discrete(4)
        
        # State space: OHLCV + indicators + account info + position details + margin info
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback * 9 + 14,), dtype=np.float32  # Added 2 more for margin info
        )
        
        # Initialize data
        self.data = None
        self.reset()
    
    def get_binance_data(self, symbol: str = "ETHUSDT", interval: str = "1m", days: int = 30) -> pd.DataFrame:
        """Fetch 30 days of data from Binance API with caching"""
        try:
            import time
            
            # Check if we can use cached data (within 1 hour)
            if (self.use_cache and 
                TradingEnv._cached_data is not None and 
                TradingEnv._cache_timestamp is not None and
                time.time() - TradingEnv._cache_timestamp < 3600):
                return TradingEnv._cached_data.copy()
            
            all_data = []
            
            # Calculate how many requests needed (1440 minutes per day, 1000 limit per request)
            total_minutes = days * 1440
            requests_needed = (total_minutes + 999) // 1000  # Round up
            
            print(f"📡 Fetching {days} days of {symbol} data ({total_minutes} candles)...")
            
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            
            for i in range(requests_needed):
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': 1000,
                    'endTime': end_time
                }
                
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    print(f"Error fetching data: {response.status_code}")
                    break
                    
                data = response.json()
                if not data:
                    break
                    
                all_data.extend(data)
                
                # Set end_time to the start of the oldest candle for next request
                end_time = data[0][0] - 1
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
                if i % 10 == 0:  # Show progress every 10 batches
                    print(f"📊 Fetched batch {i+1}/{requests_needed}")
            
            if not all_data:
                print("No data received, using synthetic data")
                return self.generate_synthetic_data()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            
            # Cache the data
            if self.use_cache:
                TradingEnv._cached_data = df.copy()
                TradingEnv._cache_timestamp = time.time()
                print("💾 Data cached for future use")
            
            return df
            
        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self, n_points: int = 1000) -> pd.DataFrame:
        """Generate synthetic OHLCV data"""
        np.random.seed(42)
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0001, 0.01, n_points)
        prices = 2000 * np.exp(np.cumsum(returns))  # ETH-like prices
        
        # Generate OHLCV
        data = []
        for i in range(n_points):
            price = prices[i]
            volatility = np.random.uniform(0.005, 0.02)
            
            high = price * (1 + volatility * np.random.uniform(0, 1))
            low = price * (1 - volatility * np.random.uniform(0, 1))
            open_price = np.random.uniform(low, high)
            close = np.random.uniform(low, high)
            volume = np.random.uniform(100, 1000)
            
            data.append([i, open_price, high, low, close, volume])
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # EMA
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        # Fill NaN values
        df = df.bfill().fillna(0)
        
        return df
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Fetch new data
        self.data = self.get_binance_data()
        self.data = self.calculate_indicators(self.data)
        
        # Reset state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin_used = 0.0
        self.positions = []  # List of open positions
        self.step_count = 0
        self.current_idx = self.lookback
        
        # Performance tracking
        self.trade_history = []
        self.daily_pnl = []
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.last_trade_step = -self.trade_cooldown
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        if self.current_idx < self.lookback:
            self.current_idx = self.lookback
        
        # Get historical price data
        start_idx = max(0, self.current_idx - self.lookback)
        end_idx = self.current_idx
        
        hist_data = self.data.iloc[start_idx:end_idx]
        
        # Normalize price data
        close_prices = hist_data['close'].values
        if len(close_prices) > 0:
            price_norm = close_prices / close_prices[-1]
        else:
            price_norm = np.ones(self.lookback)
        
        # Pad if necessary
        if len(price_norm) < self.lookback:
            price_norm = np.pad(price_norm, (self.lookback - len(price_norm), 0), 'edge')
        
        # Technical indicators (normalized)
        features = []
        for col in ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ema_20', 'bb_upper']:
            if col in hist_data.columns:
                values = hist_data[col].values
                if len(values) < self.lookback:
                    values = np.pad(values, (self.lookback - len(values), 0), 'edge')
                
                # Normalize
                if col == 'volume':
                    values = values / (np.max(values) + 1e-8)
                elif col == 'rsi':
                    values = values / 100.0
                else:
                    values = values / (close_prices[-1] + 1e-8)
                
                features.extend(values[-self.lookback:])
            else:
                features.extend([0.0] * self.lookback)
        
        # Account information
        balance_norm = self.balance / self.initial_balance
        equity_norm = self.equity / self.initial_balance
        margin_norm = self.margin_used / self.initial_balance
        position_count = len(self.positions) / self.max_concurrent_trades
        
        # Recent performance
        recent_trades = self.trade_history[-10:] if self.trade_history else []
        win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / max(len(recent_trades), 1)
        
        # Volatility
        if len(close_prices) > 1:
            volatility = np.std(np.diff(close_prices)) / close_prices[-1]
        else:
            volatility = 0.0
        
        # Drawdown
        drawdown_norm = self.max_drawdown / self.max_drawdown_pct
        
        # Time since last trade
        time_since_trade = min((self.step_count - self.last_trade_step) / self.trade_cooldown, 1.0)
        
        # Position details for model awareness
        avg_entry_price = 0.0
        avg_sl_distance = 0.0
        avg_tp_distance = 0.0
        unrealized_pnl_norm = 0.0
        
        if self.positions:
            current_price = self.data.iloc[self.current_idx]['close']
            total_pnl = 0.0
            total_sl_dist = 0.0
            total_tp_dist = 0.0
            total_entry = 0.0
            
            for pos in self.positions:
                total_entry += pos['entry_price']
                total_sl_dist += abs(pos['stop_loss'] - pos['entry_price']) / pos['entry_price']
                total_tp_dist += abs(pos['take_profit'] - pos['entry_price']) / pos['entry_price']
                
                # Calculate unrealized PnL
                if pos['direction'] == 'buy':
                    pnl = (current_price - pos['entry_price']) * pos['lot_size'] * self.leverage
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['lot_size'] * self.leverage
                total_pnl += pnl
            
            avg_entry_price = total_entry / len(self.positions) / current_price  # Normalized
            avg_sl_distance = total_sl_dist / len(self.positions)
            avg_tp_distance = total_tp_dist / len(self.positions)
            unrealized_pnl_norm = total_pnl / self.balance
        
        # Margin safety metrics
        free_margin = self.balance - self.margin_used
        free_margin_norm = free_margin / self.initial_balance
        
        margin_level = 100.0  # Default safe level
        if self.margin_used > 0:
            margin_level = min((self.equity / self.margin_used) * 100, 500.0) / 500.0  # Normalized to 0-1
        else:
            margin_level = 1.0  # No positions = safe
        
        account_features = [
            balance_norm, equity_norm, margin_norm, position_count,
            win_rate, volatility, drawdown_norm, time_since_trade,
            avg_entry_price, avg_sl_distance, avg_tp_distance, unrealized_pnl_norm,
            free_margin_norm, margin_level
        ]
        
        observation = np.array(features + account_features, dtype=np.float32)
        
        # Ensure correct shape
        expected_size = self.lookback * 9 + 14
        if len(observation) != expected_size:
            observation = np.resize(observation, expected_size)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        self.step_count += 1
        self.current_idx = min(self.current_idx + 1, len(self.data) - 1)
        
        # Get current price
        current_price = self.data.iloc[self.current_idx]['close']
        
        # Update positions
        self._update_positions(current_price)
        
        # Execute action
        reward = 0.0
        info = {'action': action, 'price': current_price}
        self._last_action = action  # Track for reward calculation
        
        if action == 1:  # Buy
            reward += self._execute_trade('buy', current_price)
        elif action == 2:  # Sell
            reward += self._execute_trade('sell', current_price)
        elif action == 3:  # Close all
            reward += self._close_all_positions(current_price)
        
        # Calculate reward
        reward += self._calculate_reward()
        
        # Check termination
        terminated = self._check_termination()
        truncated = False  # We don't use truncation in this environment
        
        # Update equity and drawdown
        self._update_equity()
        
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_trade(self, direction: str, price: float) -> float:
        """Execute a trade"""
        # Check cooldown
        if self.step_count - self.last_trade_step < self.trade_cooldown:
            return -1.0  # Penalty for trading too frequently
        
        # Check max positions
        if len(self.positions) >= self.max_concurrent_trades:
            return -1.0
        
        # Check margin
        required_margin = self.min_lot * self.margin_per_lot
        if self.margin_used + required_margin > self.balance * 0.8:  # 80% margin usage limit
            return -1.0
        
        # Calculate spread cost
        spread_cost = self.min_lot * self.spread_per_lot
        
        # Create position
        position = {
            'direction': direction,
            'entry_price': price,
            'lot_size': self.min_lot,
            'margin': required_margin,
            'spread_cost': spread_cost,
            'entry_step': self.step_count,
            'stop_loss': price * (1 - self.stop_loss_pct) if direction == 'buy' else price * (1 + self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct) if direction == 'buy' else price * (1 - self.take_profit_pct)
        }
        
        self.positions.append(position)
        self.margin_used += required_margin
        self.balance -= spread_cost  # Pay spread immediately
        self.last_trade_step = self.step_count
        
        return -2.0  # Small penalty for opening position (spread cost)
    
    def _update_positions(self, current_price: float):
        """Update open positions and close if SL/TP hit"""
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            # Check stop loss and take profit
            if pos['direction'] == 'buy':
                if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                    positions_to_close.append(i)
            else:  # sell
                if current_price >= pos['stop_loss'] or current_price <= pos['take_profit']:
                    positions_to_close.append(i)
        
        # Close positions
        for i in reversed(positions_to_close):
            self._close_position(i, current_price)
    
    def _close_position(self, pos_idx: int, current_price: float) -> float:
        """Close a specific position"""
        if pos_idx >= len(self.positions):
            return 0.0
        
        pos = self.positions[pos_idx]
        
        # Calculate PnL
        if pos['direction'] == 'buy':
            pnl = (current_price - pos['entry_price']) * pos['lot_size'] * self.leverage
        else:
            pnl = (pos['entry_price'] - current_price) * pos['lot_size'] * self.leverage
        
        # Account for spread (already paid on entry)
        net_pnl = pnl
        
        # Update balance - ensure it never goes negative
        new_balance = self.balance + pos['margin'] + net_pnl
        if new_balance < 0:
            # Partial loss to prevent negative balance
            net_pnl = -self.balance - pos['margin'] + 0.01  # Leave $0.01
            new_balance = 0.01
        
        self.balance = new_balance
        self.margin_used -= pos['margin']
        
        # Ensure margin_used doesn't go negative
        self.margin_used = max(0, self.margin_used)
        
        # Record trade
        trade_record = {
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': current_price,
            'lot_size': pos['lot_size'],
            'pnl': net_pnl,
            'duration': self.step_count - pos['entry_step'],
            'expected_sl': (pos['entry_price'] - pos['stop_loss']) * pos['lot_size'] * self.leverage if pos['direction'] == 'buy' else (pos['stop_loss'] - pos['entry_price']) * pos['lot_size'] * self.leverage,
            'expected_tp': (pos['take_profit'] - pos['entry_price']) * pos['lot_size'] * self.leverage if pos['direction'] == 'buy' else (pos['entry_price'] - pos['take_profit']) * pos['lot_size'] * self.leverage
        }
        self.trade_history.append(trade_record)
        
        # Remove position
        del self.positions[pos_idx]
        
        return net_pnl
    
    def _close_all_positions(self, current_price: float) -> float:
        """Close all open positions"""
        total_pnl = 0.0
        while self.positions:
            pnl = self._close_position(0, current_price)
            total_pnl += pnl
        return total_pnl * 0.1  # Small reward for closing
    
    def _calculate_reward(self) -> float:
        """Calculate step reward"""
        reward = 0.0
        
        # Encourage action - small penalty for holding too long
        if len(self.positions) == 0 and self.step_count - self.last_trade_step > 10:
            reward -= 0.5  # Penalty for excessive holding
        
        # Trade-based rewards with SL/TP awareness
        if self.trade_history:
            last_trade = self.trade_history[-1]
            if last_trade['pnl'] > 0:
                reward += 10.0  # Increased profitable trade bonus
                # Extra bonus if hit TP (not just random profit)
                if last_trade['pnl'] > last_trade.get('expected_tp', 0) * 0.8:
                    reward += 5.0  # TP hit bonus
            else:
                reward -= 5.0  # Reduced losing trade penalty
                # Less penalty if hit SL (controlled loss vs random loss)
                if abs(last_trade['pnl']) < abs(last_trade.get('expected_sl', 0)) * 1.2:
                    reward += 2.0  # SL hit mitigation
        
        # Small reward for taking any trading action (not hold)
        if hasattr(self, '_last_action') and self._last_action in [1, 2, 3]:
            reward += 0.2  # Small action bonus
        
        # Position management reward
        for pos in self.positions:
            current_price = self.data.iloc[self.current_idx]['close']
            
            # Reward for positions moving toward TP
            if pos['direction'] == 'buy':
                price_move = (current_price - pos['entry_price']) / pos['entry_price']
                tp_target = (pos['take_profit'] - pos['entry_price']) / pos['entry_price']
                if price_move > 0 and tp_target > 0:
                    reward += 0.5 * (price_move / tp_target)  # Increased progress reward
            else:
                price_move = (pos['entry_price'] - current_price) / pos['entry_price']
                tp_target = (pos['entry_price'] - pos['take_profit']) / pos['entry_price']
                if price_move > 0 and tp_target > 0:
                    reward += 0.5 * (price_move / tp_target)  # Increased progress reward
        
        # Margin safety penalties
        free_margin = self.balance - self.margin_used
        if free_margin < self.initial_balance * 0.1:  # Less than 10% free margin
            reward -= 10.0  # Danger zone penalty
        
        if self.margin_used > 0:
            margin_level = (self.equity / self.margin_used) * 100
            if margin_level < 50:  # Critical margin level
                reward -= 15.0 * (50 - margin_level) / 50  # Escalating penalty
        
        # Account wipeout penalty
        if self.balance <= self.initial_balance * 0.3:  # 70% loss
            reward -= 50.0  # Severe penalty for approaching wipeout
        
        # Drawdown penalty
        if self.max_drawdown > 0:
            reward -= 2.0 * (self.max_drawdown / self.balance)
        
        # Daily profit bonus/penalty - more aggressive
        if self.step_count % 50 == 0:  # Check every 50 steps (more frequent)
            daily_profit = self.balance - self.initial_balance
            if daily_profit >= 5.0:  # Lower threshold for bonus
                reward += 20.0 * (daily_profit / 5.0)  # Scaled bonus
            elif daily_profit < -2.0:  # Penalty for any significant loss
                reward -= 10.0
        
        return reward
    
    def _update_equity(self):
        """Update equity and drawdown"""
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        current_price = self.data.iloc[self.current_idx]['close']
        
        for pos in self.positions:
            if pos['direction'] == 'buy':
                pnl = (current_price - pos['entry_price']) * pos['lot_size'] * self.leverage
            else:
                pnl = (pos['entry_price'] - current_price) * pos['lot_size'] * self.leverage
            unrealized_pnl += pnl
        
        self.equity = self.balance + unrealized_pnl
        
        # Update max equity and drawdown
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Ensure balance never goes negative
        if self.balance <= 0.01:
            return True
        
        # Calculate free margin
        free_margin = self.balance - self.margin_used
        
        # Margin call - insufficient free margin for open positions
        if free_margin <= 0 and self.positions:
            # Force close all positions to prevent negative balance
            self._close_all_positions(self.data.iloc[self.current_idx]['close'])
            return True
        
        # Account wipeout - 95% loss (realistic stop out level)
        if self.balance <= self.initial_balance * 0.05:  # 95% loss
            return True
        
        # Margin level too low (margin level = equity / margin_used * 100)
        if self.margin_used > 0:
            margin_level = (self.equity / self.margin_used) * 100
            if margin_level <= 10:  # 10% margin level = forced liquidation (Exness-like)
                return True
        
        # Success
        if self.balance >= 30.0:
            return True
        
        # Max steps
        if self.step_count >= self.max_steps:
            return True
        
        # Max drawdown
        if self.max_drawdown >= self.max_drawdown_pct:
            return True
        
        # Out of data
        if self.current_idx >= len(self.data) - 1:
            return True
        
        return False
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': self.balance - self.initial_balance,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': 0.0
            }
        
        trades = pd.DataFrame(self.trade_history)
        winning_trades = trades[trades['pnl'] > 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_pnl = self.balance - self.initial_balance
        
        # Sharpe ratio approximation
        if len(trades) > 1:
            returns = trades['pnl'].values
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance
        }
