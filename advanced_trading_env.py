import gymnasium as gym
import numpy as np
import pandas as pd
import requests
import time
from typing import Dict, List, Tuple, Optional
import ta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradingEnv(gym.Env):
    def __init__(self, 
                 initial_balance: float = 10.0,
                 leverage: int = 2000,
                 spread_per_lot: float = 1.6,
                 min_lot: float = 0.01,
                 max_lot: float = 1.0,
                 margin_per_lot: float = 0.05,  # Much lower margin requirement
                 max_steps: int = 1000,
                 lookback: int = 50):
        
        super().__init__()
        
        # Enhanced parameters
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.spread_per_lot = spread_per_lot
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.margin_per_lot = margin_per_lot
        self.max_steps = max_steps
        self.lookback = lookback
        
        # Advanced risk management
        self.max_drawdown_pct = 0.15
        self.max_concurrent_trades = 5
        self.dynamic_stop_loss = True
        self.adaptive_position_sizing = True
        self.volatility_adjustment = True
        
        # Market regime detection
        self.trend_threshold = 0.02
        self.volatility_window = 20
        self.regime_memory = 10
        
        # Enhanced action space: [hold, buy_small, buy_medium, buy_large, sell_small, sell_medium, sell_large, close_all, close_profitable, close_losing]
        self.action_space = gym.spaces.Discrete(10)
        
        # Enhanced state space with more features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lookback * 15 + 25,), dtype=np.float32)
        
        # Initialize scalers
        self.price_scaler = StandardScaler()
        self.indicator_scaler = StandardScaler()
        
        self.reset()
    
    def _get_market_data(self) -> pd.DataFrame:
        """Enhanced market data with more sophisticated generation"""
        try:
            # Try real data first
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'ETHUSDT',
                'interval': '1m',
                'limit': 1000
            }
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                return df[['open', 'high', 'low', 'close', 'volume']].iloc[-self.max_steps-self.lookback:]
            
        except Exception as e:
            print(f"Real data failed: {e}, using synthetic data")
        
        # Enhanced synthetic data with realistic patterns
        np.random.seed(42)
        n_points = self.max_steps + self.lookback
        
        # Generate base price with trend and cycles
        t = np.arange(n_points)
        trend = 0.001 * t + 0.0005 * np.sin(t / 50)  # Long-term trend + cycle
        noise = np.random.normal(0, 0.02, n_points)  # Market noise
        volatility_regime = 1 + 0.5 * np.sin(t / 100)  # Volatility clustering
        
        price_changes = (trend + noise) * volatility_regime
        prices = 2000 * np.exp(np.cumsum(price_changes))
        
        # Generate OHLCV with realistic relationships
        df_data = []
        for i in range(n_points):
            base_price = prices[i]
            volatility = abs(price_changes[i]) * 100
            
            # Realistic OHLC generation
            open_price = base_price * (1 + np.random.normal(0, 0.001))
            high_low_range = volatility * np.random.uniform(0.5, 2.0)
            high_price = open_price + high_low_range * np.random.uniform(0.3, 1.0)
            low_price = open_price - high_low_range * np.random.uniform(0.3, 1.0)
            close_price = open_price + (high_price - low_price) * np.random.uniform(-0.4, 0.4)
            volume = np.random.lognormal(10, 1)
            
            df_data.append([open_price, high_price, low_price, close_price, volume])
        
        return pd.DataFrame(df_data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate comprehensive technical indicators"""
        indicators = []
        
        # Price-based indicators
        indicators.extend([
            ta.trend.sma_indicator(df['close'], window=10),
            ta.trend.sma_indicator(df['close'], window=20),
            ta.trend.ema_indicator(df['close'], window=12),
            ta.trend.ema_indicator(df['close'], window=26),
        ])
        
        # Momentum indicators
        indicators.extend([
            ta.momentum.rsi(df['close'], window=14),
            ta.momentum.stoch(df['high'], df['low'], df['close']),
            ta.momentum.williams_r(df['high'], df['low'], df['close']),
        ])
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['close'])
        indicators.extend([
            bb.bollinger_hband(),
            bb.bollinger_lband(),
            bb.bollinger_wband(),
            ta.volatility.average_true_range(df['high'], df['low'], df['close']),
        ])
        
        # Volume indicators
        try:
            indicators.extend([
                ta.volume.on_balance_volume(df['close'], df['volume']),
                ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume']),
            ])
        except AttributeError:
            # Fallback if specific volume indicators don't exist
            indicators.extend([
                df['volume'].rolling(window=20).mean(),  # Volume SMA
                df['volume'].rolling(window=20).std(),   # Volume volatility
            ])
        
        # Trend indicators
        macd = ta.trend.MACD(df['close'])
        indicators.extend([
            macd.macd(),
            macd.macd_signal(),
        ])
        
        # Convert to array and handle NaN
        indicator_array = np.column_stack(indicators)
        indicator_array = np.nan_to_num(indicator_array, nan=0.0)
        
        return indicator_array
    
    def _detect_market_regime(self, prices: np.ndarray) -> Dict[str, float]:
        """Detect current market regime"""
        if len(prices) < self.volatility_window:
            return {'trend': 0.0, 'volatility': 1.0, 'momentum': 0.0}
        
        # Trend detection
        recent_prices = prices[-self.volatility_window:]
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Volatility measurement
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Momentum measurement
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0.0
        
        return {
            'trend': np.tanh(trend / self.trend_threshold),  # Normalize to [-1, 1]
            'volatility': min(volatility / 0.5, 3.0),  # Cap at 3x normal
            'momentum': np.tanh(momentum * 100)  # Normalize momentum
        }
    
    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        """Dynamic position sizing based on confidence and volatility"""
        if not self.adaptive_position_sizing:
            return self.min_lot
        
        # Kelly criterion inspired sizing
        base_size = 0.01  # Much smaller base size
        confidence_multiplier = max(0.1, min(confidence, 1.0))
        volatility_adjustment = 1.0 / max(volatility, 0.5)
        
        # Account for current drawdown
        current_drawdown = max(0, (self.initial_balance - self.balance) / self.initial_balance)
        drawdown_adjustment = max(0.1, 1.0 - current_drawdown * 2)
        
        size = base_size * confidence_multiplier * volatility_adjustment * drawdown_adjustment
        return np.clip(size, 0.01, self.max_lot)  # Allow smaller positions
    
    def _calculate_dynamic_stops(self, entry_price: float, position_type: str, volatility: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit"""
        if not self.dynamic_stop_loss:
            stop_loss = entry_price * (0.98 if position_type == 'long' else 1.02)
            take_profit = entry_price * (1.04 if position_type == 'long' else 0.96)
            return stop_loss, take_profit
        
        # ATR-based stops
        atr_multiplier = max(1.5, min(3.0, volatility * 10))
        stop_distance = entry_price * 0.01 * atr_multiplier
        profit_distance = stop_distance * 2.0  # 2:1 risk-reward
        
        if position_type == 'long':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance
        
        return stop_loss, take_profit
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset environment state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin_used = 0.0
        self.positions = []
        self.trade_history = []
        self.step_count = 0
        self.last_action_step = -10
        
        # Performance tracking
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Market regime tracking
        self.regime_history = []
        
        # Get market data
        self.data = self._get_market_data()
        self.indicators = self._calculate_advanced_indicators(self.data)
        
        # Fit scalers
        price_data = self.data[['open', 'high', 'low', 'close', 'volume']].values
        price_data = np.nan_to_num(price_data, nan=0.0)
        self.price_scaler.fit(price_data)
        
        indicators_clean = np.nan_to_num(self.indicators, nan=0.0)
        self.indicator_scaler.fit(indicators_clean)
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Enhanced observation with market regime and advanced features"""
        if self.step_count + self.lookback >= len(self.data):
            # Pad with last available data
            end_idx = len(self.data)
            start_idx = max(0, end_idx - self.lookback)
            price_window = self.data.iloc[start_idx:end_idx]
            indicator_window = self.indicators[start_idx:end_idx]
        else:
            start_idx = self.step_count
            end_idx = self.step_count + self.lookback
            price_window = self.data.iloc[start_idx:end_idx]
            indicator_window = self.indicators[start_idx:end_idx]
        
        # Normalize price data
        price_data_clean = np.nan_to_num(price_window.values, nan=0.0)
        price_features = self.price_scaler.transform(price_data_clean).flatten()
        
        # Normalize indicators
        indicator_data_clean = np.nan_to_num(indicator_window, nan=0.0)
        indicator_features = self.indicator_scaler.transform(indicator_data_clean).flatten()
        
        # Market regime features
        current_prices = price_window['close'].values
        regime = self._detect_market_regime(current_prices)
        regime_features = [regime['trend'], regime['volatility'], regime['momentum']]
        
        # Account features
        account_features = [
            self.balance / self.initial_balance - 1,  # Normalized balance change
            self.equity / self.initial_balance - 1,   # Normalized equity change
            self.margin_used / max(self.balance, 0.01),  # Margin utilization
            len(self.positions) / self.max_concurrent_trades,  # Position utilization
            self.max_drawdown,  # Current max drawdown
            (self.step_count - self.last_action_step) / 10.0,  # Time since last action
        ]
        
        # Position features
        position_features = []
        for i in range(5):  # Max 5 positions
            if i < len(self.positions):
                pos = self.positions[i]
                current_price = price_window['close'].iloc[-1]
                unrealized_pnl = self._calculate_unrealized_pnl(pos, current_price)
                position_features.extend([
                    pos['size'] / self.max_lot,  # Normalized size
                    1.0 if pos['type'] == 'long' else -1.0,  # Position type
                    unrealized_pnl / self.initial_balance,  # Normalized PnL
                    (self.step_count - pos['entry_step']) / 100.0,  # Position age
                ])
            else:
                position_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Performance features
        win_rate = self.winning_trades / max(self.total_trades, 1)
        performance_features = [
            win_rate,
            self.total_pnl / self.initial_balance,
            len(self.trade_history) / max(self.step_count, 1),  # Trade frequency
        ]
        
        # Combine all features
        observation = np.concatenate([
            price_features,
            indicator_features,
            regime_features,
            account_features,
            position_features,
            performance_features
        ]).astype(np.float32)
        
        # Handle NaN and infinite values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip extreme values
        observation = np.clip(observation, -10.0, 10.0)
        
        # Ensure fixed size
        target_size = self.observation_space.shape[0]
        if len(observation) < target_size:
            observation = np.pad(observation, (0, target_size - len(observation)))
        elif len(observation) > target_size:
            observation = observation[:target_size]
        
        return observation
    
    def step(self, action: int):
        if self.step_count >= self.max_steps - 1:
            return self._get_observation(), 0, True, False, self.get_metrics()
        
        # Get current price
        current_price = self.data.iloc[self.step_count + self.lookback]['close']
        
        # Update positions and calculate PnL
        self._update_positions(current_price)
        
        # Execute action with enhanced logic
        reward = self._execute_enhanced_action(action, current_price)
        
        # Update market regime
        prices = self.data.iloc[max(0, self.step_count):self.step_count + self.lookback]['close'].values
        regime = self._detect_market_regime(prices)
        self.regime_history.append(regime)
        
        self.step_count += 1
        
        # Check termination conditions
        terminated = (self.balance <= self.initial_balance * 0.1 or  # Margin call
                     self.max_drawdown >= self.max_drawdown_pct or
                     self.step_count >= self.max_steps - 1)
        
        return self._get_observation(), reward, terminated, False, self.get_metrics()
    
    def _execute_enhanced_action(self, action: int, current_price: float) -> float:
        """Execute action with enhanced position sizing and risk management"""
        reward = 0.0
        
        # Get market regime for decision making
        regime = self.regime_history[-1] if self.regime_history else {'volatility': 1.0, 'trend': 0.0}
        
        if action == 0:  # Hold
            reward = 0.0  # No penalty for hold
            
        elif action in [1, 2, 3]:  # Buy (small, medium, large)
            size_multipliers = [0.5, 1.0, 2.0]
            base_size = self._calculate_position_size(0.7, regime['volatility'])
            size = base_size * size_multipliers[action - 1]
            
            if self._can_open_position(size, current_price):
                stop_loss, take_profit = self._calculate_dynamic_stops(current_price, 'long', regime['volatility'])
                position = {
                    'type': 'long',
                    'size': size,
                    'entry_price': current_price + self.spread_per_lot / size,  # Include spread
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_step': self.step_count,
                    'margin': size * self.margin_per_lot
                }
                self.positions.append(position)
                self.margin_used += position['margin']
                self.last_action_step = self.step_count
                reward = 0.1  # Small reward for taking action
            else:
                reward = -0.5  # Penalty for invalid action
                
        elif action in [4, 5, 6]:  # Sell (small, medium, large)
            size_multipliers = [0.5, 1.0, 2.0]
            base_size = self._calculate_position_size(0.7, regime['volatility'])
            size = base_size * size_multipliers[action - 4]
            
            if self._can_open_position(size, current_price):
                stop_loss, take_profit = self._calculate_dynamic_stops(current_price, 'short', regime['volatility'])
                position = {
                    'type': 'short',
                    'size': size,
                    'entry_price': current_price - self.spread_per_lot / size,  # Include spread
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_step': self.step_count,
                    'margin': size * self.margin_per_lot
                }
                self.positions.append(position)
                self.margin_used += position['margin']
                self.last_action_step = self.step_count
                reward = 0.1
            else:
                reward = -0.5
                
        elif action == 7:  # Close all positions
            reward = self._close_all_positions(current_price)
            
        elif action == 8:  # Close profitable positions
            reward = self._close_profitable_positions(current_price)
            
        elif action == 9:  # Close losing positions
            reward = self._close_losing_positions(current_price)
        
        return reward
    
    def _can_open_position(self, size: float, price: float) -> bool:
        """Enhanced position validation"""
        required_margin = size * self.margin_per_lot
        available_margin = self.balance - self.margin_used
        
        return (available_margin >= required_margin and
                len(self.positions) < self.max_concurrent_trades and
                size >= 0.01 and  # Allow very small positions
                self.step_count - self.last_action_step >= 1)  # Shorter cooldown
    
    def _close_profitable_positions(self, current_price: float) -> float:
        """Close only profitable positions"""
        reward = 0.0
        positions_to_remove = []
        
        for i, pos in enumerate(self.positions):
            pnl = self._calculate_unrealized_pnl(pos, current_price)
            if pnl > 0:
                reward += self._close_position(pos, current_price, pnl)
                positions_to_remove.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
        
        return reward
    
    def _close_losing_positions(self, current_price: float) -> float:
        """Close only losing positions"""
        reward = 0.0
        positions_to_remove = []
        
        for i, pos in enumerate(self.positions):
            pnl = self._calculate_unrealized_pnl(pos, current_price)
            if pnl < 0:
                reward += self._close_position(pos, current_price, pnl)
                positions_to_remove.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
        
        return reward
    
    def _close_all_positions(self, current_price: float) -> float:
        """Close all positions"""
        total_reward = 0.0
        
        for pos in self.positions:
            pnl = self._calculate_unrealized_pnl(pos, current_price)
            total_reward += self._close_position(pos, current_price, pnl)
        
        self.positions.clear()
        return total_reward
    
    def _close_position(self, position: dict, current_price: float, pnl: float) -> float:
        """Close individual position and calculate reward"""
        self.balance += pnl
        self.margin_used -= position['margin']
        self.total_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            reward = 10.0 * (pnl / self.initial_balance)  # Scaled reward
        else:
            reward = -5.0 * abs(pnl / self.initial_balance)  # Scaled penalty
        
        # Record trade
        self.trade_history.append({
            'type': position['type'],
            'size': position['size'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'pnl': pnl,
            'duration': self.step_count - position['entry_step']
        })
        
        return reward
    
    def _update_positions(self, current_price: float):
        """Update positions with stop loss and take profit"""
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            # Check stop loss and take profit
            if pos['type'] == 'long':
                if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                    positions_to_close.append(i)
            else:  # short
                if current_price >= pos['stop_loss'] or current_price <= pos['take_profit']:
                    positions_to_close.append(i)
        
        # Close positions that hit stops
        for i in reversed(positions_to_close):
            pos = self.positions[i]
            pnl = self._calculate_unrealized_pnl(pos, current_price)
            self._close_position(pos, current_price, pnl)
            self.positions.pop(i)
        
        # Update equity and drawdown
        self.equity = self.balance + sum(self._calculate_unrealized_pnl(pos, current_price) for pos in self.positions)
        self.peak_balance = max(self.peak_balance, self.equity)
        self.max_drawdown = max(self.max_drawdown, (self.peak_balance - self.equity) / self.peak_balance)
    
    def _calculate_unrealized_pnl(self, position: dict, current_price: float) -> float:
        """Calculate unrealized PnL for a position"""
        if position['type'] == 'long':
            return (current_price - position['entry_price']) * position['size'] * self.leverage
        else:
            return (position['entry_price'] - current_price) * position['size'] * self.leverage
    
    def get_metrics(self) -> Dict[str, float]:
        """Enhanced metrics calculation"""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # Calculate Sharpe ratio
        if len(self.trade_history) > 1:
            returns = [trade['pnl'] / self.initial_balance for trade in self.trade_history]
            sharpe_ratio = np.mean(returns) / max(np.std(returns), 0.001) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        return {
            'final_balance': self.balance,
            'total_pnl': self.total_pnl,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity': self.equity,
            'margin_used': self.margin_used,
            'open_positions': len(self.positions)
        }
