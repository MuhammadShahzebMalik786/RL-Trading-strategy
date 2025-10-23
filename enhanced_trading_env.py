import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any
from smc_analyzer import SMCAnalyzer, SupportResistanceAnalyzer
from deep_learning_models import DeepLearningEnsemble

class EnhancedTradingEnv(gym.Env):
    def __init__(self, initial_balance: float = 10.0, leverage: int = 2000, max_steps: int = 1000):
        super().__init__()
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.max_steps = max_steps
        
        # Market analysis components
        self.smc_analyzer = SMCAnalyzer()
        self.sr_analyzer = SupportResistanceAnalyzer()
        self.dl_ensemble = DeepLearningEnsemble()
        
        # Action space: 0=hold, 1=buy_small, 2=buy_large, 3=sell_small, 4=sell_large, 5=close_all
        self.action_space = gym.spaces.Discrete(6)
        
        # Enhanced observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32
        )
        
        # Generate synthetic forex data
        self.data = self._generate_forex_data()
        self.reset()
    
    def _generate_forex_data(self, length: int = 5000) -> np.ndarray:
        """Generate realistic forex price data"""
        np.random.seed(42)
        
        # Start with base price
        base_price = 1.1000
        prices = [base_price]
        
        # Generate OHLCV data
        data = []
        
        for i in range(length):
            # Random walk with trend and volatility
            change = np.random.normal(0, 0.0005)  # 5 pip volatility
            if i > 100:
                # Add trend component
                trend = np.sin(i / 200) * 0.0002
                change += trend
            
            new_price = prices[-1] + change
            
            # Generate OHLC from close price
            volatility = abs(np.random.normal(0, 0.0003))
            high = new_price + volatility
            low = new_price - volatility
            open_price = prices[-1] if i > 0 else new_price
            volume = np.random.uniform(1000, 5000)
            
            data.append([open_price, high, low, new_price, volume])
            prices.append(new_price)
        
        return np.array(data)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset account state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.current_step = 100  # Start after enough data for analysis
        self.done = False
        
        # Train deep learning models on historical data
        if not self.dl_ensemble.trained:
            print("Training deep learning models...")
            train_data = self.data[:1000]  # Use first 1000 bars for training
            self.dl_ensemble.train_models(train_data, epochs=20)
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            return self._get_observation(), 0, True, False, {}
        
        # Get current market analysis
        market_analysis = self._analyze_market()
        
        # Execute action based on market analysis
        reward = self._execute_action(action, market_analysis)
        
        # Update positions and account
        self._update_positions()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        self.done = (self.current_step >= len(self.data) - 1 or 
                    self.current_step >= self.max_steps or 
                    self.equity <= self.initial_balance * 0.5)
        
        return self._get_observation(), reward, self.done, False, {
            'balance': self.balance,
            'equity': self.equity,
            'positions': len(self.positions),
            'market_analysis': market_analysis
        }
    
    def _analyze_market(self) -> Dict:
        """Comprehensive market analysis using SMC, S/R, and Deep Learning"""
        current_data = self.data[max(0, self.current_step-100):self.current_step+1]
        
        # SMC Analysis
        smc_analysis = self.smc_analyzer.identify_structure(current_data)
        
        # Support/Resistance Analysis
        sr_analysis = self.sr_analyzer.find_levels(current_data)
        
        # Deep Learning Predictions
        dl_predictions = self.dl_ensemble.predict(current_data)
        dl_signal = self.dl_ensemble.get_ensemble_signal(dl_predictions)
        
        # Current price and levels
        current_price = self.data[self.current_step, 3]  # Close price
        
        # Determine trade signals
        trade_signal = self._generate_trade_signal(smc_analysis, sr_analysis, dl_signal, current_price)
        
        return {
            'smc': smc_analysis,
            'support_resistance': sr_analysis,
            'deep_learning': {
                'predictions': dl_predictions,
                'signal': dl_signal
            },
            'trade_signal': trade_signal,
            'current_price': current_price
        }
    
    def _generate_trade_signal(self, smc: Dict, sr: Dict, dl_signal: int, current_price: float) -> Dict:
        """Generate trading signals based on all analyses"""
        signals = {'action': 'hold', 'confidence': 0.0, 'size': 'small'}
        
        # Deep Learning Signal (primary)
        if dl_signal == 2:  # Buy signal
            signals['action'] = 'buy'
            signals['confidence'] += 0.4
        elif dl_signal == 0:  # Sell signal
            signals['action'] = 'sell'
            signals['confidence'] += 0.4
        
        # SMC Confirmation
        if smc['bos_signals']['bullish'] and len(smc['bos_signals']['bullish']) > 0:
            if signals['action'] == 'buy':
                signals['confidence'] += 0.3
            elif signals['action'] == 'hold':
                signals['action'] = 'buy'
                signals['confidence'] += 0.2
        
        if smc['bos_signals']['bearish'] and len(smc['bos_signals']['bearish']) > 0:
            if signals['action'] == 'sell':
                signals['confidence'] += 0.3
            elif signals['action'] == 'hold':
                signals['action'] = 'sell'
                signals['confidence'] += 0.2
        
        # Support/Resistance Confirmation
        for support in sr['support']:
            if abs(current_price - support['price']) / current_price < 0.001:  # Near support
                if signals['action'] == 'buy':
                    signals['confidence'] += 0.2
                    signals['size'] = 'large' if signals['confidence'] > 0.7 else 'small'
        
        for resistance in sr['resistance']:
            if abs(current_price - resistance['price']) / current_price < 0.001:  # Near resistance
                if signals['action'] == 'sell':
                    signals['confidence'] += 0.2
                    signals['size'] = 'large' if signals['confidence'] > 0.7 else 'small'
        
        return signals
    
    def _execute_action(self, action: int, market_analysis: Dict) -> float:
        """Execute trading action with market analysis consideration"""
        current_price = market_analysis['current_price']
        trade_signal = market_analysis['trade_signal']
        
        reward = 0
        
        # Only execute if RL action aligns with market analysis (smart filtering)
        if action == 0:  # Hold
            reward = 0.01  # Small reward for patience
        
        elif action in [1, 2] and trade_signal['action'] == 'buy':  # Buy actions
            position_size = 0.5 if action == 1 else 1.0  # Small vs large
            if trade_signal['size'] == 'large':
                position_size *= 1.5
            
            position_value = self.balance * position_size * self.leverage
            
            self.positions.append({
                'type': 'buy',
                'entry_price': current_price,
                'size': position_value,
                'step': self.current_step
            })
            
            reward = trade_signal['confidence'] * 0.1
        
        elif action in [3, 4] and trade_signal['action'] == 'sell':  # Sell actions
            position_size = 0.5 if action == 3 else 1.0
            if trade_signal['size'] == 'large':
                position_size *= 1.5
            
            position_value = self.balance * position_size * self.leverage
            
            self.positions.append({
                'type': 'sell',
                'entry_price': current_price,
                'size': position_value,
                'step': self.current_step
            })
            
            reward = trade_signal['confidence'] * 0.1
        
        elif action == 5:  # Close all positions
            reward = self._close_all_positions()
        
        else:
            # Penalize actions that go against market analysis
            reward = -0.05
        
        return reward
    
    def _update_positions(self):
        """Update all open positions"""
        current_price = self.data[self.current_step, 3]
        
        for position in self.positions:
            if position['type'] == 'buy':
                pnl = (current_price - position['entry_price']) * position['size'] / position['entry_price']
            else:
                pnl = (position['entry_price'] - current_price) * position['size'] / position['entry_price']
            
            position['pnl'] = pnl
        
        # Update equity
        total_pnl = sum(pos.get('pnl', 0) for pos in self.positions)
        self.equity = self.balance + total_pnl
    
    def _close_all_positions(self) -> float:
        """Close all positions and calculate reward"""
        total_pnl = sum(pos.get('pnl', 0) for pos in self.positions)
        self.balance += total_pnl
        self.positions = []
        return total_pnl * 0.1  # Reward based on PnL
    
    def _get_observation(self) -> np.ndarray:
        """Get enhanced observation including market analysis"""
        if self.current_step < 50:
            return np.zeros(100, dtype=np.float32)
        
        # Price data (last 20 bars)
        price_data = self.data[self.current_step-19:self.current_step+1, :4].flatten()
        
        # Account state
        account_state = np.array([
            self.balance / self.initial_balance,
            self.equity / self.initial_balance,
            len(self.positions) / 10.0,
            sum(pos.get('pnl', 0) for pos in self.positions) / self.initial_balance
        ])
        
        # Market analysis features
        market_analysis = self._analyze_market()
        
        # SMC features
        smc_features = np.array([
            len(market_analysis['smc']['swing_highs']) / 10.0,
            len(market_analysis['smc']['swing_lows']) / 10.0,
            len(market_analysis['smc']['bos_signals']['bullish']) / 5.0,
            len(market_analysis['smc']['bos_signals']['bearish']) / 5.0,
            len(market_analysis['smc']['order_blocks']) / 5.0
        ])
        
        # S/R features
        sr_features = np.array([
            len(market_analysis['support_resistance']['support']) / 5.0,
            len(market_analysis['support_resistance']['resistance']) / 5.0,
            market_analysis['support_resistance']['support'][0]['strength'] if market_analysis['support_resistance']['support'] else 0,
            market_analysis['support_resistance']['resistance'][0]['strength'] if market_analysis['support_resistance']['resistance'] else 0
        ])
        
        # Deep learning features
        dl_probs = list(market_analysis['deep_learning']['predictions'].values())
        dl_features = np.concatenate([pred for pred in dl_probs])  # All model predictions
        
        # Pad to reach 100 features
        observation = np.concatenate([price_data, account_state, smc_features, sr_features, dl_features])
        
        # Pad or truncate to exactly 100 features
        if len(observation) < 100:
            observation = np.pad(observation, (0, 100 - len(observation)))
        else:
            observation = observation[:100]
        
        return observation.astype(np.float32)
