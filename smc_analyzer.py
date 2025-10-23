import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

class SMCAnalyzer:
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        
    def identify_structure(self, data: np.ndarray) -> Dict:
        """Identify market structure using SMC principles"""
        highs = data[:, 1]  # High prices
        lows = data[:, 2]   # Low prices
        closes = data[:, 3] # Close prices
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(highs, mode='high')
        swing_lows = self._find_swing_points(lows, mode='low')
        
        # Identify BOS (Break of Structure)
        bos_signals = self._detect_bos(swing_highs, swing_lows, closes)
        
        # Find liquidity zones
        liquidity_zones = self._find_liquidity_zones(highs, lows)
        
        # Detect order blocks
        order_blocks = self._detect_order_blocks(data)
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'bos_signals': bos_signals,
            'liquidity_zones': liquidity_zones,
            'order_blocks': order_blocks
        }
    
    def _find_swing_points(self, prices: np.ndarray, mode: str, window: int = 5) -> List[int]:
        """Find swing highs/lows"""
        points = []
        for i in range(window, len(prices) - window):
            if mode == 'high':
                if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
                   all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                    points.append(i)
            else:
                if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
                   all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                    points.append(i)
        return points
    
    def _detect_bos(self, swing_highs: List[int], swing_lows: List[int], closes: np.ndarray) -> Dict:
        """Detect Break of Structure"""
        bos_signals = {'bullish': [], 'bearish': []}
        
        if len(swing_lows) >= 2:
            for i in range(1, len(swing_lows)):
                prev_low = closes[swing_lows[i-1]]
                curr_idx = swing_lows[i]
                if closes[curr_idx] > prev_low:
                    bos_signals['bullish'].append(curr_idx)
        
        if len(swing_highs) >= 2:
            for i in range(1, len(swing_highs)):
                prev_high = closes[swing_highs[i-1]]
                curr_idx = swing_highs[i]
                if closes[curr_idx] < prev_high:
                    bos_signals['bearish'].append(curr_idx)
        
        return bos_signals
    
    def _find_liquidity_zones(self, highs: np.ndarray, lows: np.ndarray) -> Dict:
        """Identify liquidity zones (support/resistance)"""
        zones = {'support': [], 'resistance': []}
        
        # Find resistance zones from recent highs
        recent_highs = highs[-20:]
        resistance_level = np.percentile(recent_highs, 90)
        zones['resistance'].append(resistance_level)
        
        # Find support zones from recent lows
        recent_lows = lows[-20:]
        support_level = np.percentile(recent_lows, 10)
        zones['support'].append(support_level)
        
        return zones
    
    def _detect_order_blocks(self, data: np.ndarray) -> List[Dict]:
        """Detect institutional order blocks"""
        order_blocks = []
        closes = data[:, 3]
        volumes = data[:, 4] if data.shape[1] > 4 else np.ones(len(data))
        
        for i in range(10, len(data) - 5):
            # Look for high volume candles followed by strong moves
            if volumes[i] > np.mean(volumes[i-10:i]) * 1.5:
                # Check for strong move after
                future_move = abs(closes[i+5] - closes[i]) / closes[i]
                if future_move > 0.01:  # 1% move
                    order_blocks.append({
                        'index': i,
                        'price': closes[i],
                        'strength': future_move
                    })
        
        return order_blocks

class SupportResistanceAnalyzer:
    def __init__(self, min_touches: int = 2, tolerance: float = 0.001):
        self.min_touches = min_touches
        self.tolerance = tolerance
    
    def find_levels(self, data: np.ndarray) -> Dict:
        """Find support and resistance levels"""
        highs = data[:, 1]
        lows = data[:, 2]
        
        support_levels = self._find_horizontal_levels(lows, 'support')
        resistance_levels = self._find_horizontal_levels(highs, 'resistance')
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def _find_horizontal_levels(self, prices: np.ndarray, level_type: str) -> List[Dict]:
        """Find horizontal support/resistance levels"""
        levels = []
        
        # Group similar price levels
        unique_prices = []
        for price in prices:
            if not any(abs(price - up) / up < self.tolerance for up in unique_prices):
                unique_prices.append(price)
        
        # Count touches for each level
        for level_price in unique_prices:
            touches = sum(1 for price in prices if abs(price - level_price) / level_price < self.tolerance)
            
            if touches >= self.min_touches:
                strength = min(touches / len(prices) * 10, 1.0)
                levels.append({
                    'price': level_price,
                    'touches': touches,
                    'strength': strength,
                    'type': level_type
                })
        
        return sorted(levels, key=lambda x: x['strength'], reverse=True)[:5]
