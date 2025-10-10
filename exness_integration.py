"""
Exness Demo Account Integration - LIVE CONNECTION
Account: 259296516 | Server: Exness-MT5Trial15 | Leverage: 1:2000
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class ExnessDemo:
    def __init__(self):
        """Initialize Exness demo connection with your credentials"""
        self.account_id = 259296516
        self.password = "ItsAMt5ac3#@"
        self.server = "Exness-MT5Trial15"
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to Exness demo account"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                print(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Connect to account
            if not mt5.login(self.account_id, password=self.password, server=self.server):
                print(f"❌ Login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            print(f"✅ Connected to Exness Demo Account: {self.account_id}")
            print(f"📡 Server: {self.server}")
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                print(f"💰 Balance: ${account_info.balance}")
                print(f"📊 Equity: ${account_info.equity}")
                print(f"🔧 Leverage: 1:{account_info.leverage}")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get real account balance, equity, margin info"""
        if not self.connected:
            return {}
        
        account_info = mt5.account_info()
        if account_info is None:
            return {}
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'currency': account_info.currency,
            'leverage': account_info.leverage
        }
    
    def get_positions(self) -> List[Dict]:
        """Get real open positions"""
        if not self.connected:
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        position_list = []
        for pos in positions:
            position_list.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == 0 else 'sell',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'time': pos.time
            })
        
        return position_list
    
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   sl: float = None, tp: float = None) -> Dict:
        """Place a real trading order on Exness demo"""
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'status': 'error', 'message': f'Symbol {symbol} not found'}
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {'status': 'error', 'message': f'Failed to get {symbol} price'}
        
        # Determine order type and price
        if order_type.lower() == 'buy':
            trade_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            trade_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "RL Trading Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL/TP if provided
        if sl:
            request["sl"] = sl
        if tp:
            request["tp"] = tp
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'status': 'error',
                'message': f'Order failed: {result.comment}',
                'retcode': result.retcode
            }
        
        print(f"✅ Order placed: {order_type} {volume} {symbol} at {price}")
        if sl: print(f"🛑 Stop Loss: {sl}")
        if tp: print(f"🎯 Take Profit: {tp}")
        
        return {
            'status': 'success',
            'order_id': result.order,
            'symbol': symbol,
            'type': order_type,
            'volume': volume,
            'price': result.price,
            'sl': sl,
            'tp': tp
        }
    
    def close_position(self, ticket: int) -> bool:
        """Close a specific position"""
        if not self.connected:
            return False
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        
        # Prepare close request
        if position.type == 0:  # Buy position
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:  # Sell position
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": trade_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "RL Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Position {ticket} closed")
            return True
        else:
            print(f"❌ Failed to close position {ticket}: {result.comment}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all open positions"""
        positions = self.get_positions()
        success = True
        
        for pos in positions:
            if not self.close_position(pos['ticket']):
                success = False
        
        return success
    
    def get_price(self, symbol: str) -> Dict:
        """Get current bid/ask prices"""
        if not self.connected:
            return {}
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        
        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': tick.time
        }
    
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols on the account"""
        if not self.connected:
            return []
        
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        
        symbol_names = [symbol.name for symbol in symbols]
        return symbol_names
    
    def find_eth_symbols(self) -> List[str]:
        """Find all ETH-related symbols"""
        all_symbols = self.get_all_symbols()
        eth_symbols = [s for s in all_symbols if 'ETH' in s.upper()]
        return eth_symbols
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("🔌 Disconnected from Exness")

# Test connection
def test_connection():
    """Test the Exness demo connection"""
    exness = ExnessDemo()
    
    if exness.connect():
        print("\n🎯 CONNECTION SUCCESSFUL!")
        
        # Get account info
        account = exness.get_account_info()
        print(f"\n📊 Account Status:")
        print(f"Balance: ${account.get('balance', 0)}")
        print(f"Equity: ${account.get('equity', 0)}")
        print(f"Free Margin: ${account.get('free_margin', 0)}")
        print(f"Leverage: 1:{account.get('leverage', 0)}")
        
        # Get positions
        positions = exness.get_positions()
        print(f"\n📈 Open Positions: {len(positions)}")
        
        # Get EURUSD price
        price = exness.get_price("EURUSD")
        if price:
            print(f"\n💱 EURUSD: Bid={price['bid']}, Ask={price['ask']}")
        
        # Find ETH symbols
        eth_symbols = exness.find_eth_symbols()
        print(f"\n🔍 Available ETH symbols: {eth_symbols}")
        
        # Test first ETH symbol if available
        if eth_symbols:
            eth_price = exness.get_price(eth_symbols[0])
            if eth_price:
                print(f"💰 {eth_symbols[0]}: Bid={eth_price['bid']}, Ask={eth_price['ask']}")
        
        exness.disconnect()
        return True
    else:
        print("❌ Connection failed!")
        return False

if __name__ == "__main__":
    print("🚀 Testing Exness Demo Connection...")
    test_connection()
