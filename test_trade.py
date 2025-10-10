"""
Simple test trade script
"""

from exness_integration import ExnessDemo

def test_trade():
    """Place a simple test trade"""
    exness = ExnessDemo()
    
    # Connect
    if not exness.connect():
        print("❌ Connection failed")
        return
    
    # Get current price
    price_data = exness.get_price("ETHUSDm")
    if not price_data:
        print("❌ Cannot get price")
        return
    
    # Get symbol info to check minimum volume
    import MetaTrader5 as mt5
    symbol_info = mt5.symbol_info("ETHUSDm")
    if symbol_info:
        min_volume = symbol_info.volume_min
        volume_step = symbol_info.volume_step
        print(f"📏 Min volume: {min_volume}, Volume step: {volume_step}")
        
        # Use minimum volume
        volume = min_volume
    else:
        print("⚠️  Using default volume")
        volume = 0.1  # Try standard minimum
    
    current_price = price_data['bid']
    print(f"💰 Current ETH price: ${current_price}")
    print(f"📊 Using volume: {volume}")
    
    # Calculate SL/TP
    sl = current_price * 0.98  # 2% stop loss
    tp = current_price * 1.04  # 4% take profit
    
    print(f"🛑 Stop Loss: ${sl:.2f}")
    print(f"🎯 Take Profit: ${tp:.2f}")
    
    # Place BUY order
    print("📈 Placing BUY order...")
    result = exness.place_order(
        symbol="ETHUSDm",
        order_type="buy",
        volume=volume,  # Use correct volume
        sl=sl,
        tp=tp
    )
    
    if result['status'] == 'success':
        print("✅ Trade placed successfully!")
        print(f"Order ID: {result['order_id']}")
        print(f"Volume: {result['volume']} lots")
        print(f"Price: ${result['price']}")
    else:
        print(f"❌ Trade failed: {result.get('message', 'Unknown error')}")
    
    # Show account status
    account = exness.get_account_info()
    positions = exness.get_positions()
    
    print(f"\n📊 Account Status:")
    print(f"Balance: ${account['balance']}")
    print(f"Equity: ${account['equity']}")
    print(f"Open Positions: {len(positions)}")
    
    exness.disconnect()

if __name__ == "__main__":
    print("🧪 Testing Trade Execution...")
    test_trade()
