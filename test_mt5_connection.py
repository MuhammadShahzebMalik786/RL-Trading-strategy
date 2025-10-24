import MetaTrader5 as mt5

def test_connection():
    print("🔍 Testing MT5 connection...")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        print(f"Error: {mt5.last_error()}")
        return
    
    print("✅ MT5 initialized")
    
    # Get input
    login = int(input("Enter Account Number: "))
    password = input("Enter Password: ")
    server = input("Enter Server Name: ")
    
    # Test login
    if mt5.login(login, password=password, server=server):
        print("✅ Login successful!")
        
        # Show account info
        account = mt5.account_info()
        if account:
            print(f"Name: {account.name}")
            print(f"Balance: ${account.balance}")
            print(f"Server: {account.server}")
            print(f"Currency: {account.currency}")
        
        # Test symbol
        symbol_info = mt5.symbol_info("ETHUSD")
        if symbol_info:
            print("✅ ETHUSD available")
        else:
            print("❌ ETHUSD not available")
            # Try alternatives
            for sym in ["ETHUSDT", "ETH/USD", "ETHEREUM"]:
                if mt5.symbol_info(sym):
                    print(f"✅ Alternative found: {sym}")
                    break
    else:
        print("❌ Login failed")
        print(f"Error: {mt5.last_error()}")
    
    mt5.shutdown()

if __name__ == "__main__":
    test_connection()
