import MetaTrader5 as mt5
import numpy as np
from stable_baselines3 import PPO
import time

class SimpleMT5Trader:
    def __init__(self):
        self.model = PPO.load("models/best_model.zip")
        self.symbol = "ETHUSD"
        self.lot_size = 0.01
        
    def start_trading(self):
        # Initialize MT5 (uses existing connection)
        if not mt5.initialize():
            print("❌ MT5 not available")
            return
            
        print("✅ Using existing MT5 connection")
        
        # Check account
        account = mt5.account_info()
        if not account:
            print("❌ No account info")
            return
            
        print(f"Account: {account.login}")
        print(f"Balance: ${account.balance:.2f}")
        
        # Check symbol
        if not mt5.symbol_info(self.symbol):
            print(f"❌ {self.symbol} not available")
            # Try alternatives
            for alt in ["ETHUSDT", "ETH"]:
                if mt5.symbol_info(alt):
                    self.symbol = alt
                    print(f"✅ Using {alt}")
                    break
        
        step = 0
        while True:
            try:
                # Get current price
                tick = mt5.symbol_info_tick(self.symbol)
                if not tick:
                    print("❌ No price data")
                    time.sleep(5)
                    continue
                
                # Simple action (for testing)
                action = step % 10  # Cycle through actions
                
                # Get account info
                account = mt5.account_info()
                
                print(f"Step {step} | Price: ${tick.bid:.2f} | Balance: ${account.balance:.2f} | Action: {action}")
                
                step += 1
                time.sleep(10)  # 10 second intervals
                
            except KeyboardInterrupt:
                print("\n✅ Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    trader = SimpleMT5Trader()
    trader.start_trading()
