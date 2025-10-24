import MetaTrader5 as mt5
import numpy as np
from stable_baselines3 import PPO
import time

class ETHTrader:
    def __init__(self):
        self.model = PPO.load("models/best_model.zip")
        self.symbol = "ETHUSDm"
        self.lot_size = 0.1  # Minimum lot size
        
    def execute_trade(self, action):
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return "No price"
            
        if action == 0:  # Hold
            return "Hold"
            
        elif action in [1, 2, 3]:  # Buy Small/Med/Large
            lot = self.lot_size  # Use minimum lot for all sizes
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "RL Bot Buy",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            return f"Buy {lot} - {'Success' if result and result.retcode == 10009 else 'Failed'}"
            
        elif action in [4, 5, 6]:  # Sell Small/Med/Large
            lot = self.lot_size  # Use minimum lot for all sizes
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "deviation": 20,
                "magic": 234000,
                "comment": "RL Bot Sell",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            return f"Sell {lot} - {'Success' if result and result.retcode == 10009 else 'Failed'}"
            
        elif action == 7:  # Close All
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                return "No positions"
            closed = 0
            for pos in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": tick.bid if pos.type == 0 else tick.ask,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "RL Close",
                }
                result = mt5.order_send(close_request)
                if result and result.retcode == 10009:
                    closed += 1
            return f"Closed {closed}"
        
        return "Hold"  # Default to hold for other actions
    
    def start_trading(self):
        if not mt5.initialize():
            return
            
        print(f"✅ Trading {self.symbol} with lot size {self.lot_size}")
        
        step = 0
        while True:
            try:
                account = mt5.account_info()
                tick = mt5.symbol_info_tick(self.symbol)
                
                # Use trained model
                obs = np.random.random(750)  # Placeholder observation
                action, _ = self.model.predict(obs, deterministic=True)
                
                result = self.execute_trade(action)
                
                print(f"Step {step} | Price: ${tick.bid:.2f} | Balance: ${account.balance:.2f} | {result}")
                
                step += 1
                time.sleep(60)  # 1 minute intervals
                
            except KeyboardInterrupt:
                print("\n✅ Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    trader = ETHTrader()
    trader.start_trading()
