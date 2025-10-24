import MetaTrader5 as mt5
import numpy as np
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import time
import pandas as pd

class MT5LiveTrader:
    def __init__(self, login, password, server):
        self.login = login
        self.password = password
        self.server = server
        self.model = PPO.load("models/best_model.zip")
        self.symbol = "ETHUSD"
        self.lot_size = 0.01  # Minimum lot
        
    def connect(self):
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
            
        if not mt5.login(self.login, password=self.password, server=self.server):
            print("❌ MT5 login failed")
            return False
            
        print("✅ Connected to MT5")
        return True
    
    def get_account_info(self):
        account = mt5.account_info()
        if account:
            print(f"Balance: ${account.balance:.2f}")
            print(f"Equity: ${account.equity:.2f}")
            print(f"Margin: ${account.margin:.2f}")
        return account
    
    def get_market_data(self):
        # Get last 50 bars for model input
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 50)
        if rates is None:
            return None
        return pd.DataFrame(rates)
    
    def execute_trade(self, action):
        current_price = mt5.symbol_info_tick(self.symbol).bid
        
        if action == 0:  # Hold
            return "Hold"
            
        elif action in [1, 2, 3]:  # Buy Small/Med/Large
            lot = self.lot_size * (0.5 if action == 1 else 1.0 if action == 2 else 2.0)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "price": current_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"RL Bot Buy {action}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            return f"Buy {lot} lots - {result.comment if result else 'Failed'}"
            
        elif action in [4, 5, 6]:  # Sell Small/Med/Large
            lot = self.lot_size * (0.5 if action == 4 else 1.0 if action == 5 else 2.0)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"RL Bot Sell {action}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            return f"Sell {lot} lots - {result.comment if result else 'Failed'}"
            
        elif action == 7:  # Close All
            positions = mt5.positions_get(symbol=self.symbol)
            closed = 0
            for pos in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": current_price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "RL Bot Close All",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                if mt5.order_send(close_request):
                    closed += 1
            return f"Closed {closed} positions"
            
        elif action in [8, 9]:  # Close Profit/Loss
            positions = mt5.positions_get(symbol=self.symbol)
            closed = 0
            for pos in positions:
                profit_condition = pos.profit > 0 if action == 8 else pos.profit < 0
                if profit_condition:
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                        "position": pos.ticket,
                        "price": current_price,
                        "deviation": 20,
                        "magic": 234000,
                        "comment": f"RL Bot Close {'Profit' if action == 8 else 'Loss'}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    if mt5.order_send(close_request):
                        closed += 1
            return f"Closed {closed} {'profitable' if action == 8 else 'losing'} positions"
    
    def start_trading(self):
        if not self.connect():
            return
            
        print("🚀 Starting live trading...")
        env = AdvancedTradingEnv()
        obs, _ = env.reset()
        
        step = 0
        while True:
            try:
                # Get account info
                account = self.get_account_info()
                
                # Get model prediction
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Execute trade
                result = self.execute_trade(action)
                
                # Update environment (simulate)
                obs, reward, done, _, info = env.step(action)
                step += 1
                
                print(f"Step {step} | Balance: ${account.balance:.2f} | Action: {result} | Reward: {reward:.2f}")
                
                if done:
                    obs, _ = env.reset()
                    step = 0
                    print("Episode reset")
                
                time.sleep(60)  # Wait 1 minute
                
            except KeyboardInterrupt:
                print("\n✅ Trading stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)
        
        mt5.shutdown()

if __name__ == "__main__":
    # You'll provide these
    LOGIN = input("Enter MT5 Login: ")
    PASSWORD = input("Enter MT5 Password: ")
    SERVER = input("Enter MT5 Server: ")
    
    trader = MT5LiveTrader(LOGIN, PASSWORD, SERVER)
    trader.start_trading()
