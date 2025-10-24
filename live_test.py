import websocket
import json
import numpy as np
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import threading
import time

class LiveBinanceTest:
    def __init__(self):
        self.model = PPO.load("models/best_model.zip")
        self.env = AdvancedTradingEnv()
        self.price_data = []
        self.running = True
        
    def on_message(self, ws, message):
        data = json.loads(message)
        price = float(data['c'])  # Close price
        self.price_data.append(price)
        
        if len(self.price_data) >= 50:  # Need lookback window
            # Update environment with real price
            self.env.current_price = price
            obs = self.env._get_observation()
            
            # Get model prediction
            action, _ = self.model.predict(obs, deterministic=True)
            
            print(f"Price: ${price:.2f} | Action: {self.get_action_name(action)} | Balance: ${self.env.balance:.2f}")
            
            # Keep only last 1000 prices
            if len(self.price_data) > 1000:
                self.price_data = self.price_data[-1000:]
    
    def get_action_name(self, action):
        actions = ["Hold", "Buy Small", "Buy Med", "Buy Large", 
                  "Sell Small", "Sell Med", "Sell Large", 
                  "Close All", "Close Profit", "Close Loss"]
        return actions[action]
    
    def on_error(self, ws, error):
        print(f"Error: {error}")
    
    def start_stream(self):
        ws = websocket.WebSocketApp(
            "wss://stream.binance.com:9443/ws/ethusdt@ticker",
            on_message=self.on_message,
            on_error=self.on_error
        )
        ws.run_forever()

if __name__ == "__main__":
    print("🚀 Starting live Binance ETH/USDT test...")
    print("Press Ctrl+C to stop")
    
    tester = LiveBinanceTest()
    try:
        tester.start_stream()
    except KeyboardInterrupt:
        print("\n✅ Test stopped")
