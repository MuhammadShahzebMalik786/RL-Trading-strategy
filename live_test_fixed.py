import websocket
import json
import numpy as np
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import time

class LiveBinanceTest:
    def __init__(self):
        self.model = PPO.load("models/best_model.zip")
        self.env = AdvancedTradingEnv()
        self.obs, _ = self.env.reset()
        self.step_count = 0
        
    def on_message(self, ws, message):
        data = json.loads(message)
        price = float(data['c'])
        
        # Get action from model
        action, _ = self.model.predict(self.obs, deterministic=True)
        
        # Execute action in environment
        self.obs, reward, done, _, info = self.env.step(action)
        self.step_count += 1
        
        print(f"Step {self.step_count} | Price: ${price:.2f} | Action: {self.get_action_name(action)} | Balance: ${self.env.balance:.2f} | Reward: {reward:.2f}")
        
        if done:
            print("Episode finished, resetting...")
            self.obs, _ = self.env.reset()
            self.step_count = 0
    
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
    print("🚀 Starting FIXED live test...")
    tester = LiveBinanceTest()
    try:
        tester.start_stream()
    except KeyboardInterrupt:
        print("\n✅ Test stopped")
