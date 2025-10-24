import MetaTrader5 as mt5
import numpy as np
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import time
import json

class LiveLearningTrader:
    def __init__(self):
        self.symbol = "ETHUSDm"
        self.lot_size = 0.1
        self.virtual_balance = 10.0  # Virtual $10 chunks
        self.chunk_number = 1
        self.max_chunks = 300  # 3000/10 = 300 chunks
        self.starting_real_balance = None
        
        # Create fresh environment and model
        self.env = AdvancedTradingEnv(initial_balance=10.0)
        self.model = PPO("MlpPolicy", self.env, verbose=1, learning_rate=0.0003)
        self.obs, _ = self.env.reset()
        
        # Track performance
        self.chunk_results = []
        
    def execute_real_trade(self, action):
        """Execute real trade on MT5"""
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return False, "No price"
            
        if action == 0:  # Hold
            return True, "Hold"
            
        elif action in [1, 2, 3]:  # Buy
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Live Learn Chunk {self.chunk_number}",
            }
            result = mt5.order_send(request)
            success = result and result.retcode == 10009
            return success, f"Buy - {'Success' if success else 'Failed'}"
            
        elif action in [4, 5, 6]:  # Sell
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Live Learn Chunk {self.chunk_number}",
            }
            result = mt5.order_send(request)
            success = result and result.retcode == 10009
            return success, f"Sell - {'Success' if success else 'Failed'}"
            
        elif action == 7:  # Close All
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                return True, "No positions"
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
                }
                result = mt5.order_send(close_request)
                if result and result.retcode == 10009:
                    closed += 1
            return True, f"Closed {closed}"
        
        return True, "Hold"
    
    def calculate_real_pnl(self):
        """Calculate real PnL from MT5"""
        account = mt5.account_info()
        if not account or not self.starting_real_balance:
            return 0
        return account.balance - self.starting_real_balance
    
    def start_chunk(self):
        """Start new $10 chunk"""
        print(f"\n🚀 Starting Chunk {self.chunk_number}/300")
        self.virtual_balance = 10.0
        self.obs, _ = self.env.reset()
        
        # Record starting balance
        account = mt5.account_info()
        if account:
            self.starting_real_balance = account.balance
    
    def end_chunk(self, reason):
        """End current chunk and save results"""
        real_pnl = self.calculate_real_pnl()
        virtual_pnl = self.env.balance - 10.0
        
        result = {
            'chunk': self.chunk_number,
            'reason': reason,
            'virtual_pnl': virtual_pnl,
            'real_pnl': real_pnl,
            'virtual_balance': self.env.balance,
            'real_balance': mt5.account_info().balance if mt5.account_info() else 0
        }
        
        self.chunk_results.append(result)
        
        print(f"📊 Chunk {self.chunk_number} Complete:")
        print(f"   Reason: {reason}")
        print(f"   Virtual PnL: ${virtual_pnl:.2f}")
        print(f"   Real PnL: ${real_pnl:.2f}")
        print(f"   Real Balance: ${result['real_balance']:.2f}")
        
        # Save results
        with open('live_learning_results.json', 'w') as f:
            json.dump(self.chunk_results, f, indent=2)
        
        # Save model after each chunk
        self.model.save(f"models/live_model_chunk_{self.chunk_number}")
        
        self.chunk_number += 1
        return self.chunk_number <= self.max_chunks
    
    def start_live_learning(self):
        if not mt5.initialize():
            print("❌ MT5 not available")
            return
            
        print("🧠 Starting Live Learning on MT5")
        print(f"Symbol: {self.symbol}")
        print(f"Lot Size: {self.lot_size}")
        print(f"Virtual Chunks: $10 each")
        
        self.start_chunk()
        step = 0
        
        while self.chunk_number <= self.max_chunks:
            try:
                # Get action from model
                action, _ = self.model.predict(self.obs, deterministic=False)
                
                # Execute real trade
                success, trade_result = self.execute_real_trade(action)
                
                # Update virtual environment
                self.obs, reward, done, _, info = self.env.step(action)
                
                # Learn from experience
                if step % 10 == 0:  # Learn every 10 steps
                    self.model.learn(total_timesteps=1, reset_num_timesteps=False)
                
                # Get current prices and balances
                tick = mt5.symbol_info_tick(self.symbol)
                account = mt5.account_info()
                real_pnl = self.calculate_real_pnl()
                
                print(f"Chunk {self.chunk_number} Step {step} | Price: ${tick.bid:.2f} | Virtual: ${self.env.balance:.2f} | Real PnL: ${real_pnl:.2f} | {trade_result}")
                
                # Check chunk end conditions
                chunk_ended = False
                
                if self.env.balance <= 0:  # Virtual balance wiped
                    chunk_ended = self.end_chunk("Virtual balance wiped")
                elif real_pnl <= -10:  # Real loss of $10
                    chunk_ended = self.end_chunk("Real loss $10")
                elif self.env.balance >= 20:  # Virtual profit target
                    chunk_ended = self.end_chunk("Virtual profit target")
                elif real_pnl >= 10:  # Real profit target
                    chunk_ended = self.end_chunk("Real profit $10")
                elif done:  # Episode finished
                    chunk_ended = self.end_chunk("Episode complete")
                
                if chunk_ended and self.chunk_number <= self.max_chunks:
                    self.start_chunk()
                    step = 0
                else:
                    step += 1
                
                time.sleep(30)  # 30 second intervals
                
            except KeyboardInterrupt:
                print(f"\n✅ Learning stopped at chunk {self.chunk_number}")
                self.end_chunk("Manual stop")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(10)
        
        print(f"\n🎯 Live Learning Complete!")
        print(f"Completed {len(self.chunk_results)} chunks")
        print(f"Results saved to live_learning_results.json")

if __name__ == "__main__":
    trader = LiveLearningTrader()
    trader.start_live_learning()
