import MetaTrader5 as mt5
import numpy as np
from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv
import time
import json
from datetime import datetime

class LiveLearningTrader:
    def __init__(self):
        self.symbol = "ETHUSDm"
        self.lot_size = 0.1
        self.virtual_balance = 10.0
        self.chunk_number = 1
        self.max_chunks = 300
        self.starting_real_balance = None
        self.chunk_start_time = None
        
        # Create fresh environment and model
        self.env = AdvancedTradingEnv(initial_balance=10.0)
        self.model = PPO("MlpPolicy", self.env, verbose=0, learning_rate=0.0003)
        self.obs, _ = self.env.reset()
        
        # Track performance
        self.chunk_results = []
        self.total_trades = 0
        self.winning_trades = 0
        
    def get_action_name(self, action):
        actions = ["Hold", "Buy Small", "Buy Med", "Buy Large", 
                  "Sell Small", "Sell Med", "Sell Large", 
                  "Close All", "Close Profit", "Close Loss"]
        return actions[action]
    
    def get_positions_info(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return "No positions"
        
        total_profit = sum(pos.profit for pos in positions)
        buy_count = sum(1 for pos in positions if pos.type == 0)
        sell_count = sum(1 for pos in positions if pos.type == 1)
        
        return f"{len(positions)} positions (Buy:{buy_count}, Sell:{sell_count}) | PnL: ${total_profit:.2f}"
    
    def execute_real_trade(self, action):
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return False, "❌ No price data"
            
        if action == 0:  # Hold
            return True, "⏸️ Hold"
            
        elif action in [1, 2, 3]:  # Buy
            print(f"   🔄 Executing BUY order...")
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Live Learn C{self.chunk_number}",
            }
            result = mt5.order_send(request)
            success = result and result.retcode == 10009
            self.total_trades += 1
            return success, f"🟢 BUY {self.lot_size} @ ${tick.ask:.2f} - {'✅ Success' if success else '❌ Failed'}"
            
        elif action in [4, 5, 6]:  # Sell
            print(f"   🔄 Executing SELL order...")
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": tick.bid,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Live Learn C{self.chunk_number}",
            }
            result = mt5.order_send(request)
            success = result and result.retcode == 10009
            self.total_trades += 1
            return success, f"🔴 SELL {self.lot_size} @ ${tick.bid:.2f} - {'✅ Success' if success else '❌ Failed'}"
            
        elif action == 7:  # Close All
            print(f"   🔄 Closing all positions...")
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                return True, "⚪ No positions to close"
            closed = 0
            total_profit = 0
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
                    total_profit += pos.profit
                    if pos.profit > 0:
                        self.winning_trades += 1
            return True, f"🔒 Closed {closed} positions | Profit: ${total_profit:.2f}"
        
        return True, "⏸️ Hold"
    
    def calculate_real_pnl(self):
        account = mt5.account_info()
        if not account or not self.starting_real_balance:
            return 0
        return account.balance - self.starting_real_balance
    
    def print_status(self, step, action, trade_result, tick, account):
        real_pnl = self.calculate_real_pnl()
        positions_info = self.get_positions_info()
        elapsed = (datetime.now() - self.chunk_start_time).total_seconds() / 60 if self.chunk_start_time else 0
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        
        print(f"\n{'='*80}")
        print(f"🧠 LIVE LEARNING - CHUNK {self.chunk_number}/300 | STEP {step}")
        print(f"{'='*80}")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')} | Elapsed: {elapsed:.1f}m")
        print(f"💰 ETH Price: ${tick.bid:.2f} | Spread: ${tick.ask - tick.bid:.2f}")
        print(f"💳 Real Balance: ${account.balance:.2f} | Real PnL: ${real_pnl:.2f}")
        print(f"🎯 Virtual Balance: ${self.env.balance:.2f} | Target: $20 or -$10")
        print(f"📊 {positions_info}")
        print(f"🤖 Model Decision: {self.get_action_name(action)}")
        print(f"⚡ Trade Result: {trade_result}")
        print(f"📈 Win Rate: {win_rate:.1f}% ({self.winning_trades}/{self.total_trades})")
        print(f"{'='*80}")
    
    def start_chunk(self):
        print(f"\n🚀 STARTING NEW CHUNK {self.chunk_number}/300")
        print(f"💰 Virtual Balance Reset: $10.00")
        self.virtual_balance = 10.0
        self.obs, _ = self.env.reset()
        self.chunk_start_time = datetime.now()
        
        account = mt5.account_info()
        if account:
            self.starting_real_balance = account.balance
            print(f"📊 Real Balance: ${account.balance:.2f}")
    
    def end_chunk(self, reason):
        real_pnl = self.calculate_real_pnl()
        virtual_pnl = self.env.balance - 10.0
        elapsed = (datetime.now() - self.chunk_start_time).total_seconds() / 60
        
        result = {
            'chunk': self.chunk_number,
            'reason': reason,
            'virtual_pnl': virtual_pnl,
            'real_pnl': real_pnl,
            'virtual_balance': self.env.balance,
            'real_balance': mt5.account_info().balance if mt5.account_info() else 0,
            'duration_minutes': elapsed,
            'trades': self.total_trades,
            'winning_trades': self.winning_trades
        }
        
        self.chunk_results.append(result)
        
        print(f"\n🏁 CHUNK {self.chunk_number} COMPLETED")
        print(f"📋 Reason: {reason}")
        print(f"⏱️ Duration: {elapsed:.1f} minutes")
        print(f"💰 Virtual PnL: ${virtual_pnl:.2f}")
        print(f"💳 Real PnL: ${real_pnl:.2f}")
        print(f"📊 Trades: {self.total_trades} | Wins: {self.winning_trades}")
        
        # Save results
        with open('live_learning_results.json', 'w') as f:
            json.dump(self.chunk_results, f, indent=2)
        
        # Save model
        self.model.save(f"models/live_model_chunk_{self.chunk_number}")
        print(f"💾 Model saved: live_model_chunk_{self.chunk_number}")
        
        self.chunk_number += 1
        return self.chunk_number <= self.max_chunks
    
    def start_live_learning(self):
        if not mt5.initialize():
            print("❌ MT5 not available")
            return
            
        print("🧠 LIVE LEARNING SYSTEM STARTING")
        print(f"🎯 Symbol: {self.symbol}")
        print(f"📏 Lot Size: {self.lot_size}")
        print(f"💰 Virtual Chunks: $10 each")
        print(f"🔢 Total Chunks: {self.max_chunks}")
        
        self.start_chunk()
        step = 0
        
        while self.chunk_number <= self.max_chunks:
            try:
                # Get action from model
                print(f"\n🤖 Model thinking...")
                action, _ = self.model.predict(self.obs, deterministic=False)
                
                # Execute real trade
                success, trade_result = self.execute_real_trade(action)
                
                # Update virtual environment
                print(f"🔄 Updating virtual environment...")
                self.obs, reward, done, _, info = self.env.step(action)
                
                # Learn from experience
                if step % 5 == 0:
                    print(f"🧠 Model learning from experience...")
                    self.model.learn(total_timesteps=1, reset_num_timesteps=False)
                
                # Get current data
                tick = mt5.symbol_info_tick(self.symbol)
                account = mt5.account_info()
                
                # Print detailed status
                self.print_status(step, action, trade_result, tick, account)
                
                # Check chunk end conditions
                real_pnl = self.calculate_real_pnl()
                chunk_ended = False
                
                if self.env.balance <= 0:
                    chunk_ended = self.end_chunk("💀 Virtual balance wiped")
                elif real_pnl <= -10:
                    chunk_ended = self.end_chunk("📉 Real loss $10")
                elif self.env.balance >= 20:
                    chunk_ended = self.end_chunk("🎯 Virtual profit target")
                elif real_pnl >= 10:
                    chunk_ended = self.end_chunk("💰 Real profit $10")
                elif done:
                    chunk_ended = self.end_chunk("✅ Episode complete")
                
                if chunk_ended and self.chunk_number <= self.max_chunks:
                    time.sleep(5)  # Brief pause between chunks
                    self.start_chunk()
                    step = 0
                else:
                    step += 1
                
                print(f"\n⏳ Waiting 60 seconds for next decision...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                print(f"\n🛑 MANUAL STOP at chunk {self.chunk_number}")
                self.end_chunk("Manual stop")
                break
            except Exception as e:
                print(f"❌ ERROR: {e}")
                time.sleep(10)
        
        print(f"\n🏆 LIVE LEARNING COMPLETE!")
        print(f"📊 Completed {len(self.chunk_results)} chunks")
        print(f"💾 Results saved to live_learning_results.json")

if __name__ == "__main__":
    trader = LiveLearningTrader()
    trader.start_live_learning()
