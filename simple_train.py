from stable_baselines3 import PPO
from advanced_trading_env import AdvancedTradingEnv

# Simple PPO-only training since it's already performing excellently
print("🧠 Training PPO model...")

env = AdvancedTradingEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

# Train for more steps since it's performing so well
model.learn(total_timesteps=200000)
model.save("models/best_model")

print("✅ Training completed! Model saved as 'models/best_model'")
print("Run 'python advanced_demo.py' to test the model")
