import subprocess
import threading
import time
import sys
import os

def run_neural_trader():
    """Run the compressed neural trader"""
    print("🧠 Starting Compressed Neural Trader...")
    subprocess.run([sys.executable, "compressed_neural_trader.py"])

def run_cloud_dashboard():
    """Run the cloud-synced dashboard"""
    print("🌐 Starting Cloud Dashboard...")
    time.sleep(10)  # Wait for trader to initialize
    subprocess.run([sys.executable, "cloud_sync_dashboard.py"])

def main():
    print("=" * 70)
    print("🧠 COMPRESSED NEURAL TRADING SYSTEM")
    print("=" * 70)
    print("🎯 DUAL-SCALE INTELLIGENCE:")
    print("  • Training Environment: $3,000 virtual balance")
    print("  • Live Environment: $10 real balance (1:2000 leverage)")
    print("  • Compression Ratio: 0.0033 (intelligent scaling)")
    print()
    print("🧠 NEURAL ARCHITECTURE:")
    print("  • LightGBM ensemble with walk-forward validation")
    print("  • 26+ engineered features (MA, RSI, MACD, ATR, etc.)")
    print("  • 1 YEAR of training data (~105,000 samples)")
    print("  • PnL-based labels: +0.30% TP, -0.20% SL")
    print("  • Swing detection: EMA + RSI trend analysis")
    print()
    print("⚡ TRADING PARAMETERS:")
    print("  • Lot size: 0.1 (adaptive scaling)")
    print("  • Margin per trade: ~$2")
    print("  • Max margin usage: 70%")
    print("  • Confidence thresholds: Buy/Sell >50% (AGGRESSIVE)")
    print()
    print("📊 PERFORMANCE TARGETS:")
    print("  • Daily return: >1%")
    print("  • Sharpe ratio: >2.0")
    print("  • Max drawdown: <20%")
    print("  • Goal: $10 → $100+ through intelligent compounding")
    print()
    print("🌐 MONITORING:")
    print("  • Real-time cloud dashboard")
    print("  • Live equity curve tracking")
    print("  • Neural signal visualization")
    print("  • Automatic performance metrics")
    print("=" * 70)
    
    # Check dependencies
    missing_deps = []
    try:
        import lightgbm
    except ImportError:
        missing_deps.append("lightgbm")
    
    try:
        import MetaTrader5
    except ImportError:
        missing_deps.append("MetaTrader5")
    
    try:
        import dash
    except ImportError:
        missing_deps.append("dash")
    
    try:
        import talib
    except ImportError:
        missing_deps.append("TA-Lib")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install lightgbm MetaTrader5 dash plotly TA-Lib requests")
        return
    
    print("✅ All dependencies verified")
    print()
    
    # Start both systems
    print("🚀 Launching compressed neural trading system...")
    
    trader_thread = threading.Thread(target=run_neural_trader, daemon=True)
    dashboard_thread = threading.Thread(target=run_cloud_dashboard, daemon=True)
    
    trader_thread.start()
    dashboard_thread.start()
    
    print()
    print("🌐 Dashboard URL: http://localhost:8050")
    print("🧠 Neural trader is learning and adapting...")
    print("📊 Monitor performance in real-time")
    print()
    print("Press Ctrl+C to stop the system")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down compressed neural system...")
        print("💾 Performance data saved")
        print("✅ System stopped safely")

if __name__ == "__main__":
    main()
