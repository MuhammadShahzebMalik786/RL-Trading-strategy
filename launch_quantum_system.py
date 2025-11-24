#!/usr/bin/env python3
"""
🚀 Quantum Trading System Launcher V2.0
Unified interface for all advanced trading systems
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import json

class QuantumLauncher:
    """Unified launcher for quantum trading systems"""
    
    def __init__(self):
        self.systems = {
            'quantum': {
                'file': 'quantum_trader_v2.py',
                'description': 'Quantum-inspired RL trading with 2048 features',
                'requirements': ['stable-baselines3', 'torch', 'talib', 'gymnasium']
            },
            'neural': {
                'file': 'neural_architect_v2.py', 
                'description': 'Evolutionary neural architecture search',
                'requirements': ['torch', 'sklearn', 'pandas', 'numpy']
            },
            'advanced': {
                'file': 'advanced_train.py',
                'description': 'Original advanced RL system',
                'requirements': ['stable-baselines3', 'optuna', 'talib']
            },
            'demo': {
                'file': 'advanced_demo.py',
                'description': 'Demo existing trained models',
                'requirements': ['stable-baselines3', 'matplotlib']
            },
            'mt5': {
                'file': 'mt5_realistic.py',
                'description': 'MT5 live trading system',
                'requirements': ['MetaTrader5', 'stable-baselines3']
            }
        }
    
    def show_menu(self):
        """Display main menu"""
        print("🌟" + "=" * 60 + "🌟")
        print("🚀 QUANTUM TRADING SYSTEM V2.0 🚀")
        print("🌟" + "=" * 60 + "🌟")
        print()
        
        print("📋 Available Systems:")
        for key, system in self.systems.items():
            status = "✅" if os.path.exists(system['file']) else "❌"
            print(f"  {key:10} {status} {system['description']}")
        
        print()
        print("🎯 Quick Actions:")
        print("  setup      📦 Install all requirements")
        print("  status     📊 Check system status")
        print("  clean      🧹 Clean temporary files")
        print("  benchmark  ⚡ Run performance benchmark")
        print("  quit       🚪 Exit launcher")
        print()
    
    def install_requirements(self):
        """Install all system requirements"""
        print("📦 Installing requirements...")
        
        all_requirements = set()
        for system in self.systems.values():
            all_requirements.update(system['requirements'])
        
        # Create comprehensive requirements file
        requirements_content = """
# Quantum Trading System V2.0 Requirements
stable-baselines3>=2.0.0
torch>=2.0.0
gymnasium>=0.28.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
optuna>=3.0.0
TA-Lib>=0.4.25
MetaTrader5>=5.0.37
plotly>=5.0.0
dash>=2.0.0
yfinance>=0.2.0
scipy>=1.7.0
seaborn>=0.11.0
jupyter>=1.0.0
tensorboard>=2.8.0
""".strip()
        
        with open('requirements_quantum.txt', 'w') as f:
            f.write(requirements_content)
        
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_quantum.txt'], 
                         check=True)
            print("✅ Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
    
    def check_status(self):
        """Check system status"""
        print("📊 System Status Check")
        print("-" * 40)
        
        # Check files
        print("📁 File Status:")
        for key, system in self.systems.items():
            exists = os.path.exists(system['file'])
            status = "✅ Found" if exists else "❌ Missing"
            print(f"  {key:10} {status:10} {system['file']}")
        
        print()
        
        # Check directories
        print("📂 Directory Status:")
        dirs = ['models', 'logs', 'plots', 'tensorboard_logs']
        for dir_name in dirs:
            exists = os.path.exists(dir_name)
            status = "✅ Found" if exists else "❌ Missing"
            print(f"  {dir_name:15} {status}")
        
        print()
        
        # Check data files
        print("📈 Data Files:")
        data_files = ['eth_data.csv']
        for file_name in data_files:
            exists = os.path.exists(file_name)
            size = os.path.getsize(file_name) / (1024*1024) if exists else 0
            status = f"✅ {size:.1f}MB" if exists else "❌ Missing"
            print(f"  {file_name:15} {status}")
        
        print()
        
        # Check Python packages
        print("🐍 Package Status:")
        packages = ['stable_baselines3', 'torch', 'gymnasium', 'talib', 'optuna']
        for package in packages:
            try:
                __import__(package)
                print(f"  {package:20} ✅ Installed")
            except ImportError:
                print(f"  {package:20} ❌ Missing")
    
    def clean_system(self):
        """Clean temporary files"""
        print("🧹 Cleaning system...")
        
        # Files to clean
        clean_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '.pytest_cache',
            'tensorboard_logs/PPO_*',
            'logs/*.log',
            'temp_*'
        ]
        
        import glob
        import shutil
        
        cleaned_count = 0
        
        for pattern in clean_patterns:
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                try:
                    if os.path.isdir(match):
                        shutil.rmtree(match)
                    else:
                        os.remove(match)
                    cleaned_count += 1
                    print(f"  Removed: {match}")
                except Exception as e:
                    print(f"  Error removing {match}: {e}")
        
        print(f"✅ Cleaned {cleaned_count} items")
    
    def run_benchmark(self):
        """Run performance benchmark"""
        print("⚡ Running Performance Benchmark...")
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            },
            'tests': {}
        }
        
        # Test 1: Import speed
        print("  🔍 Testing import speeds...")
        import time
        
        packages = ['numpy', 'pandas', 'torch', 'stable_baselines3']
        for package in packages:
            start_time = time.time()
            try:
                __import__(package)
                import_time = time.time() - start_time
                benchmark_results['tests'][f'{package}_import'] = import_time
                print(f"    {package:20} {import_time:.3f}s")
            except ImportError:
                print(f"    {package:20} ❌ Not available")
        
        # Test 2: Data processing speed
        print("  📊 Testing data processing...")
        if os.path.exists('eth_data.csv'):
            start_time = time.time()
            import pandas as pd
            data = pd.read_csv('eth_data.csv')
            data['sma'] = data['close'].rolling(20).mean()
            processing_time = time.time() - start_time
            benchmark_results['tests']['data_processing'] = processing_time
            print(f"    Data processing:     {processing_time:.3f}s")
        
        # Test 3: Model creation speed
        print("  🧠 Testing model creation...")
        try:
            start_time = time.time()
            from stable_baselines3 import PPO
            import gymnasium as gym
            
            env = gym.make('CartPole-v1')
            model = PPO('MlpPolicy', env, verbose=0)
            model_time = time.time() - start_time
            benchmark_results['tests']['model_creation'] = model_time
            print(f"    Model creation:      {model_time:.3f}s")
        except Exception as e:
            print(f"    Model creation:      ❌ Error: {e}")
        
        # Save benchmark results
        with open('benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print("✅ Benchmark complete! Results saved to benchmark_results.json")
    
    def launch_system(self, system_key):
        """Launch specific system"""
        if system_key not in self.systems:
            print(f"❌ Unknown system: {system_key}")
            return
        
        system = self.systems[system_key]
        
        if not os.path.exists(system['file']):
            print(f"❌ System file not found: {system['file']}")
            return
        
        print(f"🚀 Launching {system['description']}...")
        print(f"📁 File: {system['file']}")
        print("-" * 50)
        
        try:
            subprocess.run([sys.executable, system['file']], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error launching system: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ System stopped by user")
    
    def run(self):
        """Main launcher loop"""
        while True:
            self.show_menu()
            
            try:
                choice = input("🎯 Enter your choice: ").strip().lower()
                
                if choice == 'quit' or choice == 'q':
                    print("👋 Goodbye!")
                    break
                elif choice == 'setup':
                    self.install_requirements()
                elif choice == 'status':
                    self.check_status()
                elif choice == 'clean':
                    self.clean_system()
                elif choice == 'benchmark':
                    self.run_benchmark()
                elif choice in self.systems:
                    self.launch_system(choice)
                else:
                    print(f"❌ Unknown choice: {choice}")
                
                input("\n⏸️ Press Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("\n⏸️ Press Enter to continue...")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Quantum Trading System Launcher V2.0')
    parser.add_argument('--system', '-s', choices=['quantum', 'neural', 'advanced', 'demo', 'mt5'],
                       help='Launch specific system directly')
    parser.add_argument('--setup', action='store_true', help='Install requirements and exit')
    parser.add_argument('--status', action='store_true', help='Check status and exit')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark and exit')
    
    args = parser.parse_args()
    
    launcher = QuantumLauncher()
    
    if args.setup:
        launcher.install_requirements()
    elif args.status:
        launcher.check_status()
    elif args.benchmark:
        launcher.run_benchmark()
    elif args.system:
        launcher.launch_system(args.system)
    else:
        launcher.run()


if __name__ == "__main__":
    main()
