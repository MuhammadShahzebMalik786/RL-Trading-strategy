import yfinance as yf
import pandas as pd

# Download ETH-USD data (last 2 years)
print("Downloading ETH-USD data...")
eth_data = yf.download("ETH-USD", period="2y", interval="1h", auto_adjust=False)

# Clean and prepare data
eth_data = eth_data.dropna()

# Flatten MultiIndex columns
if isinstance(eth_data.columns, pd.MultiIndex):
    eth_data.columns = eth_data.columns.droplevel(1)

# Reset index first, then convert to lowercase
eth_data = eth_data.reset_index()
eth_data.columns = [col.lower() for col in eth_data.columns]

# Save to CSV
eth_data.to_csv("eth_data.csv", index=False)
print(f"✅ Saved {len(eth_data)} rows of ETH data")
print(f"Columns: {eth_data.columns.tolist()}")
print(f"Date range: {eth_data['datetime'].min()} to {eth_data['datetime'].max()}")
print(f"Price range: ${eth_data['close'].min():.2f} - ${eth_data['close'].max():.2f}")
