import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import requests
import threading

class CloudSyncDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        self.performance_data = {
            'timestamps': [],
            'equity': [],
            'balance': [],
            'trades': [],
            'signals': []
        }
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("🧠 Compressed Neural Trader Dashboard", 
                   style={'textAlign': 'center', 'color': '#1f77b4'}),
            
            # Key metrics row
            html.Div([
                html.Div([
                    html.H3("💰 Account Status"),
                    html.Div(id='account-status')
                ], className='four columns'),
                
                html.Div([
                    html.H3("🧠 Neural Intelligence"),
                    html.Div(id='neural-status')
                ], className='four columns'),
                
                html.Div([
                    html.H3("📊 Performance"),
                    html.Div(id='performance-metrics')
                ], className='four columns'),
            ], className='row'),
            
            # Charts
            html.Div([
                dcc.Graph(id='equity-curve-chart'),
            ]),
            
            html.Div([
                dcc.Graph(id='price-signals-chart'),
            ]),
            
            # Trade log
            html.Div([
                html.H3("📋 Recent Neural Trades"),
                html.Div(id='trade-log', style={'height': '300px', 'overflow': 'auto'})
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=3*1000,  # 3 seconds
                n_intervals=0
            )
        ])
    
    def get_mt5_data(self):
        """Get real-time MT5 data"""
        try:
            if not mt5.initialize():
                return None
            
            account = mt5.account_info()
            positions = mt5.positions_get()
            
            # Get recent deals
            from_date = datetime.now() - timedelta(hours=24)
            deals = mt5.history_deals_get(from_date, datetime.now())
            
            return {
                'account': account,
                'positions': positions,
                'deals': deals
            }
        except:
            return None
    
    def calculate_performance_metrics(self, equity_history):
        """Calculate key performance metrics"""
        if len(equity_history) < 2:
            return {'daily_return': 0, 'sharpe': 0, 'max_dd': 0}
        
        returns = pd.Series(equity_history).pct_change().dropna()
        
        # Daily return (annualized)
        daily_return = returns.mean() * 288 * 100  # 288 5-min periods per day
        
        # Sharpe ratio
        sharpe = (returns.mean() / returns.std() * np.sqrt(288)) if returns.std() > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series(equity_history)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_dd = abs(drawdown.min()) * 100
        
        return {
            'daily_return': daily_return,
            'sharpe': sharpe,
            'max_dd': max_dd
        }
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('account-status', 'children'),
             Output('neural-status', 'children'),
             Output('performance-metrics', 'children'),
             Output('equity-curve-chart', 'figure'),
             Output('price-signals-chart', 'figure'),
             Output('trade-log', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Get MT5 data
            mt5_data = self.get_mt5_data()
            
            if mt5_data and mt5_data['account']:
                account = mt5_data['account']
                positions = mt5_data['positions'] or []
                
                # Update performance tracking
                current_time = datetime.now()
                self.performance_data['timestamps'].append(current_time)
                self.performance_data['equity'].append(account.equity)
                self.performance_data['balance'].append(account.balance)
                
                # Keep last 1000 points
                if len(self.performance_data['timestamps']) > 1000:
                    for key in self.performance_data:
                        self.performance_data[key] = self.performance_data[key][-1000:]
                
                # Account status
                margin_usage = (account.margin / account.balance * 100) if account.balance > 0 else 0
                total_pnl = sum(pos.profit for pos in positions)
                
                account_status = html.Div([
                    html.P(f"💰 Balance: ${account.balance:.2f}"),
                    html.P(f"📊 Equity: ${account.equity:.2f}"),
                    html.P(f"🔒 Margin: {margin_usage:.1f}%"),
                    html.P(f"📈 Open PnL: ${total_pnl:.2f}"),
                    html.P(f"🎯 Positions: {len(positions)}")
                ])
                
                # Neural status
                compression_ratio = 10 / 3000
                neural_status = html.Div([
                    html.P(f"🧠 Model: LightGBM Ensemble"),
                    html.P(f"🔄 Compression: {compression_ratio:.4f}"),
                    html.P(f"📊 Training: $3000 → $10"),
                    html.P(f"⚡ Status: {'🟢 ACTIVE' if len(positions) > 0 else '🟡 SCANNING'}"),
                    html.P(f"🎯 Confidence: >60%")
                ])
                
                # Performance metrics
                perf_metrics = self.calculate_performance_metrics(self.performance_data['equity'])
                
                performance_display = html.Div([
                    html.P(f"📈 Daily Return: {perf_metrics['daily_return']:.2f}%"),
                    html.P(f"📊 Sharpe Ratio: {perf_metrics['sharpe']:.2f}"),
                    html.P(f"📉 Max Drawdown: {perf_metrics['max_dd']:.2f}%"),
                    html.P(f"🎯 Target: >1% daily"),
                    html.P(f"💎 Goal: $10 → $100")
                ])
                
            else:
                account_status = html.P("❌ MT5 not connected")
                neural_status = html.P("❌ Neural system offline")
                performance_display = html.P("❌ No performance data")
            
            # Equity curve
            if len(self.performance_data['timestamps']) > 1:
                equity_fig = go.Figure()
                equity_fig.add_trace(go.Scatter(
                    x=self.performance_data['timestamps'],
                    y=self.performance_data['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='#1f77b4', width=3)
                ))
                equity_fig.add_trace(go.Scatter(
                    x=self.performance_data['timestamps'],
                    y=self.performance_data['balance'],
                    mode='lines',
                    name='Balance',
                    line=dict(color='#ff7f0e', width=2)
                ))
                equity_fig.update_layout(
                    title="💰 Compressed Neural Equity Curve",
                    xaxis_title="Time",
                    yaxis_title="Value ($)",
                    template="plotly_dark",
                    height=400
                )
            else:
                equity_fig = go.Figure()
                equity_fig.update_layout(title="💰 Equity Curve (Loading...)", template="plotly_dark")
            
            # Price signals chart
            try:
                # Get recent ETH price data
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': 'ETHUSDT',
                    'interval': '5m',
                    'limit': 100
                }
                response = requests.get(url, params=params, timeout=5)
                price_data = response.json()
                
                timestamps = [datetime.fromtimestamp(candle[0]/1000) for candle in price_data]
                prices = [float(candle[4]) for candle in price_data]  # Close prices
                
                price_fig = go.Figure()
                price_fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode='lines',
                    name='ETH/USDT',
                    line=dict(color='#2ca02c', width=2)
                ))
                
                # Add trade markers if available
                if mt5_data and mt5_data['deals']:
                    deal_times = []
                    deal_prices = []
                    deal_colors = []
                    
                    for deal in mt5_data['deals'][-20:]:  # Last 20 deals
                        deal_times.append(datetime.fromtimestamp(deal.time))
                        deal_prices.append(deal.price)
                        deal_colors.append('green' if deal.type == 0 else 'red')
                    
                    if deal_times:
                        price_fig.add_trace(go.Scatter(
                            x=deal_times,
                            y=deal_prices,
                            mode='markers',
                            name='Neural Trades',
                            marker=dict(size=10, color=deal_colors)
                        ))
                
                price_fig.update_layout(
                    title="📊 ETH Price & Neural Signals",
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    template="plotly_dark",
                    height=400
                )
                
            except:
                price_fig = go.Figure()
                price_fig.update_layout(title="📊 Price Chart (Loading...)", template="plotly_dark")
            
            # Trade log
            if mt5_data and mt5_data['deals']:
                recent_deals = list(mt5_data['deals'])[-10:]  # Last 10 trades
                trade_log = html.Div([
                    html.Div([
                        html.Span(f"{datetime.fromtimestamp(deal.time).strftime('%H:%M:%S')} ", 
                                style={'color': '#888'}),
                        html.Span(f"{'BUY' if deal.type == 0 else 'SELL'} ", 
                                style={'color': 'green' if deal.type == 0 else 'red', 'fontWeight': 'bold'}),
                        html.Span(f"{deal.volume} lots @ ${deal.price:.2f} "),
                        html.Span(f"PnL: ${deal.profit:.2f}", 
                                style={'color': 'green' if deal.profit > 0 else 'red'})
                    ], style={'marginBottom': '5px'}) for deal in reversed(recent_deals)
                ])
            else:
                trade_log = html.P("No recent trades")
            
            return account_status, neural_status, performance_display, equity_fig, price_fig, trade_log
    
    def run(self, host='0.0.0.0', port=8050):
        """Start the cloud-synced dashboard"""
        print(f"🌐 Cloud Dashboard: http://{host}:{port}")
        print("📊 Real-time neural trading monitoring")
        self.app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    dashboard = CloudSyncDashboard()
    dashboard.run()
