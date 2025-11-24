import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import MetaTrader5 as mt5

class TradingDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("🚀 Advanced AI ETH Trader Dashboard", 
                   style={'textAlign': 'center', 'color': '#2E86AB'}),
            
            # Real-time metrics
            html.Div([
                html.Div([
                    html.H3("Account Metrics", style={'color': '#A23B72'}),
                    html.Div(id='account-metrics')
                ], className='six columns'),
                
                html.Div([
                    html.H3("Performance Stats", style={'color': '#F18F01'}),
                    html.Div(id='performance-stats')
                ], className='six columns'),
            ], className='row'),
            
            # Charts
            html.Div([
                dcc.Graph(id='equity-curve'),
                dcc.Graph(id='price-signals'),
            ]),
            
            # Trade history
            html.Div([
                html.H3("Recent Trades", style={'color': '#C73E1D'}),
                html.Div(id='trade-history')
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
    
    def get_account_info(self):
        """Get current account information"""
        if not mt5.initialize():
            return None
        
        account = mt5.account_info()
        positions = mt5.positions_get()
        
        return {
            'balance': account.balance if account else 0,
            'equity': account.equity if account else 0,
            'margin': account.margin if account else 0,
            'free_margin': account.margin_free if account else 0,
            'positions': len(positions) if positions else 0,
            'total_pnl': sum(pos.profit for pos in positions) if positions else 0
        }
    
    def get_trade_history(self):
        """Get recent trade history"""
        if not mt5.initialize():
            return pd.DataFrame()
        
        # Get deals from last 7 days
        from_date = datetime.now() - timedelta(days=7)
        deals = mt5.history_deals_get(from_date, datetime.now())
        
        if deals:
            df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df.tail(20)  # Last 20 trades
        
        return pd.DataFrame()
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('account-metrics', 'children'),
             Output('performance-stats', 'children'),
             Output('equity-curve', 'figure'),
             Output('price-signals', 'figure'),
             Output('trade-history', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Get account info
            account_info = self.get_account_info()
            
            if account_info:
                # Account metrics
                account_metrics = html.Div([
                    html.P(f"💰 Balance: ${account_info['balance']:.2f}"),
                    html.P(f"📊 Equity: ${account_info['equity']:.2f}"),
                    html.P(f"🔒 Margin Used: ${account_info['margin']:.2f}"),
                    html.P(f"🆓 Free Margin: ${account_info['free_margin']:.2f}"),
                    html.P(f"📈 Open Positions: {account_info['positions']}"),
                    html.P(f"💵 Total PnL: ${account_info['total_pnl']:.2f}")
                ])
                
                # Performance stats
                margin_usage = (account_info['margin'] / account_info['balance'] * 100) if account_info['balance'] > 0 else 0
                daily_return = ((account_info['equity'] - 10) / 10 * 100) if account_info['equity'] > 0 else 0
                
                performance_stats = html.Div([
                    html.P(f"📊 Margin Usage: {margin_usage:.1f}%"),
                    html.P(f"📈 Total Return: {daily_return:.2f}%"),
                    html.P(f"🎯 Target: >1% daily"),
                    html.P(f"⚡ Status: {'🟢 ACTIVE' if account_info['positions'] > 0 else '🟡 WAITING'}")
                ])
            else:
                account_metrics = html.P("❌ MT5 not connected")
                performance_stats = html.P("❌ No data available")
            
            # Create equity curve (placeholder)
            equity_fig = go.Figure()
            equity_fig.add_trace(go.Scatter(
                x=[datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)],
                y=[10 + i * 0.1 for i in range(24)],  # Simulated growth
                mode='lines',
                name='Equity Curve',
                line=dict(color='#2E86AB', width=3)
            ))
            equity_fig.update_layout(
                title="💰 Equity Curve",
                xaxis_title="Time",
                yaxis_title="Balance ($)",
                template="plotly_dark"
            )
            
            # Create price signals chart (placeholder)
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=[datetime.now() - timedelta(minutes=i*5) for i in range(100, 0, -1)],
                y=[3900 + np.sin(i/10) * 50 for i in range(100)],  # Simulated ETH price
                mode='lines',
                name='ETH Price',
                line=dict(color='#F18F01', width=2)
            ))
            price_fig.update_layout(
                title="📊 ETH Price & AI Signals",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                template="plotly_dark"
            )
            
            # Trade history
            trades_df = self.get_trade_history()
            if not trades_df.empty:
                trade_history = html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Time"),
                            html.Th("Type"),
                            html.Th("Volume"),
                            html.Th("Price"),
                            html.Th("PnL")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td(row['time'].strftime('%H:%M:%S')),
                            html.Td("BUY" if row['type'] == 0 else "SELL"),
                            html.Td(f"{row['volume']:.2f}"),
                            html.Td(f"${row['price']:.2f}"),
                            html.Td(f"${row['profit']:.2f}")
                        ]) for _, row in trades_df.iterrows()
                    ])
                ])
            else:
                trade_history = html.P("No recent trades")
            
            return account_metrics, performance_stats, equity_fig, price_fig, trade_history
    
    def run(self, host='127.0.0.1', port=8050):
        """Start the dashboard server"""
        print(f"🌐 Dashboard running at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=False)

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
