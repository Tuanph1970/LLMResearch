import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SimpleStockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.data = None

    def fetch_data(self, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period)
            if len(self.data) == 0:
                raise Exception("No data found")
            print(f"✅ Fetched {len(self.data)} days of data for {self.symbol}")
            return True
        except Exception as e:
            print(f"❌ Error fetching data for {self.symbol}: {e}")
            return False

    def calculate_sma(self, period):
        """Calculate Simple Moving Average"""
        return self.data['Close'].rolling(window=period).mean()

    def calculate_ema(self, period):
        """Calculate Exponential Moving Average"""
        return self.data['Close'].ewm(span=period).mean()

    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(period)
        std = self.data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def calculate_stochastic(self, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        low_min = self.data['Low'].rolling(window=k_period).min()
        high_max = self.data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def calculate_williams_r(self, period=14):
        """Calculate Williams %R"""
        high_max = self.data['High'].rolling(window=period).max()
        low_min = self.data['Low'].rolling(window=period).min()
        williams_r = -100 * ((high_max - self.data['Close']) / (high_max - low_min))
        return williams_r

    def calculate_all_indicators(self):
        """Calculate all technical indicators"""
        if self.data is None:
            print("❌ No data available. Please fetch data first.")
            return False

        # Moving Averages
        self.data['SMA_20'] = self.calculate_sma(20)
        self.data['SMA_50'] = self.calculate_sma(50)
        self.data['SMA_200'] = self.calculate_sma(200)
        self.data['EMA_12'] = self.calculate_ema(12)
        self.data['EMA_26'] = self.calculate_ema(26)

        # RSI
        self.data['RSI'] = self.calculate_rsi()

        # MACD
        self.data['MACD'], self.data['MACD_Signal'], self.data['MACD_Hist'] = self.calculate_macd()

        # Bollinger Bands
        self.data['BB_Upper'], self.data['BB_Middle'], self.data['BB_Lower'] = self.calculate_bollinger_bands()

        # Stochastic
        self.data['Stoch_K'], self.data['Stoch_D'] = self.calculate_stochastic()

        # Williams %R
        self.data['Williams_R'] = self.calculate_williams_r()

        # Volume Moving Average
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()

        print("✅ All technical indicators calculated")
        return True

    def generate_signals(self):
        """Generate trading signals based on technical analysis"""
        if self.data is None or 'RSI' not in self.data.columns:
            print("❌ Please calculate indicators first")
            return None

        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2] if len(self.data) > 1 else latest

        signals = {
            'trend_signals': [],
            'momentum_signals': [],
            'volume_signals': [],
            'volatility_signals': [],
            'buy_signals': 0,
            'sell_signals': 0,
            'neutral_signals': 0
        }

        # Trend Analysis
        if latest['Close'] > latest['SMA_20']:
            signals['trend_signals'].append("Price above SMA20 (Bullish)")
            signals['buy_signals'] += 1
        else:
            signals['trend_signals'].append("Price below SMA20 (Bearish)")
            signals['sell_signals'] += 1

        if latest['Close'] > latest['SMA_50']:
            signals['trend_signals'].append("Price above SMA50 (Bullish)")
            signals['buy_signals'] += 1
        else:
            signals['trend_signals'].append("Price below SMA50 (Bearish)")
            signals['sell_signals'] += 1

        if latest['SMA_20'] > latest['SMA_50']:
            signals['trend_signals'].append("SMA20 > SMA50 (Bullish)")
            signals['buy_signals'] += 1
        else:
            signals['trend_signals'].append("SMA20 < SMA50 (Bearish)")
            signals['sell_signals'] += 1

        # Momentum Analysis
        if latest['RSI'] < 30:
            signals['momentum_signals'].append("RSI Oversold < 30 (Buy Signal)")
            signals['buy_signals'] += 2  # Stronger signal
        elif latest['RSI'] > 70:
            signals['momentum_signals'].append("RSI Overbought > 70 (Sell Signal)")
            signals['sell_signals'] += 2  # Stronger signal
        else:
            signals['momentum_signals'].append(f"RSI Neutral ({latest['RSI']:.1f})")
            signals['neutral_signals'] += 1

        # MACD Analysis
        if latest['MACD'] > latest['MACD_Signal']:
            signals['momentum_signals'].append("MACD above Signal (Bullish)")
            signals['buy_signals'] += 1
        else:
            signals['momentum_signals'].append("MACD below Signal (Bearish)")
            signals['sell_signals'] += 1

        if latest['MACD_Hist'] > prev['MACD_Hist']:
            signals['momentum_signals'].append("MACD Histogram Rising (Bullish)")
            signals['buy_signals'] += 1
        else:
            signals['momentum_signals'].append("MACD Histogram Falling (Bearish)")
            signals['sell_signals'] += 1

        # Stochastic Analysis
        if latest['Stoch_K'] < 20:
            signals['momentum_signals'].append("Stochastic Oversold (Buy Signal)")
            signals['buy_signals'] += 1
        elif latest['Stoch_K'] > 80:
            signals['momentum_signals'].append("Stochastic Overbought (Sell Signal)")
            signals['sell_signals'] += 1

        # Volume Analysis
        if latest['Volume'] > latest['Volume_MA'] * 1.5:
            signals['volume_signals'].append("High Volume (Strong Signal)")
            # Amplify existing signals
            if signals['buy_signals'] > signals['sell_signals']:
                signals['buy_signals'] += 1
            else:
                signals['sell_signals'] += 1
        else:
            signals['volume_signals'].append("Normal Volume")

        # Bollinger Bands Analysis
        if latest['Close'] < latest['BB_Lower']:
            signals['volatility_signals'].append("Price below Lower BB (Buy Signal)")
            signals['buy_signals'] += 1
        elif latest['Close'] > latest['BB_Upper']:
            signals['volatility_signals'].append("Price above Upper BB (Sell Signal)")
            signals['sell_signals'] += 1
        else:
            signals['volatility_signals'].append("Price within BB range")
            signals['neutral_signals'] += 1

        return signals

    def get_recommendation(self):
        """Get final buy/hold/sell recommendation"""
        signals = self.generate_signals()
        if not signals:
            return "UNKNOWN", 0, "No data available"

        buy_score = signals['buy_signals']
        sell_score = signals['sell_signals']
        neutral_score = signals['neutral_signals']

        total_signals = buy_score + sell_score + neutral_score
        confidence = max(buy_score, sell_score) / total_signals if total_signals > 0 else 0

        if buy_score > sell_score and buy_score > neutral_score:
            return "BUY", confidence, f"Buy signals: {buy_score}, Sell signals: {sell_score}"
        elif sell_score > buy_score and sell_score > neutral_score:
            return "SELL", confidence, f"Buy signals: {buy_score}, Sell signals: {sell_score}"
        else:
            return "HOLD", confidence, f"Buy signals: {buy_score}, Sell signals: {sell_score}"

    def get_detailed_report(self):
        """Generate detailed analysis report"""
        if self.data is None:
            return "❌ No data available"

        latest = self.data.iloc[-1]
        signals = self.generate_signals()
        recommendation, confidence, reasoning = self.get_recommendation()

        # Calculate price changes
        price_change_1d = ((latest['Close'] - self.data['Close'].iloc[-2]) / self.data['Close'].iloc[-2] * 100) if len(
            self.data) > 1 else 0
        price_change_5d = ((latest['Close'] - self.data['Close'].iloc[-6]) / self.data['Close'].iloc[-6] * 100) if len(
            self.data) > 5 else 0
        price_change_20d = (
                    (latest['Close'] - self.data['Close'].iloc[-21]) / self.data['Close'].iloc[-21] * 100) if len(
            self.data) > 20 else 0

        report = f"""
🏢 STOCK ANALYSIS REPORT: {self.symbol}
{'=' * 60}

📊 PRICE INFORMATION:
• Current Price: ${latest['Close']:.2f}
• 1-Day Change: {price_change_1d:+.2f}%
• 5-Day Change: {price_change_5d:+.2f}%
• 20-Day Change: {price_change_20d:+.2f}%

🎯 RECOMMENDATION: {recommendation}
• Confidence Level: {confidence:.1%}
• Reasoning: {reasoning}

📈 TECHNICAL INDICATORS:
• RSI (14): {latest['RSI']:.1f} {'[OVERSOLD]' if latest['RSI'] < 30 else '[OVERBOUGHT]' if latest['RSI'] > 70 else '[NEUTRAL]'}
• MACD: {latest['MACD']:.4f} {'▲' if latest['MACD'] > latest['MACD_Signal'] else '▼'}
• Stochastic %K: {latest['Stoch_K']:.1f}
• Williams %R: {latest['Williams_R']:.1f}

📊 MOVING AVERAGES:
• SMA 20: ${latest['SMA_20']:.2f} ({((latest['Close'] / latest['SMA_20'] - 1) * 100):+.1f}%)
• SMA 50: ${latest['SMA_50']:.2f} ({((latest['Close'] / latest['SMA_50'] - 1) * 100):+.1f}%)
• SMA 200: ${latest['SMA_200']:.2f} ({((latest['Close'] / latest['SMA_200'] - 1) * 100):+.1f}%)

🎪 BOLLINGER BANDS:
• Upper: ${latest['BB_Upper']:.2f}
• Middle: ${latest['BB_Middle']:.2f}
• Lower: ${latest['BB_Lower']:.2f}
• Position: {'ABOVE' if latest['Close'] > latest['BB_Upper'] else 'BELOW' if latest['Close'] < latest['BB_Lower'] else 'WITHIN'} bands

📊 VOLUME ANALYSIS:
• Current Volume: {latest['Volume']:,.0f}
• 20-Day Avg Volume: {latest['Volume_MA']:,.0f}
• Volume Ratio: {latest['Volume'] / latest['Volume_MA']:.1f}x

🔍 DETAILED SIGNALS:
"""

        # Add detailed signals
        for category, signal_list in signals.items():
            if isinstance(signal_list, list) and signal_list:
                report += f"\n{category.replace('_', ' ').title()}:\n"
                for signal in signal_list:
                    report += f"  • {signal}\n"

        report += f"""
⚖️ SIGNAL SUMMARY:
• Buy Signals: {signals['buy_signals']} 🟢
• Sell Signals: {signals['sell_signals']} 🔴
• Neutral Signals: {signals['neutral_signals']} 🟡

⚠️ RISK DISCLAIMER:
This analysis is for educational purposes only and should not be considered 
as financial advice. Always consult with a financial advisor and do your own 
research before making investment decisions.
"""
        return report


# Simple chatbot interface
class SimpleTradingBot:
    def __init__(self):
        self.stocks = {}

    def analyze_stock(self, symbol):
        """Analyze a single stock"""
        print(f"🔍 Analyzing {symbol.upper()}...")
        analyzer = SimpleStockAnalyzer(symbol)

        if analyzer.fetch_data():
            if analyzer.calculate_all_indicators():
                self.stocks[symbol.upper()] = analyzer
                return analyzer.get_detailed_report()

        return f"❌ Failed to analyze {symbol.upper()}"

    def quick_recommendation(self, symbol):
        """Get quick recommendation for a stock"""
        symbol = symbol.upper()
        if symbol in self.stocks:
            analyzer = self.stocks[symbol]
            rec, conf, reason = analyzer.get_recommendation()
            price = analyzer.data['Close'].iloc[-1]
            return f"{symbol}: {rec} (Confidence: {conf:.1%}) - ${price:.2f}\n{reason}"
        else:
            return self.analyze_stock(symbol)

    def compare_stocks(self, symbols):
        """Compare multiple stocks"""
        comparison = "📊 STOCK COMPARISON\n" + "=" * 50 + "\n"

        for symbol in symbols:
            try:
                result = self.quick_recommendation(symbol)
                comparison += f"\n{result}\n" + "-" * 30 + "\n"
            except Exception as e:
                comparison += f"\n{symbol.upper()}: Error - {str(e)}\n" + "-" * 30 + "\n"

        return comparison


# Demo function
def demo():
    """Demonstrate the stock analyzer"""
    print("🤖 Simple Stock Trading Bot Demo")
    print("=" * 50)

    bot = SimpleTradingBot()

    # Analyze popular stocks
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    print("\n📈 Analyzing popular stocks...")
    for symbol in symbols:
        print(f"\n{bot.analyze_stock(symbol)}")
        print("\n" + "=" * 80 + "\n")

    print("🔄 Quick comparison:")
    print(bot.compare_stocks(symbols))


if __name__ == "__main__":
    # Only requires: pip install yfinance pandas numpy
    demo()