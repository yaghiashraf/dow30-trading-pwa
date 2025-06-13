  import os
  import asyncio
  import json
  from datetime import datetime, timedelta
  from pathlib import Path
  from typing import Dict, List, Optional

  import pandas as pd
  import numpy as np
  import yfinance as yf
  import requests
  from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
  from fastapi.staticfiles import StaticFiles
  from fastapi.middleware.cors import CORSMiddleware
  from sklearn.mixture import GaussianMixture
  from sklearn.preprocessing import StandardScaler
  from scipy.fft import fft, fftfreq
  import warnings
  warnings.filterwarnings('ignore')

  try:
      from prophet import Prophet
      PROPHET_AVAILABLE = True
  except ImportError:
      from statsmodels.tsa.seasonal import STL
      PROPHET_AVAILABLE = False

  app = FastAPI(title="Dow30 Trading PWA")

  # CORS middleware
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  # Dow 30 symbols
  DOW30_SYMBOLS = [
      "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
      "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
      "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
  ]

  # Data directory
  DATA_DIR = Path("data")
  DATA_DIR.mkdir(exist_ok=True)

  # WebSocket connections
  websocket_connections = []


  class DataFetcher:
      @staticmethod
      def fetch_yahoo(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
          try:
              ticker = yf.Ticker(symbol)
              data = ticker.history(period=period, auto_adjust=True)
              if data.empty:
                  return None
              return data[['Close']].dropna()
          except Exception:
              return None

      @staticmethod
      def fetch_stooq(symbol: str) -> Optional[pd.DataFrame]:
          try:
              url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
              response = requests.get(url, timeout=10)
              response.raise_for_status()

              from io import StringIO
              df = pd.read_csv(StringIO(response.text))
              df['Date'] = pd.to_datetime(df['Date'])
              df = df.set_index('Date').sort_index()
              return df[['Close']].dropna()
          except Exception:
              return None

      @classmethod
      def fetch_data(cls, symbol: str) -> Optional[pd.DataFrame]:
          # Try Yahoo first
          data = cls.fetch_yahoo(symbol)
          if data is not None and not data.empty:
              return data

          # Fallback to Stooq
          data = cls.fetch_stooq(symbol)
          if data is not None and not data.empty:
              return data

          return None


  class SignalEngine:
      @staticmethod
      def compute_fft_features(prices: pd.Series) -> Dict:
          """Compute FFT features"""
          values = prices.values
          n = len(values)

          # FFT
          fft_vals = fft(values)
          freqs = fftfreq(n)

          # Get dominant frequencies (positive frequencies only)
          pos_mask = freqs > 0
          pos_freqs = freqs[pos_mask]
          pos_amplitudes = np.abs(fft_vals[pos_mask])

          # Find top 3 dominant frequencies
          top_indices = np.argsort(pos_amplitudes)[-3:]
          dominant_freqs = pos_freqs[top_indices]
          dominant_amplitudes = pos_amplitudes[top_indices]
          dominant_wavelengths = 1 / dominant_freqs

          return {
              'frequencies': dominant_freqs.tolist(),
              'amplitudes': dominant_amplitudes.tolist(),
              'wavelengths': dominant_wavelengths.tolist()
          }

      @staticmethod
      def compute_fibonacci_levels(prices: pd.Series) -> Dict:
          """Compute Fibonacci retracement levels"""
          high = prices.max()
          low = prices.min()
          diff = high - low

          # Standard Fibonacci levels
          levels = {
              '0.0': high,
              '23.6': high - 0.236 * diff,
              '38.2': high - 0.382 * diff,
              '50.0': high - 0.5 * diff,
              '61.8': high - 0.618 * diff,
              '78.6': high - 0.786 * diff,
              '100.0': low
          }

          # Extensions
          extensions = {
              '127.2': low - 0.272 * diff,
              '161.8': low - 0.618 * diff
          }

          return {'retracements': levels, 'extensions': extensions}

      @staticmethod
      def compute_regimes(prices: pd.Series) -> np.ndarray:
          """Compute market regimes using Gaussian Mixture Model"""
          returns = prices.pct_change().dropna()
          volatility = returns.rolling(20).std()

          # Create features
          features = pd.DataFrame({
              'returns': returns,
              'volatility': volatility
          }).dropna()

          if len(features) < 10:
              return np.zeros(len(prices))

          # Standardize features
          scaler = StandardScaler()
          features_scaled = scaler.fit_transform(features)

          # Fit Gaussian Mixture Model
          gmm = GaussianMixture(n_components=3, random_state=42)
          regimes = gmm.fit_predict(features_scaled)

          # Extend to full price series
          full_regimes = np.zeros(len(prices))
          full_regimes[-len(regimes):] = regimes

          return full_regimes

      @staticmethod
      def compute_forecast(prices: pd.Series, periods: int = 90) -> Dict:
          """Compute forecast using Prophet or STL decomposition"""
          df = prices.reset_index()
          df.columns = ['ds', 'y']

          if PROPHET_AVAILABLE:
              try:
                  model = Prophet(daily_seasonality=True, yearly_seasonality=True)
                  model.fit(df)

                  future = model.make_future_dataframe(periods=periods)
                  forecast = model.predict(future)

                  return {
                      'forecast': forecast[['ds', 'yhat', 'yhat_lower',
  'yhat_upper']].tail(periods).to_dict('records'),
                      'method': 'prophet'
                  }
              except Exception:
                  pass

          # Fallback to STL decomposition
          try:
              stl = STL(prices, seasonal=13)
              result = stl.fit()

              trend = result.trend
              seasonal = result.seasonal

              # Simple extrapolation
              last_trend = trend.iloc[-1]
              last_seasonal = seasonal.iloc[-252:] if len(seasonal) >= 252 else seasonal

              forecast_dates = pd.date_range(
                  start=prices.index[-1] + timedelta(days=1),
                  periods=periods,
                  freq='D'
              )

              # Repeat seasonal pattern
              seasonal_forecast = np.tile(last_seasonal.values, (periods // len(last_seasonal) +
  1))[:periods]
              trend_forecast = np.full(periods, last_trend)

              forecast_values = trend_forecast + seasonal_forecast

              return {
                  'forecast': [
                      {'ds': date.isoformat(), 'yhat': val, 'yhat_lower': val * 0.95, 'yhat_upper':
   val * 1.05}
                      for date, val in zip(forecast_dates, forecast_values)
                  ],
                  'method': 'stl'
              }
          except Exception:
              return {'forecast': [], 'method': 'none'}

      @classmethod
      def compute_alpha_score(cls, prices: pd.Series) -> float:
          """Compute alpha score (0-100)"""
          if len(prices) < 50:
              return 0

          current_price = prices.iloc[-1]

          # FFT features
          fft_features = cls.compute_fft_features(prices)

          # Fibonacci levels
          fib_levels = cls.compute_fibonacci_levels(prices)

          # Regimes
          regimes = cls.compute_regimes(prices)
          current_regime = regimes[-1] if len(regimes) > 0 else 1

          # Forecast
          forecast_data = cls.compute_forecast(prices)

          # Score components
          fft_score = 0
          if fft_features['amplitudes']:
              # Higher amplitude = stronger signal
              fft_score = min(np.mean(fft_features['amplitudes']) / current_price * 100, 30)

          fib_score = 0
          # Distance to nearest Fibonacci level
          all_levels = list(fib_levels['retracements'].values()) +
  list(fib_levels['extensions'].values())
          min_distance = min(abs(current_price - level) for level in all_levels)
          fib_score = max(0, 25 - (min_distance / current_price * 100))

          regime_score = 0
          # Prefer regime 2 (assuming it's the uptrend low-vol regime)
          if current_regime == 2:
              regime_score = 25
          elif current_regime == 1:
              regime_score = 15
          else:
              regime_score = 5

          forecast_score = 0
          if forecast_data['forecast']:
              next_forecast = forecast_data['forecast'][0]['yhat']
              if next_forecast > current_price:
                  forecast_score = min((next_forecast - current_price) / current_price * 100, 20)

          total_score = fft_score + fib_score + regime_score + forecast_score
          return min(max(total_score, 0), 100)


  class BacktestEngine:
      @staticmethod
      def run_backtest(prices: pd.Series, alpha_scores: pd.Series) -> Dict:
          """Run simple long/flat backtest"""
          if len(prices) != len(alpha_scores):
              return {'error': 'Price and alpha score lengths mismatch'}

          df = pd.DataFrame({
              'price': prices,
              'alpha': alpha_scores
          }).dropna()

          if len(df) < 10:
              return {'error': 'Insufficient data'}

          # Generate signals
          df['position'] = 0
          df.loc[df['alpha'] >= 70, 'position'] = 1  # Long
          df.loc[df['alpha'] <= 40, 'position'] = 0  # Flat
          df['position'] = df['position'].fillna(method='ffill')

          # Calculate returns
          df['returns'] = df['price'].pct_change()
          df['strategy_returns'] = df['position'].shift(1) * df['returns']
          df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()

          # Performance metrics
          total_return = df['cumulative_returns'].iloc[-1] - 1
          years = len(df) / 252
          cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

          volatility = df['strategy_returns'].std() * np.sqrt(252)
          sharpe = cagr / volatility if volatility > 0 else 0

          # Max drawdown
          rolling_max = df['cumulative_returns'].expanding().max()
          drawdown = (df['cumulative_returns'] - rolling_max) / rolling_max
          max_drawdown = drawdown.min()

          return {
              'total_return': total_return,
              'cagr': cagr,
              'sharpe': sharpe,
              'max_drawdown': max_drawdown,
              'volatility': volatility,
              'equity_curve': df[['cumulative_returns']].reset_index().to_dict('records')
          }


  # API Routes
  @app.get("/")
  async def root():
      return {"message": "Dow30 Trading PWA API", "status": "running", "endpoints":
  ["/api/symbols", "/api/price/{symbol}", "/api/alpha/{symbol}", "/api/backtest/{symbol}"]}

  @app.get("/api/symbols")
  async def get_symbols():
      return {"symbols": DOW30_SYMBOLS}


  @app.get("/api/price/{symbol}")
  async def get_price_data(symbol: str):
      if symbol not in DOW30_SYMBOLS:
          raise HTTPException(status_code=404, detail="Symbol not found")

      # Check cache
      cache_file = DATA_DIR / f"{symbol}.parquet"

      if cache_file.exists():
          try:
              df = pd.read_parquet(cache_file)
              return {
                  "symbol": symbol,
                  "data": df.reset_index().to_dict('records'),
                  "cached": True
              }
          except Exception:
              pass

      # Fetch fresh data
      df = DataFetcher.fetch_data(symbol)
      if df is None:
          raise HTTPException(status_code=500, detail="Failed to fetch data")

      # Cache data
      df.to_parquet(cache_file)

      return {
          "symbol": symbol,
          "data": df.reset_index().to_dict('records'),
          "cached": False
      }


  @app.get("/api/alpha/{symbol}")
  async def get_alpha_score(symbol: str):
      if symbol not in DOW30_SYMBOLS:
          raise HTTPException(status_code=404, detail="Symbol not found")

      # Get price data
      price_response = await get_price_data(symbol)
      df = pd.DataFrame(price_response["data"])
      df['Date'] = pd.to_datetime(df['Date'])
      df = df.set_index('Date')

      prices = df['Close']
      alpha_score = SignalEngine.compute_alpha_score(prices)

      # Get additional features
      fft_features = SignalEngine.compute_fft_features(prices)
      fib_levels = SignalEngine.compute_fibonacci_levels(prices)
      regimes = SignalEngine.compute_regimes(prices)
      forecast = SignalEngine.compute_forecast(prices)

      return {
          "symbol": symbol,
          "alpha_score": alpha_score,
          "fft_features": fft_features,
          "fibonacci_levels": fib_levels,
          "regimes": regimes.tolist(),
          "forecast": forecast
      }


  @app.get("/api/backtest/{symbol}")
  async def get_backtest(symbol: str):
      if symbol not in DOW30_SYMBOLS:
          raise HTTPException(status_code=404, detail="Symbol not found")

      # Get price data
      price_response = await get_price_data(symbol)
      df = pd.DataFrame(price_response["data"])
      df['Date'] = pd.to_datetime(df['Date'])
      df = df.set_index('Date')

      prices = df['Close']

      # Calculate alpha scores for entire series
      alpha_scores = []
      for i in range(50, len(prices)):
          score = SignalEngine.compute_alpha_score(prices.iloc[:i+1])
          alpha_scores.append(score)

      # Align series
      aligned_prices = prices.iloc[50:]
      alpha_series = pd.Series(alpha_scores, index=aligned_prices.index)

      backtest_results = BacktestEngine.run_backtest(aligned_prices, alpha_series)

      return {
          "symbol": symbol,
          "backtest": backtest_results
      }


  @app.post("/api/refresh")
  async def refresh_data():
      """Refresh all data and broadcast updates via WebSocket"""
      results = {}

      for symbol in DOW30_SYMBOLS:
          # Delete cache
          cache_file = DATA_DIR / f"{symbol}.parquet"
          if cache_file.exists():
              cache_file.unlink()

          # Fetch fresh data
          df = DataFetcher.fetch_data(symbol)
          if df is not None:
              df.to_parquet(cache_file)
              alpha_score = SignalEngine.compute_alpha_score(df['Close'])
              results[symbol] = alpha_score
          else:
              results[symbol] = 0

      # Broadcast to WebSocket clients
      message = json.dumps({"type": "alpha_update", "data": results})
      for connection in websocket_connections:
          try:
              await connection.send_text(message)
          except Exception:
              pass

      return {"status": "success", "alpha_scores": results}


  @app.websocket("/ws/stream")
  async def websocket_endpoint(websocket: WebSocket):
      await websocket.accept()
      websocket_connections.append(websocket)

      try:
          while True:
              await websocket.receive_text()
      except WebSocketDisconnect:
          websocket_connections.remove(websocket)


  if __name__ == "__main__":
      import uvicorn
      port = int(os.environ.get("PORT", 8000))
      uvicorn.run(app, host="0.0.0.0", port=port)
