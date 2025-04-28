import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Attention, 
                                   Bidirectional, Concatenate, Input, Lambda, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from scipy.stats import linregress
import random
import os

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. Utility Functions
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        if 'Ticker' in df.columns.names:
            df.columns = df.columns.droplevel('Ticker')
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in df.columns.values]
    return df

# 2. Enhanced Technical Indicators
def calculate_atr(high, low, close, window=14):
    tr = pd.DataFrame({
        'HL': high - low,
        'HC': np.abs(high - close.shift()),
        'LC': np.abs(low - close.shift())
    }).max(axis=1)
    return tr.rolling(window).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + (std * num_std), sma, sma - (std * num_std)

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    return k, k.rolling(d_window).mean()

def calculate_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - sma) / (0.015 * mad)

def calculate_obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def calculate_fib_levels(close, window=20):
    rolling_max = close.rolling(window).max()
    rolling_min = close.rolling(window).min()
    range_ = rolling_max - rolling_min
    return {
        'FIB_23': rolling_max - range_ * 0.236,
        'FIB_38': rolling_max - range_ * 0.382,
        'FIB_50': rolling_max - range_ * 0.5,
        'FIB_61': rolling_max - range_ * 0.618,
        'FIB_78': rolling_max - range_ * 0.786
    }

# NEW INDICATORS
def calculate_adx(high, low, close, window=14):
    """Average Directional Index - measures trend strength"""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    
    tr1 = pd.DataFrame({
        'HL': high - low,
        'HC': abs(high - close.shift()),
        'LC': abs(low - close.shift())
    }).max(axis=1)
    
    atr = tr1.rolling(window).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx, plus_di, minus_di

def calculate_vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()

def calculate_ichimoku(high, low, close):
    """Ichimoku Cloud indicator"""
    tenkan_window = 9
    kijun_window = 26
    senkou_span_b_window = 52
    
    tenkan_sen = (high.rolling(tenkan_window).max() + low.rolling(tenkan_window).min()) / 2
    kijun_sen = (high.rolling(kijun_window).max() + low.rolling(kijun_window).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)
    senkou_span_b = ((high.rolling(senkou_span_b_window).max() + low.rolling(senkou_span_b_window).min()) / 2).shift(kijun_window)
    chikou_span = close.shift(-kijun_window)
    
    return {
        'ICH_TENKAN': tenkan_sen,
        'ICH_KIJUN': kijun_sen,
        'ICH_SENKOU_A': senkou_span_a,
        'ICH_SENKOU_B': senkou_span_b,
        'ICH_CHIKOU': chikou_span
    }

def calculate_volume_profile(close, volume, window=20):
    """Volume profile analysis"""
    vp = (close * volume).rolling(window).sum() / volume.rolling(window).sum()
    return vp

def calculate_supertrend(high, low, close, atr_multiplier=3, atr_window=10):
    """Supertrend indicator - shows trend direction"""
    atr = calculate_atr(high, low, close, atr_window)
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    supertrend = pd.Series(index=close.index, dtype='float64')
    direction = pd.Series(index=close.index, dtype='int64')
    
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1  # Start with downtrend
    
    for i in range(1, len(close)):
        if close.iloc[i-1] > supertrend.iloc[i-1]:
            current_direction = 1
        else:
            current_direction = -1
            
        if current_direction == 1:
            supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
            
        direction.iloc[i] = current_direction
    
    return supertrend, direction

def calculate_heikin_ashi(open, high, low, close):
    """Heikin Ashi candlestick values"""
    ha_close = (open + high + low + close) / 4
    ha_open = (open.shift(1) + close.shift(1)) / 2
    ha_open.iloc[0] = open.iloc[0]
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    
    return {
        'HA_OPEN': ha_open,
        'HA_HIGH': ha_high,
        'HA_LOW': ha_low,
        'HA_CLOSE': ha_close
    }

def calculate_zscore(series, window=20):
    """Z-Score for mean reversion strategies"""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std

# 3. Enhanced Feature Engineering
def add_technical_indicators(df, ticker):
    df = df.copy()
    ticker = ticker.replace('"', '').replace("'", "")
    df = flatten_columns(df)
    
    # Determine column names dynamically
    if 'Close' in df.columns:
        cols = {
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'open': 'Open',
            'volume': 'Volume'
        }
    elif f'Close_{ticker}' in df.columns:
        cols = {
            'close': f'Close_{ticker}',
            'high': f'High_{ticker}',
            'low': f'Low_{ticker}',
            'open': f'Open_{ticker}',
            'volume': f'Volume_{ticker}'
        }
    else:
        close_candidates = [col for col in df.columns if 'close' in col.lower()]
        high_candidates = [col for col in df.columns if 'high' in col.lower()]
        low_candidates = [col for col in df.columns if 'low' in col.lower()]
        open_candidates = [col for col in df.columns if 'open' in col.lower()]
        volume_candidates = [col for col in df.columns if 'volume' in col.lower()]
        
        if close_candidates and high_candidates and low_candidates and open_candidates and volume_candidates:
            cols = {
                'close': close_candidates[0],
                'high': high_candidates[0],
                'low': low_candidates[0],
                'open': open_candidates[0],
                'volume': volume_candidates[0]
            }
        else:
            raise ValueError(f"Could not determine column names for ticker {ticker}. Available columns: {df.columns.tolist()}")
    
    # Check if required columns exist
    for key, col in cols.items():
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    close = df[cols['close']]
    high = df[cols['high']]
    low = df[cols['low']]
    open = df[cols['open']]
    volume = df[cols['volume']]
    
    # Basic Price Transformations
    df['CLOSE_PCT_CHANGE'] = close.pct_change()
    df['LOG_RETURNS'] = np.log(close) - np.log(close.shift(1))
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    df['EMA_200'] = close.ewm(span=200, adjust=False).mean()
    df['EMA_20_DIST'] = (close - df['EMA_20']) / df['EMA_20']
    df['EMA_CROSS'] = (df['EMA_20'] > df['EMA_50']).astype(int)
    
    # Momentum Indicators
    df['RSI_14'] = calculate_rsi(close, 14)
    df['RSI_7'] = calculate_rsi(close, 7)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(close)
    df['STOCH_K'], df['STOCH_D'] = calculate_stochastic(high, low, close)
    df['CCI_20'] = calculate_cci(high, low, close, 20)
    df['ROC_10'] = close.pct_change(10) * 100
    df['ROC_5'] = close.pct_change(5) * 100
    df['MOM_10'] = close.diff(10)
    
    # Volatility Indicators
    df['ATR_14'] = calculate_atr(high, low, close, 14)
    df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = calculate_bollinger_bands(close)
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
    df['NATR_14'] = df['ATR_14'] / close * 100
    
    # Volume Indicators
    df['OBV'] = calculate_obv(close, volume)
    df['VOLUME_MA_20'] = volume.rolling(20).mean()
    df['VOLUME_SPIKE'] = (volume / df['VOLUME_MA_20'] - 1)
    df['VWAP'] = calculate_vwap(high, low, close, volume)
    df['VOLUME_PROFILE'] = calculate_volume_profile(close, volume)
    
    # Trend Indicators
    df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = calculate_adx(high, low, close)
    ichimoku = calculate_ichimoku(high, low, close)
    df = pd.concat([df, pd.DataFrame(ichimoku)], axis=1)
    df['SUPERTREND'], df['SUPERTREND_DIR'] = calculate_supertrend(high, low, close)
    
    # Mean Reversion Indicators
    df['ZSCORE_20'] = calculate_zscore(close, 20)
    
    # Candlestick Patterns
    ha = calculate_heikin_ashi(open, high, low, close)
    df = pd.concat([df, pd.DataFrame(ha)], axis=1)
    
    # Fibonacci Levels
    fib = calculate_fib_levels(close)
    df = pd.concat([df, pd.DataFrame(fib)], axis=1)
    
    # Gaps
    df['GAP_PCT'] = (open - close.shift(1)) / close.shift(1)
    df['GAP_DIR'] = np.where(df['GAP_PCT'] > 0.01, 1, np.where(df['GAP_PCT'] < -0.01, -1, 0))
    
    # Price Extremes
    df['EXTREME_UP'] = ((high - high.rolling(5).max().shift(1)) / close).abs()
    df['EXTREME_DOWN'] = ((low - low.rolling(5).min().shift(1)) / close).abs()
    
    # Volatility Ratios
    df['VOLATILITY_RATIO'] = df['ATR_14'] / close.rolling(20).std()
    
    # Price Patterns
    df['HIGHER_HIGH'] = (high > high.shift(1)).astype(int)
    df['LOWER_LOW'] = (low < low.shift(1)).astype(int)
    
    # Remove infinite values and fill NA
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

def create_enhanced_model(input_shape):
    """Create a reproducible model architecture with attention mechanism"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer with return_sequences=True
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)))(inputs)
    lstm1 = Dropout(0.3, seed=SEED)(lstm1)
    
    # Attention mechanism - FIXED SHAPE MISMATCH
    attention = Dense(1, activation='tanh', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED))(lstm1)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(256)(attention)  # 256 matches Bidirectional LSTM output
    attention = tf.keras.layers.Permute([2, 1])(attention)
    
    # Multiply attention weights with LSTM outputs
    sent_representation = tf.keras.layers.Multiply()([lstm1, attention])
    
    # Second LSTM layer
    lstm2 = Bidirectional(LSTM(64, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED)))(sent_representation)
    lstm2 = Dropout(0.3, seed=SEED)(lstm2)
    
    # Dense layers
    dense1 = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=SEED))(lstm2)
    dense1 = Dropout(0.3, seed=SEED)(dense1)
    
    # Output layers
    price_output = Dense(1, name='price_output')(dense1)
    move_output = Dense(3, activation='softmax', name='move_output')(dense1)
    
    # Create model
    model = Model(inputs=inputs, outputs=[price_output, move_output])
    
    return model

def analyze_stock(ticker, start_date='2020-01-01', end_date=None):
    """Main function to analyze any stock ticker with reproducible results"""
    # Clean the ticker name
    ticker = ticker.replace('"', '').replace("'", "").strip()
    
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    try:
        # Data Preparation
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return {"error": "No data returned for this ticker and date range"}
        
        # Check if we have enough data
        if len(data) < 100:
            return {"warning": f"Only {len(data)} data points available, which may affect prediction quality", "continue": True}
        
        # For Canadian stocks, we need to handle the column names differently
        if '.CA' in ticker:
            # Canadian stocks often have different column naming
            cols = {
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'open': 'Open',
                'volume': 'Volume'
            }
            # Ensure columns exist
            for col in cols.values():
                if col not in data.columns:
                    return {"error": f"Column {col} not found in data for Canadian stock"}
        else:
            # Handle regular US stocks
            if 'Close' in data.columns:
                cols = {
                    'close': 'Close',
                    'high': 'High',
                    'low': 'Low',
                    'open': 'Open',
                    'volume': 'Volume'
                }
            else:
                return {"error": "Could not determine column names in data"}
        
        # Add technical indicators
        try:
            data = add_technical_indicators(data, ticker)
        except ValueError as e:
            return {"error": f"Error calculating indicators: {str(e)}"}
        
        # Target Engineering
        close_col = cols['close']
        data['TARGET_15D'] = data[close_col].pct_change(15).shift(-15)
        
        if len(data) < 15:
            return {"error": "Not enough data to calculate 15-day targets"}
        
        threshold_upper = data['TARGET_15D'].quantile(0.9)
        threshold_lower = data['TARGET_15D'].quantile(0.1)
        data['RAPID_MOVE'] = 0
        data.loc[data['TARGET_15D'] >= threshold_upper, 'RAPID_MOVE'] = 1
        data.loc[data['TARGET_15D'] <= threshold_lower, 'RAPID_MOVE'] = -1
        data['RAPID_MOVE_CLASS'] = data['RAPID_MOVE'] + 1  # Convert to 0,1,2
        
        # Feature Selection - Simplified for Canadian stocks
        features_to_use = [
            close_col, 'RSI_14', 'MACD', 'ATR_14', 'BB_WIDTH', 'EMA_20_DIST',
            'VOLUME_SPIKE', 'GAP_PCT', 'VOLATILITY_RATIO', 'ADX',
            'VWAP', 'SUPERTREND_DIR', 'ZSCORE_20', 'CLOSE_PCT_CHANGE'
        ]
        
        # Ensure all features exist
        features_to_use = [f for f in features_to_use if f in data.columns]
        
        if len(features_to_use) < 10:
            return {"error": "Not enough features available for prediction"}
        
        # Data Scaling
        target_scaler = RobustScaler()
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        
        scaled_features = feature_scaler.fit_transform(data[features_to_use])
        scaled_target = target_scaler.fit_transform(data[['TARGET_15D']].fillna(0))
        
        # Create sequences
        X, y_reg, y_cls = [], [], []
        seq_length = min(60, len(scaled_features) - 15)
        
        for i in range(seq_length, len(scaled_features) - 15):
            X.append(scaled_features[i-seq_length:i])
            y_reg.append(scaled_target[i + 15][0])
            y_cls.append(data['RAPID_MOVE_CLASS'].iloc[i])
        
        if len(X) < 50:
            return {"warning": f"Only {len(X)} sequences available, which may not be enough for reliable predictions", "continue": True}
        
        X = np.array(X)
        y_reg = np.array(y_reg)
        y_cls = np.array(y_cls)
        
        # Train-Test Split (consistent split with seed)
        np.random.seed(SEED)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split = int(0.8 * len(X))
        X_train, X_test = X[indices[:split]], X[indices[split:]]
        y_reg_train, y_reg_test = y_reg[indices[:split]], y_reg[indices[split:]]
        y_cls_train, y_cls_test = y_cls[indices[:split]], y_cls[indices[split:]]
        
        # Build and Train Model
        model = create_enhanced_model((X_train.shape[1], X_train.shape[2]))
        model.compile(
            optimizer=Adam(0.001),
            loss={
                'price_output': 'mse',
                'move_output': 'sparse_categorical_crossentropy'
            },
            metrics={'move_output': 'accuracy'},
            loss_weights={'price_output': 0.7, 'move_output': 0.3}
        )
        
        history = model.fit(
            X_train,
            {'price_output': y_reg_train, 'move_output': y_cls_train},
            validation_data=(X_test, {'price_output': y_reg_test, 'move_output': y_cls_test}),
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.1, patience=10, monitor='val_loss')
            ],
            verbose=0
        )
        
        # Make predictions for the most recent data
        latest_sequence = scaled_features[-seq_length:]
        latest_sequence = latest_sequence.reshape(1, seq_length, len(features_to_use))
        price_pred, move_pred = model.predict(latest_sequence, verbose=0)
        
        # Convert predictions back to original scale
        price_pred = target_scaler.inverse_transform(price_pred)[0][0]
        move_class = np.argmax(move_pred, axis=1)[0]
        
        # Map class to direction
        move_mapping = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
        move_direction = move_mapping[move_class]
        
        # Calculate confidence based on softmax probabilities
        move_confidence = move_pred[0][move_class] * 100
        
        # Calculate current price
        current_price = data[close_col].iloc[-1]
        target_price = current_price * (1 + price_pred)
        
        # Calculate linear regression for trend
        close_vals = data[close_col].values[-30:]
        x = np.arange(len(close_vals))
        slope, intercept, r_value, p_value, std_err = linregress(x, close_vals)
        trend = "Uptrend" if slope > 0 else "Downtrend"
        trend_strength = abs(r_value)
        
        # Calculate support/resistance levels
        recent_low = data['Low'].rolling(20).min().iloc[-1]
        recent_high = data['High'].rolling(20).max().iloc[-1]
        dist_to_support = (current_price - recent_low) / current_price * 100
        dist_to_resistance = (recent_high - current_price) / current_price * 100
        
        # Fibonacci levels
        fib_levels = calculate_fib_levels(data[close_col])
        
        # Find closest Fibonacci level to predicted price
        predicted_price = current_price * (1 + price_pred)
        closest_fib_level = min(fib_levels.items(), key=lambda x: abs(x[1].iloc[-1] - predicted_price))
        
        # Movement probabilities
        move_probs = move_pred[0]
        
        # Results dictionary
        result = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'predicted_move_percent': round(price_pred * 100, 2),
            'predicted_target_price': round(target_price, 2),
            'movement_direction': move_direction,
            'confidence': round(move_confidence, 1),
            'current_trend': trend,
            'trend_strength': round(trend_strength * 100, 1),
            'rsi_14': round(data['RSI_14'].iloc[-1], 2),
            'atr_14': round(data['ATR_14'].iloc[-1], 2),
            'volatility': round(data['BB_WIDTH'].iloc[-1], 2),
            'macd': round(data['MACD'].iloc[-1], 2),
            'last_updated': data.index[-1].strftime('%Y-%m-%d'),
            'support': round(recent_low, 2),
            'resistance': round(recent_high, 2),
            'dist_to_support': round(dist_to_support, 2),
            'dist_to_resistance': round(dist_to_resistance, 2),
            'fib_23': round(fib_levels['FIB_23'].iloc[-1], 2),
            'fib_38': round(fib_levels['FIB_38'].iloc[-1], 2),
            'fib_50': round(fib_levels['FIB_50'].iloc[-1], 2),
            'fib_61': round(fib_levels['FIB_61'].iloc[-1], 2),
            'fib_78': round(fib_levels['FIB_78'].iloc[-1], 2),
            'closest_fib_level': closest_fib_level[0],
            'closest_fib_price': round(closest_fib_level[1].iloc[-1], 2),
            'bearish_prob': round(move_probs[0]*100, 1),
            'neutral_prob': round(move_probs[1]*100, 1),
            'bullish_prob': round(move_probs[2]*100, 1)
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Error during analysis: {str(e)}"}