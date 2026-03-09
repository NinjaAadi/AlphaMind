"""
Technical indicator calculations for stock analysis.

Computes RSI, MACD, SMA, VWAP from historical price data.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

from utils.constants import (
    RSI_PERIOD,
    MACD_FAST_PERIOD,
    MACD_SLOW_PERIOD,
    MACD_SIGNAL_PERIOD,
    SMA_SHORT_PERIOD,
    SMA_LONG_PERIOD,
)

logger = logging.getLogger(__name__)


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: List of closing prices (oldest to newest)
        period: Number of periods for the average
    
    Returns:
        SMA value or None if insufficient data
    """
    if not prices or len(prices) < period:
        return None
    
    # Use the most recent 'period' prices
    recent_prices = prices[-period:]
    return sum(recent_prices) / period


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: List of closing prices (oldest to newest)
        period: Number of periods for the EMA
    
    Returns:
        EMA value or None if insufficient data
    """
    if not prices or len(prices) < period:
        return None
    
    multiplier = 2 / (period + 1)
    
    # Start with SMA for the first EMA value
    ema = sum(prices[:period]) / period
    
    # Calculate EMA for remaining prices
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def calculate_rsi(prices: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    
    Args:
        prices: List of closing prices (oldest to newest)
        period: RSI period (default 14)
    
    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if not prices or len(prices) < period + 1:
        return None
    
    # Calculate price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # Separate gains and losses
    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]
    
    # Calculate initial average gain and loss (first 'period' values)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Use Wilder's smoothing for subsequent values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Avoid division by zero
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def calculate_macd(
    prices: List[float],
    fast_period: int = MACD_FAST_PERIOD,
    slow_period: int = MACD_SLOW_PERIOD,
    signal_period: int = MACD_SIGNAL_PERIOD
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line)
    Histogram = MACD Line - Signal Line
    
    Args:
        prices: List of closing prices (oldest to newest)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram) or (None, None, None)
    """
    if not prices or len(prices) < slow_period + signal_period:
        return None, None, None
    
    # Calculate MACD line values for each point
    macd_values = []
    
    for i in range(slow_period, len(prices) + 1):
        subset = prices[:i]
        fast_ema = calculate_ema(subset, fast_period)
        slow_ema = calculate_ema(subset, slow_period)
        
        if fast_ema is not None and slow_ema is not None:
            macd_values.append(fast_ema - slow_ema)
    
    if len(macd_values) < signal_period:
        return None, None, None
    
    # Calculate signal line (EMA of MACD values)
    signal = calculate_ema(macd_values, signal_period)
    
    # Current MACD value
    macd = macd_values[-1] if macd_values else None
    
    # Calculate histogram
    histogram = None
    if macd is not None and signal is not None:
        histogram = macd - signal
    
    return (
        round(macd, 4) if macd is not None else None,
        round(signal, 4) if signal is not None else None,
        round(histogram, 4) if histogram is not None else None
    )


def calculate_vwap(
    prices: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[int]
) -> Optional[float]:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3
    
    Args:
        prices: List of closing prices
        highs: List of high prices
        lows: List of low prices
        volumes: List of volumes
    
    Returns:
        VWAP value or None if insufficient data
    """
    if not prices or not highs or not lows or not volumes:
        return None
    
    if len(prices) != len(highs) or len(prices) != len(lows) or len(prices) != len(volumes):
        return None
    
    total_volume = sum(volumes)
    if total_volume == 0:
        return None
    
    # Calculate cumulative typical price * volume
    tp_volume_sum = 0
    for i in range(len(prices)):
        typical_price = (highs[i] + lows[i] + prices[i]) / 3
        tp_volume_sum += typical_price * volumes[i]
    
    vwap = tp_volume_sum / total_volume
    
    return round(vwap, 2)


def get_rsi_signal(rsi: Optional[float]) -> Optional[str]:
    """
    Get RSI signal interpretation.
    
    Args:
        rsi: RSI value (0-100)
    
    Returns:
        "oversold" if RSI < 30, "overbought" if RSI > 70, "neutral" otherwise
    """
    if rsi is None:
        return None
    
    if rsi < 30:
        return "oversold"
    elif rsi > 70:
        return "overbought"
    else:
        return "neutral"


def get_sma_crossover_signal(sma_short: Optional[float], sma_long: Optional[float]) -> Optional[str]:
    """
    Get SMA crossover signal.
    
    Args:
        sma_short: Short-term SMA (e.g., SMA 50)
        sma_long: Long-term SMA (e.g., SMA 200)
    
    Returns:
        "bullish" if short > long (golden cross), "bearish" if short < long (death cross)
    """
    if sma_short is None or sma_long is None:
        return None
    
    if sma_short > sma_long:
        return "bullish"
    elif sma_short < sma_long:
        return "bearish"
    else:
        return "neutral"


# --- Array-based calculations for charting ---


def calculate_sma_series(prices: List[float], period: int) -> List[Optional[float]]:
    """
    Calculate SMA for each point in the price series.
    
    Args:
        prices: List of closing prices (oldest to newest)
        period: SMA period
    
    Returns:
        List of SMA values (None for first period-1 values)
    """
    result = []
    for i in range(len(prices)):
        if i < period - 1:
            result.append(None)
        else:
            window = prices[i - period + 1:i + 1]
            result.append(round(sum(window) / period, 2))
    return result


def calculate_rsi_series(prices: List[float], period: int = RSI_PERIOD) -> List[Optional[float]]:
    """
    Calculate RSI for each point in the price series.
    
    Args:
        prices: List of closing prices (oldest to newest)
        period: RSI period (default 14)
    
    Returns:
        List of RSI values (None for first period values)
    """
    if len(prices) < period + 1:
        return [None] * len(prices)
    
    result = [None] * period  # First 'period' values are None
    
    # Calculate price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [max(0, c) for c in changes]
    losses = [abs(min(0, c)) for c in changes]
    
    # Initial average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # RSI for first calculable point
    if avg_loss == 0:
        result.append(100.0 if avg_gain > 0 else 50.0)
    else:
        rs = avg_gain / avg_loss
        result.append(round(100 - (100 / (1 + rs)), 2))
    
    # RSI for subsequent points using Wilder's smoothing
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            result.append(100.0 if avg_gain > 0 else 50.0)
        else:
            rs = avg_gain / avg_loss
            result.append(round(100 - (100 / (1 + rs)), 2))
    
    return result


def calculate_ema_series(prices: List[float], period: int) -> List[Optional[float]]:
    """
    Calculate EMA for each point in the price series.
    
    Args:
        prices: List of closing prices (oldest to newest)
        period: EMA period
    
    Returns:
        List of EMA values (None for first period-1 values)
    """
    if len(prices) < period:
        return [None] * len(prices)
    
    result = [None] * (period - 1)
    multiplier = 2 / (period + 1)
    
    # First EMA is SMA
    ema = sum(prices[:period]) / period
    result.append(round(ema, 4))
    
    # Calculate EMA for remaining prices
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
        result.append(round(ema, 4))
    
    return result


def calculate_macd_series(
    prices: List[float],
    fast_period: int = MACD_FAST_PERIOD,
    slow_period: int = MACD_SLOW_PERIOD,
    signal_period: int = MACD_SIGNAL_PERIOD
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    Calculate MACD, Signal, and Histogram series.
    
    Args:
        prices: List of closing prices (oldest to newest)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
    
    Returns:
        Tuple of (MACD series, Signal series, Histogram series)
    """
    n = len(prices)
    
    if n < slow_period + signal_period:
        return [None] * n, [None] * n, [None] * n
    
    # Calculate EMA series
    fast_ema = calculate_ema_series(prices, fast_period)
    slow_ema = calculate_ema_series(prices, slow_period)
    
    # MACD line = Fast EMA - Slow EMA
    macd_line = []
    for i in range(n):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line.append(round(fast_ema[i] - slow_ema[i], 4))
        else:
            macd_line.append(None)
    
    # Signal line = EMA of MACD line (need to filter out Nones first)
    # Find first non-None index
    first_valid = next((i for i, v in enumerate(macd_line) if v is not None), None)
    
    if first_valid is None or n - first_valid < signal_period:
        return macd_line, [None] * n, [None] * n
    
    # Calculate signal line EMA on valid MACD values
    valid_macd = [v for v in macd_line if v is not None]
    signal_ema = calculate_ema_series(valid_macd, signal_period)
    
    # Map signal back to full series
    signal_line = [None] * first_valid
    signal_line.extend(signal_ema)
    
    # Pad if needed
    while len(signal_line) < n:
        signal_line.append(None)
    
    # Histogram = MACD - Signal
    histogram = []
    for i in range(n):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(round(macd_line[i] - signal_line[i], 4))
        else:
            histogram.append(None)
    
    return macd_line, signal_line, histogram
