import io as _io
import datetime as _dt
import pandas as _pd
import numpy as _np
#import yfinance as _yf
from analisys import stats as _stats

def _score_str(val):
    """Returns + sign for positive values (used in plots)"""
    return ("" if "-" in val else "+") + str(val)

def to_prices(returns, base=1e5):
    """Converts returns series to price data"""
    returns = returns.copy().fillna(0).replace(
        [_np.inf, -_np.inf], float('NaN'))

    return base + base * _stats.compsum(returns)

def log_returns(returns, nperiods=None):
    """Shorthand for to_log_returns"""
    return to_log_returns(returns, nperiods)


def to_log_returns(returns, nperiods=None):
    """Converts returns series to log returns"""
    try:
        return _np.log(returns+1).replace([_np.inf, -_np.inf], float('NaN'))
    except Exception:
        return 0.

def group_returns(returns, groupby, compounded=False):
    """Summarize returns
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        return returns.groupby(groupby).apply(_stats.comp)
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    """Aggregates returns based on date periods"""
    if period is None or 'day' in period:
        return returns
    index = returns.index

    if 'month' in period:
        return group_returns(returns, index.month, compounded=compounded)

    if 'quarter' in period:
        return group_returns(returns, index.quarter, compounded=compounded)

    if period == "A" or any(x in period for x in ['year', 'eoy', 'yoy']):
        return group_returns(returns, index.year, compounded=compounded)

    if 'week' in period:
        return group_returns(returns, index.week, compounded=compounded)

    if 'eow' in period or period == "W":
        return group_returns(returns, [index.year, index.week],
                             compounded=compounded)

    if 'eom' in period or period == "M":
        return group_returns(returns, [index.year, index.month],
                             compounded=compounded)

    if 'eoq' in period or period == "Q":
        return group_returns(returns, [index.year, index.quarter],
                             compounded=compounded)

    if not isinstance(period, str):
        return group_returns(returns, period, compounded)

    return returns


def _prepare_prices(data, base=1.):
    """Converts return data into prices + cleanup"""
    data = data.copy()
    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() <= 0 or data[col].dropna().max() < 1:
                data[col] = to_prices(data[col], base)

    # is it returns?
    # elif data.min() < 0 and data.max() < 1:
    elif data.min() < 0 or data.max() < 1:
        data = to_prices(data, base)

    if isinstance(data, (_pd.DataFrame, _pd.Series)):
        data = data.fillna(0).replace(
            [_np.inf, -_np.inf], float('NaN'))

    return data

def _round_to_closest(val, res, decimals=None):
    """Round to closest resolution"""
    if decimals is None and "." in str(res):
        decimals = len(str(res).split('.')[1])
    return round(round(val / res) * res, decimals)

def _count_consecutive(data):
    """Counts consecutive data (like cumsum() with reset on zeroes)"""
    def _count(data):
        return data * (data.groupby(
            (data != data.shift(1)).cumsum()).cumcount() + 1)

    if isinstance(data, _pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)

def make_portfolio(returns, start_balance=1e5,
                   mode="comp", round_to=None):
    """Calculates compounded value of portfolio"""

    if mode.lower() in ["cumsum", "sum"]:
        p1 = start_balance + start_balance * returns.cumsum()
    elif mode.lower() in ["compsum", "comp"]:
        p1 = to_prices(returns, start_balance)
    else:
        # fixed amount every day
        comp_rev = (start_balance + start_balance *
                    returns.shift(1)).fillna(start_balance) * returns
        p1 = start_balance + comp_rev.cumsum()

    # add day before with starting balance
    p0 = _pd.Series(data=start_balance,
                    index=p1.index + _pd.Timedelta(days=-1))[:1]

    portfolio = _pd.concat([p0, p1])

    if isinstance(returns, _pd.DataFrame):
        portfolio.loc[:1, :] = start_balance
        portfolio.drop(columns=[0], inplace=True)

    if round_to:
        portfolio = _np.round(portfolio, round_to)

    return portfolio
