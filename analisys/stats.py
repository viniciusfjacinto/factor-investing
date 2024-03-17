from warnings import warn
import pandas as _pd
import numpy as _np
from math import ceil as _ceil, sqrt as _sqrt
from scipy.stats import (
    norm as _norm, linregress as _linregress
)

from analisys import utils as _utils


# ======== STATS ========

def compsum(returns):
    return returns.add(1).cumprod() - 1

def comp(returns):
    return returns.add(1).prod() - 1

def annualize_rets(returns, periods=252):
    return (returns.add(1).prod())**(periods/len(returns))-1

def annualize_vol(returns, periods=252):
    return returns.std()*_np.sqrt(periods)

def annualize_vol_sortino(returns, periods=252):
    dfs=returns - returns.mean()
    dfs[dfs > 0] = 0
    downside = _np.sqrt((dfs**2).sum()/len(dfs))*_np.sqrt(periods)
    return downside

def distribution(returns):
    def get_outliers(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.
        filtered = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)
        return {
            "values": data.loc[filtered].tolist(),
            "outliers": data.loc[~filtered].tolist(),
        }

    apply_fnc = comp
    daily = returns.dropna()

    return {
        "Diário": get_outliers(daily),
        "Semanal": get_outliers(daily.resample('W-MON').apply(apply_fnc)),
        "Mensal": get_outliers(daily.resample('M').apply(apply_fnc)),
        "Trimestral": get_outliers(daily.resample('Q').apply(apply_fnc)),
        "Anual": get_outliers(daily.resample('A').apply(apply_fnc))
    }


def expected_return(returns, aggregate=None, compounded=True):
    returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return _np.product(1 + returns) ** (1 / len(returns)) - 1


def geometric_mean(retruns, aggregate=None, compounded=True):
    return expected_return(retruns, aggregate, compounded)


def ghpr(retruns, aggregate=None, compounded=True):
    return expected_return(retruns, aggregate, compounded)


def outliers(returns, quantile=.95):
    return returns[returns > returns.quantile(quantile)].dropna(how='all')


def remove_outliers(returns, quantile=.95):
    return returns[returns < returns.quantile(quantile)]


def best(returns, aggregate=None, compounded=True):
    return _utils.aggregate_returns(returns, aggregate, compounded).max()


def worst(returns, aggregate=None, compounded=True):
    return _utils.aggregate_returns(returns, aggregate, compounded).min()


def consecutive_wins(returns, aggregate=None, compounded=True):
    returns = _utils.aggregate_returns(returns, aggregate, compounded) > 0
    return _utils._count_consecutive(returns).max()


def consecutive_losses(returns, aggregate=None, compounded=True):
    returns = _utils.aggregate_returns(returns, aggregate, compounded) < 0
    return _utils._count_consecutive(returns).max()


def exposure(returns):
    def _exposure(ret):
        ex = len(ret[(~_np.isnan(ret)) & (ret != 0)]) / len(ret)
        return _ceil(ex * 100) / 100

    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _exposure(returns[col])
        return _pd.Series(_df)
    return _exposure(returns)


def win_rate(returns, aggregate=None, compounded=True):
    def _win_rate(series):
        try:
            return len(series[series > 0]) / len(series[series != 0])
        except Exception:
            return 0.

    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _win_rate(returns[col])

        return _pd.Series(_df)

    return _win_rate(returns)


def avg_return(returns, aggregate=None, compounded=True):
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns != 0].dropna().mean()


def avg_win(returns, aggregate=None, compounded=True):
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns > 0].dropna().mean()


def avg_loss(returns, aggregate=None, compounded=True):
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns < 0].dropna().mean()


def volatility(returns, periods=252, annualize=True):
    std = returns.std()
    if annualize:
        return std * _np.sqrt(periods)
    return std


def rolling_volatility(returns, rolling_period=126, periods_per_year=252):
    return returns.rolling(rolling_period).std() * _np.sqrt(periods_per_year)


def implied_volatility(returns, periods=252, annualize=True):
    logret = _utils.log_returns(returns)
    if annualize:
        return logret.rolling(periods).std() * _np.sqrt(periods)
    return logret.std()


def autocorr_penalty(returns):
    num = len(returns)
    coef = _np.abs(_np.corrcoef(returns[:-1], returns[1:])[0, 1])
    corr = [((num - x)/num) * coef ** x for x in range(1, num)]
    return _np.sqrt(1 + 2 * _np.sum(corr))


# ======= METRICS =======

def sharpe(returns, rf, periods=252, smart=False):
    ann_ret = annualize_rets(returns,periods)
    ann_rf = annualize_rets(rf,periods)
    divisor = annualize_vol(returns,periods) 
    if smart:
        # penalize sharpe with auto correlation
        divisor = divisor * autocorr_penalty(returns)    
    res = (ann_ret-ann_rf) / divisor   
    return res


def smart_sharpe(returns, rf, periods=252):
    return sharpe(returns, rf, periods=252, smart=True)


def rolling_sharpe(returns, rf, rolling_period=126, periods_per_year=252):  
    ann_ret = returns.rolling(rolling_period).apply(annualize_rets)
    ann_rf = rf.rolling(rolling_period).apply(annualize_rets)
    ann_vol = returns.rolling(rolling_period).apply(annualize_vol)  
    res=(ann_ret - ann_rf)/ann_vol
    return res

def sortino(returns, rf, periods=252, smart=False):
    ann_ret = annualize_rets(returns,periods)
    ann_rf = annualize_rets(rf,periods)
    downside = annualize_vol_sortino(returns,periods) 
    if smart:
        # penalize sharpe with auto correlation
        downside = downside * autocorr_penalty(returns)  
    res = (ann_ret-ann_rf) / downside    
    return res


def smart_sortino(returns, rf, periods=252):
    return sortino(returns, rf, periods, smart=True)


def rolling_sortino(returns, rf, rolling_period=126, periods_per_year=252):
    ann_ret = returns.rolling(rolling_period).apply(annualize_rets)
    ann_rf = rf.rolling(rolling_period).apply(annualize_rets)
    ann_vol = returns.rolling(rolling_period).apply(annualize_vol_sortino)   
    res=(ann_ret.iloc[:,0] - ann_rf.iloc[:,0])/ann_vol.iloc[:,0]
    return res


def adjusted_sortino(returns, periods=252, smart=False):
    data = sortino(
        returns, rf, periods=periods, smart=smart)
    return data / _sqrt(2)


def omega(returns, rf, periods=252):
    returns_less_thresh = _pd.DataFrame(returns - rf)
    numer = returns_less_thresh[returns_less_thresh > 0.0].sum().values[0]
    denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum().values[0]

    if denom > 0.0:
        return numer / denom

    return _np.nan


def gain_to_pain_ratio(returns, resolution="D"):
    returns = returns.resample(resolution).sum()
    downside = abs(returns[returns < 0].sum())
    return returns.sum() / downside


def cagr(returns, periods=252):
    ann_ret = annualize_rets(returns,periods)
    return ann_ret


def rar(returns):
    return (cagr(returns)-cagr(rf))/ exposure(returns)


def skew(returns):
    return returns.skew()


def kurtosis(returns):
    return returns.kurtosis()


def calmar(returns):
    cagr_ratio = cagr(returns)
    max_dd = max_drawdown(returns)
    return cagr_ratio / abs(max_dd)


def ulcer_index(returns):
    dd = to_drawdown_series(returns)
    return _np.sqrt(_np.divide((dd**2).sum(), returns.shape[0] - 1))


def ulcer_performance_index(returns, rf):
    return (comp(returns)-comp(rf)) / ulcer_index(returns)


def upi(returns, rf):
    return ulcer_performance_index(returns, rf)


def serenity_index(returns, rf):
    dd = to_drawdown_series(returns)
    pitfall = - cvar(dd) / returns.std()
    return (comp(returns)-comp(rf)) / (ulcer_index(returns) * pitfall)


def risk_of_ruin(returns):
    wins = win_rate(returns)
    return ((1 - wins) / (1 + wins)) ** len(returns)


def ror(returns):
    return risk_of_ruin(returns)


def value_at_risk(returns, sigma=1, confidence=0.95):
    mu = returns.mean()
    sigma *= returns.std()
    if confidence > 1:
        confidence = confidence/100
    return _norm.ppf(1-confidence, mu, sigma)


def var(returns, sigma=1, confidence=0.95):
    return value_at_risk(returns, sigma, confidence)


def conditional_value_at_risk(returns, sigma=1, confidence=0.95):
    var = value_at_risk(returns, sigma, confidence)
    c_var = returns[returns < var].values.mean()
    return c_var if ~_np.isnan(c_var) else var


def cvar(returns, sigma=1, confidence=0.95):
    return conditional_value_at_risk(
        returns, sigma, confidence)


def expected_shortfall(returns, sigma=1, confidence=0.95):
    return conditional_value_at_risk(returns, sigma, confidence)


def tail_ratio(returns, cutoff=0.95):
    return abs(returns.quantile(cutoff) / returns.quantile(1-cutoff))


def payoff_ratio(returns):
    return avg_win(returns) / abs(avg_loss(returns))


def win_loss_ratio(returns):
    return payoff_ratio(returns)


def profit_ratio(returns):
    wins = returns[returns >= 0]
    loss = returns[returns < 0]
    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())
    try:
        return win_ratio / loss_ratio
    except Exception:
        return 0.


def profit_factor(returns):
    return abs(returns[returns >= 0].sum() / returns[returns < 0].sum())


def cpc_index(returns):
    return profit_factor(returns) * win_rate(returns) * \
        win_loss_ratio(returns)


def common_sense_ratio(returns):
    return profit_factor(returns) * tail_ratio(returns)


def outlier_win_ratio(returns, quantile=.99):
    return returns.quantile(quantile).mean() / returns[returns >= 0].mean()


def outlier_loss_ratio(returns, quantile=.01):
    return returns.quantile(quantile).mean() / returns[returns < 0].mean()


def recovery_factor(returns):
    total_returns = comp(returns)
    max_dd = max_drawdown(returns)
    return total_returns / abs(max_dd)


def risk_return_ratio(returns):   
    ann_ret = annualize_rets(returns,periods)
    divisor = annualize_vol(returns,periods)   
    res = (ann_ret) / divisor
    
    return res


def max_drawdown(prices):
    prices = _utils._prepare_prices(prices)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def to_drawdown_series(returns):
    prices = _utils._prepare_prices(returns)
    dd = prices / _np.maximum.accumulate(prices) - 1.
    return dd.replace([_np.inf, -_np.inf, -0], 0)


def drawdown_details(drawdown):
    def _drawdown_details(drawdown):
        # mark no drawdown
        no_dd = drawdown == 0

        # extract dd start dates
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts].index)

        # extract end dates
        ends = no_dd & (~no_dd).shift(1)
        ends = list(ends[ends].index)

        # no drawdown :)
        if not starts:
            return _pd.DataFrame(
                index=[], columns=('start', 'valley', 'end', 'days',
                                   'max drawdown', '99% max drawdown'))

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # build dataframe from results
        data = []
        for i, _ in enumerate(starts):
            dd = drawdown[starts[i]:ends[i]]
            clean_dd = -remove_outliers(-dd, .99)
            data.append((starts[i], dd.idxmin(), ends[i],
                         (ends[i] - starts[i]).days,
                         dd.min() * 100, clean_dd.min() * 100))

        df = _pd.DataFrame(data=data,
                           columns=('start', 'valley', 'end', 'days',
                                    'max drawdown',
                                    '99% max drawdown'))
        df['days'] = df['days'].astype(int)
        df['max drawdown'] = df['max drawdown'].astype(float)
        df['99% max drawdown'] = df['99% max drawdown'].astype(float)

        df['start'] = df['start'].dt.strftime('%Y-%m-%d')
        df['end'] = df['end'].dt.strftime('%Y-%m-%d')
        df['valley'] = df['valley'].dt.strftime('%Y-%m-%d')

        return df

    if isinstance(drawdown, _pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return _pd.concat(_dfs, axis=1)

    return _drawdown_details(drawdown)


def kelly_criterion(returns):
    win_loss_ratio = payoff_ratio(returns)
    win_prob = win_rate(returns)
    lose_prob = 1 - win_prob
    return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio


# ==== VS. BENCHMARK ====

def r_squared(returns, benchmark):
    # slope, intercept, r_val, p_val, std_err = _linregress(
    _, _, r_val, _, _ = _linregress(returns,benchmark)
    return r_val**2


def r2(returns, benchmark):
    return r_squared(returns, benchmark)


def information_ratio(returns, benchmark):
    diff_rets = returns - benchmark
    return diff_rets.mean() / diff_rets.std()


def greeks(returns, benchmark, periods=252.):
    # find covariance
    matrix = _np.cov(returns, benchmark)
    beta = matrix[0, 1] / matrix[1, 1]

    # calculates measures now
    alpha = returns.mean() - beta * benchmark.mean()
    alpha = alpha * periods

    return _pd.Series({
        "beta":  beta,
        "alpha": alpha,
        # "vol": _np.sqrt(matrix[0, 0]) * _np.sqrt(periods)
    }).fillna(0)


def rolling_greeks(returns, benchmark, periods=252):
    df = _pd.DataFrame(data={
        "returns": returns,
        "benchmark": benchmark
    })
    df = df.fillna(0)
    corr = df.rolling(int(periods)).corr().unstack()['returns']['benchmark']
    std = df.rolling(int(periods)).std()
    beta = corr * std['returns'] / std['benchmark']

    alpha = df['returns'].mean() - beta * df['benchmark'].mean()

    # alpha = alpha * periods
    return _pd.DataFrame(index=returns.index, data={
        "beta": beta,
        "alpha": alpha
    }).fillna(0)


def compare(returns, benchmark, aggregate=None, compounded=True, round_vals=None):
    data = _pd.DataFrame(data={
        'Benchmark': _utils.aggregate_returns(
            benchmark, aggregate, compounded) * 100,
        'Returns': _utils.aggregate_returns(
            returns, aggregate, compounded) * 100
    })

    data['Multiplier'] = data['Returns'] / data['Benchmark']
    data['Won'] = _np.where(data['Returns'] >= data['Benchmark'], '+', '-')

    if round_vals is not None:
        return _np.round(data, round_vals)

    return data


def monthly_returns(returns, eoy=True, compounded=True):
    if isinstance(returns, _pd.DataFrame):
        warn("Pandas DataFrame was passed (Series expeted). "
             "Only first column will be used.")
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and 'close' in returns.columns:
            returns = returns['close']
        else:
            returns = returns[returns.columns[0]]

    original_returns = returns.copy()

    returns = _pd.DataFrame(
        _utils.group_returns(returns,
                             returns.index.strftime('%Y-%m-01'),
                             compounded))

    returns.columns = ['Returns']
    returns.index = _pd.to_datetime(returns.index)

    # get returnsframe
    returns['Year'] = returns.index.strftime('%Y')
    returns['Month'] = returns.index.strftime('%b')

    # make pivot table
    returns = returns.pivot('Year', 'Month', 'Returns').fillna(0)

    # handle missing months
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        if month not in returns.columns:
            returns.loc[:, month] = 0

    # order columns by month
    returns = returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

    if eoy:
        returns['eoy'] = _utils.group_returns(
            original_returns, original_returns.index.year).values

    returns.columns = map(lambda x: str(x).upper(), returns.columns)
    returns.index.name = None

    return returns
