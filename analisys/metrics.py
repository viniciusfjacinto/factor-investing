import pandas as _pd
import numpy as _np
from math import sqrt as _sqrt, ceil as _ceil
from datetime import (
    datetime as _dt, timedelta as _td
)
from base64 import b64encode as _b64encode
import re as _regex
from tabulate import tabulate as _tabulate
from . import (stats as _stats,utils as _utils, plots as _plots)

def _get_trading_periods(periods_per_year=252):
    half_year = _ceil(periods_per_year/2)
    return periods_per_year, half_year

def _calc_dd(df, display=True, as_pct=False):
    dd = _stats.to_drawdown_series(df)
    dd_info = _stats.drawdown_details(dd)

    if dd_info.empty:
        return _pd.DataFrame()

    if "returns" in dd_info:
        ret_dd = dd_info['returns']
    else:
        ret_dd = dd_info

    dd_stats = {
        'returns': {
            'Drawdown (máximo) %': ret_dd.sort_values(
                by='max drawdown', ascending=True
            )['max drawdown'].values[0] / 100,
            'Drawdown mais longo (dias)': str(_np.round(ret_dd.sort_values(
                by='days', ascending=False)['days'].values[0])),
            'Drawdown (médio) %': ret_dd['max drawdown'].mean() / 100,
            'Dias drawdown (médio)': str(_np.round(ret_dd['days'].mean()))
        }
    }
    if "benchmark" in df and (dd_info.columns, _pd.MultiIndex):
        bench_dd = dd_info['benchmark'].sort_values(by='max drawdown')
        dd_stats['benchmark'] = {
            'Drawdown (máximo) %': bench_dd.sort_values(
                by='max drawdown', ascending=True
            )['max drawdown'].values[0] / 100,
            'Drawdown mais longo (dias)': str(_np.round(bench_dd.sort_values(
                by='days', ascending=False)['days'].values[0])),
            'Drawdown (médio) %': bench_dd['max drawdown'].mean() / 100,
            'Dias drawdown (médio)': str(_np.round(bench_dd['days'].mean()))
        }

    # pct multiplier
    pct = 100 if display or as_pct else 1

    dd_stats = _pd.DataFrame(dd_stats).T
    dd_stats['Drawdown (máximo) %'] = dd_stats['Drawdown (máximo) %'].astype(float) * pct
    dd_stats['Drawdown (médio) %'] = dd_stats['Drawdown (médio) %'].astype(float) * pct

    return dd_stats.T

def metrics(returns, benchmark, rf, inicio, fim, display=True,
            sep=False, compounded=True,
            periods_per_year=252,
            match_dates=False, **kwargs):

    win_year, _ = _get_trading_periods(periods_per_year)
    
    blank = ['', '']
    #Ajustando dados e formando dataframe de análise

    returns1 = returns[returns.columns[0]][inicio:fim]
    benchmark1= benchmark[benchmark.columns[0]][inicio:fim]
    rf= rf[rf.columns[0]][inicio:fim]
        
    df = _pd.DataFrame({"returns": returns1})
    df["benchmark"] = benchmark1
    df = df.fillna(0)
    
    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # return df
    dd = _calc_dd(df, display=(display or "internal" in kwargs),
                  as_pct=kwargs.get("as_pct", False))

    metrics = _pd.DataFrame()
    s_start = {'returns': df['returns'].index.strftime('%Y-%m-%d')[0]}
    s_end = {'returns': df['returns'].index.strftime('%Y-%m-%d')[-1]}

    if "benchmark" in df:
        s_start['benchmark'] = df['benchmark'].index.strftime('%Y-%m-%d')[0]
        s_end['benchmark'] = df['benchmark'].index.strftime('%Y-%m-%d')[-1]

    metrics['Período inicial'] = _pd.Series(s_start)
    metrics['Período final'] = _pd.Series(s_end)
    metrics['CDI %'] = _stats.annualize_rets(rf)
    metrics['Dados disponíveis %'] = _stats.exposure(df) * pct
    
    metrics['~'] = blank
    
    
    metrics['Retorno acumulado %'] = _stats.comp(df) * pct
    metrics['Retorno anualizado %'] = _stats.cagr(df) * pct

    metrics['~~~~~~~~~~~~~~'] = blank

    metrics['Sharpe'] = _stats.sharpe(df, rf)
    metrics['Smart Sharpe'] = _stats.smart_sharpe(df, rf)
    metrics['Sortino'] = _stats.sortino(df, rf)
    metrics['Smart Sortino'] = _stats.smart_sortino(df, rf)
    metrics['Sortino/√2'] = metrics['Sortino'] / _sqrt(2)
    metrics['Smart Sortino/√2'] = metrics['Smart Sortino'] / _sqrt(2)
    metrics['Omega'] = _stats.omega(df, rf)

    metrics['~~~~~~~~'] = blank
    metrics['Drawdown (máximo) %'] = blank
    metrics['Drawdown mais longo (dias)'] = blank

    ret_vol = _stats.volatility(df['returns'], 252, True) * pct
    if "benchmark" in df:
        bench_vol = _stats.volatility(df['benchmark'], 252, True) * pct
        metrics['Volatilidade (an.) %'] = [ret_vol, bench_vol]
        metrics['R^2'] = _stats.r_squared(df['returns'], df['benchmark'])

    metrics['Calmar'] = _stats.calmar(df)
    metrics['Skew'] = _stats.skew(df)
    metrics['Kurtosis'] = _stats.kurtosis(df)
    
    metrics['~~~~~~~~~~'] = blank

    metrics['Retorno esperado diário %'] = _stats.expected_return(df) * pct
    metrics['Retorno esperado mensal%'] = _stats.expected_return(df, aggregate='M') * pct
    metrics['Retorno esperado anual%'] = _stats.expected_return(df, aggregate='A') * pct

    metrics['Kelly Criterion %'] = _stats.kelly_criterion(df) * pct
    metrics['Risk of Ruin %'] = _stats.risk_of_ruin(df)

    metrics['Value-at-Risk diário (VaR) %'] = -abs(_stats.var(df) * pct)
    metrics['Value-at-Risk condicional diário (cVaR) %'] = -abs(_stats.cvar(df) * pct)
    
    metrics['~~~~~~'] = blank
    
    metrics['Gain/Pain Ratio'] = _stats.gain_to_pain_ratio(df)
    metrics['GPR (1M)'] = _stats.gain_to_pain_ratio(df, "M")
    metrics['GPR (3M)'] = _stats.gain_to_pain_ratio(df, "Q")
    metrics['GPR (6M)'] = _stats.gain_to_pain_ratio(df, "2Q")
    metrics['GPR (1Y)'] = _stats.gain_to_pain_ratio(df, "A")
    metrics['~~~~~~~'] = blank
    
    metrics['Payoff Ratio'] = _stats.payoff_ratio(df)
    metrics['Profit Factor'] = _stats.profit_factor(df)
    metrics['Common Sense Ratio'] = _stats.common_sense_ratio(df)
    metrics['CPC Index'] = _stats.cpc_index(df)
    metrics['Tail Ratio'] = _stats.tail_ratio(df)
    metrics['Outlier Win Ratio'] = _stats.outlier_win_ratio(df)
    metrics['Outlier Loss Ratio'] = _stats.outlier_loss_ratio(df)
    
    # returns
    metrics['~~'] = blank
    comp_func = _stats.comp

    today = df.index[-1]  # _dt.today()

    d = today - _td(3*365/12)
    metrics['Retorno (3M (an.)) %'] = comp_func(
        df[df.index >= _dt(d.year, d.month, d.day)]) * pct

    d = today - _td(6*365/12)
    metrics['Retorno (6M (an.)) %'] = comp_func(
        df[df.index >= _dt(d.year, d.month, d.day)]) * pct

    d = today - _td(12*365/12)
    metrics['Retorno (12M (an.)) %'] = comp_func(
        df[df.index >= _dt(d.year, d.month, d.day)]) * pct
    
    d = today - _td(3*365)
    metrics['Retorno (3Y (an.)) %'] = _stats.cagr(
        df[df.index >= _dt(d.year, d.month, d.day)
           ]) * pct
   
    d = today - _td(5*365)
    metrics['Retorno (5Y (an.)) %'] = _stats.cagr(
        df[df.index >= _dt(d.year, d.month, d.day)
           ]) * pct
    
    d = today - _td(10*365)
    metrics['Retorno (10Y (an.)) %'] = _stats.cagr(
        df[df.index >= _dt(d.year, d.month, d.day)
           ]) * pct
    
    # best/worst

    metrics['~~~'] = blank
    metrics['Maior retorno (dia) %'] = _stats.best(df) * pct
    metrics['Menor retorno (dia) %'] = _stats.worst(df) * pct
    metrics['Maior retorno (mês) %'] = _stats.best(df, aggregate='M') * pct
    metrics['Menor retorno (mês) %'] = _stats.worst(df, aggregate='M') * pct
    metrics['Maior retorno (ano) %'] = _stats.best(df, aggregate='A') * pct
    metrics['Menor retorno (ano) %'] = _stats.worst(df, aggregate='A') * pct

    # dd
    metrics['~~~~'] = blank
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics['Recovery Factor'] = _stats.recovery_factor(df)
    metrics['Ulcer Index'] = _stats.ulcer_index(df)
    metrics['Serenity Index'] = _stats.serenity_index(df, rf)
    
    # win rate
    metrics['~~~~~'] = blank
    metrics['Média de ganhos mensais %'] = _stats.avg_win(df, aggregate='M') * pct
    metrics['Média de perdas mensais %'] = _stats.avg_loss(df, aggregate='M') * pct
    metrics['Proporção de ganhos diários %'] = _stats.win_rate(df) * pct
    metrics['Proporção de ganhos mensais %'] = _stats.win_rate(df, aggregate='M') * pct
    metrics['Proporção de ganhos trimestrais %'] = _stats.win_rate(df, aggregate='Q') * pct
    metrics['Proporção de ganhos anuais %'] = _stats.win_rate(df, aggregate='A') * pct

    if "benchmark" in df:
        metrics['~~~~~~~'] = blank
        greeks = _stats.greeks(df['returns'], df['benchmark'], win_year)
        metrics['Beta'] = [str(round(greeks['beta'], 2)), '-']
        metrics['Alpha'] = [str(round(greeks['alpha'], 2)), '-']
                
    # prepare for display
    for col in metrics.columns:
        try:
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except Exception:
            pass
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + '%'
    try:
        metrics['Drawdown mais longo (dias)'] = _pd.to_numeric(
            metrics['Drawdown mais longo (dias)']).astype('int')
        metrics['Dias drawdown (médio)'] = _pd.to_numeric(
            metrics['Dias drawdown (médio)']).astype('int')

        if display or "internal" in kwargs:
            metrics['Drawdown mais longo (dias)'] = metrics['Drawdown mais longo (dias)'].astype(str)
            metrics['Dias drawdown (médio)'] = metrics['Dias drawdown (médio)'
                                                    ].astype(str)
    except Exception:
        metrics['Drawdown mais longo (dias)'] = '-'
        metrics['Dias drawdown (médio)'] = '-'
        if display or "internal" in kwargs:
            metrics['Drawdown mais longo (dias)'] = '-'
            metrics['Dias drawdown (médio)'] = '-'

    metrics.columns = [
        col if '~' not in col else '' for col in metrics.columns]
    metrics.columns = [
        col[:-1] if '%' in col else col for col in metrics.columns]
    metrics = metrics.T

    if "benchmark" in df:
        metrics.columns = ['Estratégia', 'Benchmark']
    else:
        metrics.columns = ['Estratégia']

    if display:
        print(_tabulate(metrics, headers="keys", tablefmt='simple'))
        return None

    if not sep:
        metrics = metrics[metrics.index != '']
    return metrics