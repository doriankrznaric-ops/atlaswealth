import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize

# Check for scikit-learn
try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("⚠️ SYSTEM WARNING: 'scikit-learn' missing. Run: pip install scikit-learn")

# ---------------------------------------------------------
# 1. SYSTEM CONFIGURATION
# ---------------------------------------------------------
FIRM_NAME = "ATLAS PRIVATE WEALTH"
SYSTEM_VERSION = "V47.5 REALITY CHECK (BETA STRESS + BUBBLE PENALTY)"

st.set_page_config(page_title=FIRM_NAME, layout="wide")

# ---------------------------------------------------------
# 2. DESIGN: "THE SOVEREIGN" (INSTITUTIONAL DARK)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* GLOBAL RESET */
    .stApp { background-color: #000000; color: #c9d1d9; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #1c1c1e; }
    
    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        padding: 15px;
        border-radius: 4px;
        border: 1px solid #2c2c2e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricLabel"] { font-size: 10px; color: #666; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 26px; color: #e5e5e5; font-family: 'SF Mono', 'Consolas', monospace; font-weight: 400; }
    
    /* SOVEREIGN BANNERS */
    .sov-banner { 
        padding: 20px; border-radius: 4px; margin-bottom: 25px; text-align: center; 
        font-weight: 800; letter-spacing: 2px; text-transform: uppercase; font-size: 16px; 
        border-left: 5px solid; background-color: #0e0e0e; color: #fff;
    }
    .sov-green { border-color: #10b981; background: rgba(16, 185, 129, 0.1); color: #34d399; }
    .sov-red { border-color: #ef4444; background: rgba(239, 68, 68, 0.1); color: #f87171; }
    .sov-yellow { border-color: #f59e0b; background: rgba(245, 158, 11, 0.1); color: #fbbf24; }
    .sov-blue { border-color: #3b82f6; background: rgba(59, 130, 246, 0.1); color: #60a5fa; }
    
    /* DATA TABLES */
    .aladdin-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px; }
    .aladdin-table td { padding: 8px; border-bottom: 1px solid #27272a; color: #a1a1aa; }
    .aladdin-table th { text-align: left; padding: 8px; border-bottom: 1px solid #52525b; color: #52525b; text-transform: uppercase; }
    .risk-high { color: #ef4444; font-weight: bold; }
    .risk-low { color: #10b981; font-weight: bold; }

    /* INPUTS & BUTTONS */
    .stTextInput>div>div>input { background-color: #000; color: #fff; border: 1px solid #333; }
    .stButton>button { background-color: #18181b; color: #e4e4e7; border: 1px solid #3f3f46; font-weight: 700; width: 100%; }
    .stButton>button:hover { background-color: #27272a; border-color: #fff; color: #fff; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; margin-bottom: 20px; }
    .stTabs [data-baseweb="tab"] { background-color: #0d1117; border: 1px solid #30363d; color: #8b949e; border-radius: 6px; font-size: 11px; height: 35px; }
    .stTabs [aria-selected="true"] { background-color: #161b22; color: #58a6ff; border-bottom: 2px solid #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. MATHEMATICAL ENGINES
# ---------------------------------------------------------

# --- A. MACRO PULSE ---
def get_macro_context():
    try:
        tickers = ["^TNX", "^IRX", "^VIX"]
        data = yf.download(tickers, period="10d", progress=False)['Close']
        data = data.ffill().bfill()
        
        if data.empty: return {"state": "DATA_ERROR", "multiplier": 1.0, "spread": 0, "vix": 0, "vix_status": "ERR"}
        
        if isinstance(data.columns, pd.MultiIndex):
            tnx = data['^TNX'].iloc[-1]; irx = data['^IRX'].iloc[-1]; vix = data['^VIX'].iloc[-1]
        else:
            tnx = data['^TNX'].iloc[-1]; irx = data['^IRX'].iloc[-1]; vix = data['^VIX'].iloc[-1]
        
        yield_spread = tnx - irx
        macro_state = "EXPANSION"; macro_mult = 1.0
        if yield_spread < 0:
            macro_state = "RECESSION WARNING"; macro_mult = 0.5 
            
        vix_status = "NORMAL"
        if vix > 40.0: macro_state = "MARKET CRASH"; macro_mult = 0.0; vix_status = "CRASH"
        elif vix > 25.0: vix_status = "ELEVATED"; macro_mult = min(macro_mult, 0.6)
            
        return {"state": macro_state, "multiplier": macro_mult, "spread": float(yield_spread), "vix": float(vix), "vix_status": vix_status}
    except: return {"state": "SOURCE_ERROR", "multiplier": 1.0, "spread": 0, "vix": 0, "vix_status": "ERR"}

# --- B. GMM REGIME (ML) ---
def get_market_regime_gmm():
    if not SKLEARN_AVAILABLE: return "ML ENGINE OFFLINE", 1.0, "sov-blue"
    try:
        data = yf.download(["^VIX", "SPY"], period="6mo", progress=False)['Close']
        data = data.ffill().bfill()
        
        if isinstance(data.columns, pd.MultiIndex): vix_val = data['^VIX'].iloc[-1]
        else: vix_val = data['^VIX'].iloc[-1]

        if vix_val < 20.0: return "LOW VOLATILITY REGIME (GROWTH)", 1.0, "sov-green"

        df = pd.DataFrame()
        df['VIX'] = data['^VIX'] if isinstance(data.columns, pd.MultiIndex) else data['^VIX']
        df['SPY_VOL'] = data['SPY'].pct_change().rolling(5).std() if isinstance(data.columns, pd.MultiIndex) else data['SPY'].pct_change().rolling(5).std()
        df = df.dropna()
        if df.empty: return "DATA INSUFFICIENT", 1.0, "sov-blue"

        X = df.values
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        current_state = gmm.predict([X[-1]])[0]
        means = gmm.means_[:, 0]
        high_risk_state = np.argmax(means)
        
        if current_state == high_risk_state: return "HIGH VOLATILITY REGIME (CRISIS)", 0.0, "sov-red"
        else: return "LOW VOLATILITY REGIME (GROWTH)", 1.0, "sov-green"
    except: return "REGIME DETECTION FAILED", 1.0, "sov-blue"

# --- C. VOLATILITY CLUSTERING ---
def get_volatility_context(returns):
    try:
        vol = returns.ewm(span=20, adjust=False).std() * np.sqrt(252)
        current_vol = vol.iloc[-1]
        mean_vol = vol.mean()
        vol_ratio = current_vol / mean_vol if mean_vol > 0 else 1.0
        
        status = "STABLE"; penalty = 1.0
        if vol_ratio > 1.5: status = "HIGH ENERGY"; penalty = 0.7
        elif vol_ratio > 1.2: status = "RISING"; penalty = 0.9
        return {"current": current_vol, "mean": mean_vol, "status": status, "penalty": penalty}
    except: return {"current": 0, "mean": 0, "status": "ERROR", "penalty": 1.0}

# --- D. BLACK-LITTERMAN LITE ---
def black_litterman_lite(expected_return_atlas, confidence=0.5):
    try:
        market_equilibrium = 0.10 
        bl_return = (confidence * expected_return_atlas) + ((1 - confidence) * market_equilibrium)
        return bl_return, market_equilibrium
    except: return expected_return_atlas, 0.0

# --- E. SCENARIO WAR ROOM (FIXED: BETA ADJUSTED) ---
def run_scenario_stress(hist, current_price, position_value=10000):
    """
    Calculates Beta-adjusted stress scenarios. 
    High Beta stocks will show larger losses than Low Beta stocks.
    """
    try:
        # 1. Calculate Beta vs SPY
        spy = yf.download("SPY", period="1y", progress=False)['Close']
        if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
        spy_ret = spy.pct_change().dropna()
        stock_ret = hist['Close'].pct_change().dropna()
        
        # Align data
        aligned_stock, aligned_spy = stock_ret.align(spy_ret, join='inner')
        
        if len(aligned_stock) > 30:
            covariance = np.cov(aligned_stock, aligned_spy)[0][1]
            variance = np.var(aligned_spy)
            beta = covariance / variance
        else:
            beta = 1.0 # Default if data missing
            
        # Limit Beta to avoid absurd numbers (0.5 to 3.0 range)
        beta = max(0.5, min(beta, 3.0))
        
    except:
        beta = 1.0 # Fallback

    # Base Market Shocks (What SPY would do)
    base_shocks = {
        "OIL SHOCK ($150)": -0.15,       # Market drops 15%
        "FED HAWKISH SURPRISE": -0.05,   # Market drops 5%
        "FLASH CRASH (15 MIN)": -0.09,   # Market drops 9%
        "RECESSION (HARD LANDING)": -0.25 # Market drops 25%
    }
    
    results = {}
    for name, mkt_drop in base_shocks.items():
        # Adjust drop by stock's beta (Stock drop = Market drop * Beta)
        stock_drop = mkt_drop * beta
        results[name] = {
            "price": current_price * (1 + stock_drop), 
            "pnl": position_value * stock_drop,
            "drop_pct": stock_drop * 100
        }
    return results

# --- F. KALMAN FILTER (R=1000) ---
class KalmanFilter:
    def __init__(self, R=1000.0, Q=0.001): 
        self.R = R; self.Q = Q; self.x = None; self.P = 1.0; self.K = None 
    def update(self, measurement):
        if self.x is None: self.x = measurement; return self.x
        self.P = self.P + self.Q
        self.K = self.P / (self.P + self.R)
        self.x = self.x + self.K * (measurement - self.x)
        self.P = (1 - self.K) * self.P
        return self.x

def apply_kalman(series): 
    kf = KalmanFilter(R=1000.0, Q=0.001)
    return [kf.update(x) for x in series]

def get_news_sentiment(ticker_obj):
    try:
        news = ticker_obj.news
        if not news: return 0.0, "NO DATA"
        bullish = ["soar", "jump", "beat", "profit", "growth", "record", "buy", "upgrade", "bull", "surge"]
        bearish = ["plunge", "drop", "miss", "loss", "crash", "sell", "downgrade", "bear", "suit", "investigation"]
        score = 0; count = 0
        for item in news:
            title = item.get('title', '').lower(); count += 1
            for w in bullish: 
                if w in title: score += 1
            for w in bearish: 
                if w in title: score -= 1
        final_score = score / count if count > 0 else 0
        sentiment = "NEUTRAL"
        if final_score > 0.15: sentiment = "POSITIVE"
        if final_score < -0.15: sentiment = "NEGATIVE"
        return final_score, sentiment
    except: return 0.0, "ERROR"

def get_macro_matrix(ticker_returns):
    try:
        macro_tickers = {"USD (UUP)": "UUP", "OIL (USO)": "USO", "BONDS (TLT)": "TLT"}
        sensitivities = {}
        risk_flag = False
        ticker_returns.index = ticker_returns.index.tz_localize(None)
        
        for name, ticker in macro_tickers.items():
            try:
                m_data = yf.download(ticker, period="1y", progress=False)['Close']
                if isinstance(m_data, pd.DataFrame): m_data = m_data.iloc[:, 0]
                m_data.index = m_data.index.tz_localize(None)
                if m_data.empty: continue
                m_ret = m_data.pct_change().dropna()
                stock_aligned, macro_aligned = ticker_returns.align(m_ret, join='inner')
                if len(stock_aligned) > 15:
                    corr = stock_aligned.corr(macro_aligned)
                    sensitivities[name] = corr
                    if abs(corr) > 0.5: risk_flag = True
            except: continue
        if not sensitivities: return {"USD (UUP)": 0.0, "OIL (USO)": 0.0, "BONDS (TLT)": 0.0}, False
        return sensitivities, risk_flag
    except: return {"ERROR": 0.0}, False

def get_derivative_pressure(ticker_obj):
    try:
        exps = ticker_obj.options
        if not exps: return 1.0, "NO OPTIONS", 1.0
        opt = ticker_obj.option_chain(exps[0])
        calls_vol = opt.calls['volume'].sum(); puts_vol = opt.puts['volume'].sum()
        if calls_vol == 0: return 0.0, "BEARISH WALL", 1.0
        pc_ratio = puts_vol / calls_vol
        state = "NEUTRAL"; pressure_mult = 1.0
        if pc_ratio > 1.5: state = "PUT WALL (RESISTANCE)"; pressure_mult = 0.8
        elif pc_ratio < 0.5: state = "GAMMA SQUEEZE (BULLISH)"; pressure_mult = 1.0
        return pc_ratio, state, pressure_mult
    except: return 0.0, "DATA N/A", 1.0

def calc_liquidity_risk(hist, price, position_size_usd=100000):
    avg_vol = hist['Volume'].rolling(20).mean().iloc[-1]
    avg_dollar_vol = avg_vol * price
    participation_rate = 0.10 
    daily_liquidity_avail = avg_dollar_vol * participation_rate
    days_to_exit = position_size_usd / daily_liquidity_avail if daily_liquidity_avail > 0 else 999
    slippage_risk = "LOW"; liquidity_mult = 1.0
    if days_to_exit > 3.0: slippage_risk = "HIGH (ILLIQUID)"; liquidity_mult = 0.5 
    elif days_to_exit > 1.0: slippage_risk = "MEDIUM"; liquidity_mult = 0.8
    return {"days_to_exit": days_to_exit, "daily_dollar_vol": avg_dollar_vol, "slippage_risk": slippage_risk, "mult": liquidity_mult}

def get_factor_exposure(ticker_returns):
    factors = {"MARKET (SPY)": "SPY", "TECH (QQQ)": "QQQ", "SMALL (IWM)": "IWM", "VALUE (DIA)": "DIA"}
    factor_corrs = {}
    try:
        for name, ticker in factors.items():
            try:
                hist = yf.download(ticker, period="1y", progress=False)['Close']
                if isinstance(hist, pd.DataFrame): hist = hist.iloc[:, 0]
                if hist.empty: continue
                factor_rets = hist.pct_change().dropna()
                aligned = pd.concat([ticker_returns, factor_rets], axis=1, join='inner').dropna()
                if not aligned.empty: factor_corrs[name] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            except: continue
        return factor_corrs
    except: return {}

# --- G. OPTIMIZER, BACKTESTER, CFD UTILS ---
def optimize_portfolio(tickers, max_alloc=1.0):
    if not tickers: return None
    try:
        data = yf.download(tickers, period="1y", progress=False)['Close']
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov() * 252
        mean_returns = returns.mean() * 252
        num_assets = len(tickers)
        def neg_sharpe(weights):
            p_ret = np.sum(mean_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(p_ret - 0.045) / p_vol
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, max_alloc) for _ in range(num_assets))
        opt = minimize(neg_sharpe, num_assets*[1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        return {"weights": dict(zip(tickers, opt.x)), "return": np.sum(mean_returns*opt.x), "risk": np.sqrt(np.dot(opt.x.T, np.dot(cov_matrix, opt.x))), "sharpe": -opt.fun}
    except: return None

def run_stress_test(tickers, weights):
    if not tickers: return None
    scenarios = {"2008 GFC": ("2008-09-01", "2008-11-20"), "2020 COVID": ("2020-02-19", "2020-03-23")}
    results = {}
    for name, (start, end) in scenarios.items():
        try:
            data = yf.download(tickers, start=start, end=end, progress=False)['Close']
            if data.empty: results[name] = 0.0; continue
            norm_data = data / data.iloc[0] * 100
            w_list = [weights.get(t, 0) for t in data.columns]
            port_val = norm_data.dot(w_list)
            results[name] = (port_val.min() - 100)
        except: results[name] = 0.0
    return results

def run_backtest(ticker, start_date, capital):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date)
        if hist.empty: return None
        entry = hist['Close'].iloc[0]; curr = hist['Close'].iloc[-1]
        val = (capital / entry) * curr
        curve = (capital / entry) * hist['Close']
        ret = hist['Close'].pct_change().dropna()
        mu = ret.mean()*252; sig = ret.std()*np.sqrt(252)
        paths = np.zeros((100, 252)); paths[:, 0] = val
        for i in range(100):
            paths[i, 1:] = val * np.exp(np.cumsum((mu - 0.5 * sig**2)/252 + sig/np.sqrt(252) * np.random.normal(0, 1, 251)))
        return {"val": val, "ret": ((val-capital)/capital)*100, "curve": curve, "mc_future": paths}
    except: return None

def calc_cfd_metrics(entry, cap, lev, s_long, s_short, days, div, is_short):
    pos = cap * lev
    swap = (pos * (s_short if is_short else s_long)) / 365 * days
    div_imp = (pos * div / 365) * days * (-1 if is_short else 1) if div > 0 else 0
    liq = entry * (1 + (1/lev)) if is_short else entry * (1 - (1/lev))
    return {"Pos": pos, "Swap": swap, "Div": div_imp, "Liq": liq, "Dist": abs(entry-liq)/entry*100}

# --- H. MASTER ENGINE (GET_FULL_ANALYSIS) ---
def get_full_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: return None
        
        info = stock.info
        price = hist['Close'].iloc[-1]
        ret = hist['Close'].pct_change().dropna()
        sigma = ret.std() * np.sqrt(252)
        
        # --- MODULES ---
        macro = get_macro_context()
        gmm_name, gmm_mult, gmm_color = get_market_regime_gmm()
        macro_sens, macro_risk = get_macro_matrix(ret)
        pc_ratio, gamma_state, gamma_mult = get_derivative_pressure(stock)
        sent_score, sent_label = get_news_sentiment(stock)
        liq_data = calc_liquidity_risk(hist, price)
        vol_context = get_volatility_context(ret) # OMEGA
        war_room = run_scenario_stress(hist, price, 10000) # OMEGA BETA FIX
        
        kalman = apply_kalman(hist['Close'].values)
        factors = get_factor_exposure(ret)
        
        # --- ANTI-BUBBLE Z-SCORE CHECK (FIXING HIGH CONVICTION BIAS) ---
        # 1. Calculate how far Price is from Kalman Trend
        residual = price - kalman[-1]
        resid_std = (hist['Close'] - kalman).std()
        z_resid = residual / resid_std if resid_std > 0 else 0
        
        # 2. Bubble Penalty
        bubble_penalty = 1.0
        if z_resid > 2.0: bubble_penalty = 0.5 # Overbought: Cut size in half
        if z_resid > 3.0: bubble_penalty = 0.1 # Bubble: Kill trade
        if z_resid < -2.0: bubble_penalty = 0.8 # Oversold: Caution
        
        # --- SOVEREIGN MULTIPLIER 3.1 ---
        w_regime = 1.5; w_macro = 1.0; w_gamma = 0.5
        f_regime = gmm_mult if gmm_mult is not None else 1.0
        f_macro = 0.8 if macro_risk else 1.0
        f_gamma = gamma_mult
        f_sent = 0.5 if sent_label == "NEGATIVE" else 1.0
        f_liq = liq_data['mult']
        f_garch = vol_context['penalty'] 
        
        confidence_lambda = 1.0
        if liq_data['slippage_risk'] == "HIGH": confidence_lambda = 0.6
        if sent_label == "NO DATA": confidence_lambda = 0.8
        
        if macro['multiplier'] == 0.0: f_regime = 0.0
        
        # Multiply everything, including new Bubble Penalty
        sovereign_mult = (f_regime**w_regime) * (f_macro**w_macro) * (f_gamma**w_gamma) * f_sent * f_liq * f_garch * bubble_penalty * confidence_lambda
        
        mu = ret.mean()*252
        bl_return, market_eq = black_litterman_lite(mu, confidence=confidence_lambda)
        
        k_raw = (bl_return - 0.045) / (sigma**2)
        base_kelly = max(0, min(k_raw * 0.25, 0.50)) * 100
        final_size = base_kelly * sovereign_mult
        
        # STRICTER VERDICTS
        verdict = "NEUTRAL"
        v_color = "sov-yellow"
        
        if sovereign_mult == 0.0: 
            verdict = "HARD STOP (CRISIS/EXIT)"
            v_color = "sov-red"
        elif final_size > 20.0 and z_resid < 1.5: # Only High Conviction if NOT overbought
            verdict = "HIGH CONVICTION BUY"
            v_color = "sov-green"
        elif final_size > 5.0: 
            verdict = "ACCUMULATE"
            v_color = "sov-green"
        elif z_resid > 2.0:
            verdict = "OVERBOUGHT (WAIT)"
            v_color = "sov-yellow"
        else: 
            verdict = "WATCHLIST ONLY (RISK)"
            v_color = "sov-red"
        
        # MC RESTORED
        dt = 1/252; sims = 300 
        paths = np.zeros((sims, 253)); paths[:, 0] = price
        for i in range(sims):
            shocks = np.random.normal(0, 1, 252)
            paths[i, 1:] = price * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks))

        d1 = (np.log(price/price)+(0.045+0.5*sigma**2))/(sigma)
        delta = norm.cdf(d1)
        theta = (-price*norm.pdf(d1)*sigma/2)/365
        gamma = norm.pdf(d1)/(price*sigma)

        return {
            "symbol": ticker, "price": price, "sigma": sigma, "z": z_resid,
            "kalman": kalman, "hist": hist, "factors": factors,
            "macro": macro, "regime": {"name": gmm_name, "color": gmm_color},
            "macro_matrix": macro_sens,
            "risk": {"pc_ratio": pc_ratio, "gamma_state": gamma_state},
            "sentiment": {"label": sent_label, "score": sent_score},
            "liquidity": liq_data,
            "vol_context": vol_context, 
            "war_room": war_room, 
            "bl": {"return": bl_return, "market": market_eq},
            "size": {"base": base_kelly, "final": final_size, "mult": sovereign_mult, "lambda": confidence_lambda},
            "verdict": {"text": verdict, "color": v_color},
            "greeks": {"delta": delta, "gamma": gamma, "theta": theta},
            "mc": paths
        }
    except Exception as e: return None

# ---------------------------------------------------------
# 4. UI LAYOUT (FULL UI RESTORED)
# ---------------------------------------------------------
macro_data = get_macro_context()

st.sidebar.markdown(f"### {FIRM_NAME}")
st.sidebar.markdown(f"<div style='color:#666; font-size:10px; margin-bottom:15px;'>{SYSTEM_VERSION}</div>", unsafe_allow_html=True)
st.sidebar.markdown("**GLOBAL MACRO PULSE**")

m_col = "#3fb950" if macro_data['multiplier'] == 1.0 else "#da3633"
st.sidebar.markdown(f"<div style='color:{m_col}; font-weight:bold; margin-bottom:10px;'>{macro_data['state']}</div>", unsafe_allow_html=True)
st.sidebar.metric("YIELD SPREAD (10Y-13W)", f"{macro_data['spread']:.3f}%")
st.sidebar.metric("VIX (FEAR INDEX)", f"{macro_data['vix']:.2f}", macro_data['vix_status'])
st.sidebar.progress(float(macro_data['multiplier']))

menu = st.sidebar.radio("MODULES", ["QUANTUM LAB", "ALADDIN PORTFOLIO", "CFD WAR ROOM", "TIME MACHINE", "DEAL MECHANICS", "COMPLIANCE"])

if menu == "QUANTUM LAB":
    st.markdown("## SOVEREIGN QUANTUM CORE")
    c1, c2 = st.columns([3, 1])
    t = c1.text_input("ASSET TICKER", "NVDA").upper()
    if c2.button("INITIATE OMEGA SCAN"):
        with st.spinner("Analyzing Volatility Clusters & Black-Litterman View..."):
            data = get_full_analysis(t)
        
        if data:
            v_color = data['verdict']['color']
            st.markdown(f"<div class='sov-banner {v_color}'>{data['verdict']['text']} | SIZE: {data['size']['final']:.1f}%</div>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("REGIME (GMM)", "DETECTED", data['regime']['name'])
            m2.metric("BUBBLE SCORE (Z)", f"{data['z']:.2f}", "Overbought" if data['z']>2.0 else "Safe")
            m3.metric("CONFIDENCE (LAMBDA)", f"{data['size']['lambda']:.2f}", "Data Quality")
            m4.metric("BLACK-LITTERMAN RET", f"{data['bl']['return']*100:.1f}%", f"Mkt: {data['bl']['market']*100:.1f}%")

            t1, t2, t3, t4 = st.tabs(["CHART & KALMAN", "WAR ROOM (SCENARIOS)", "MACRO MATRIX", "RISK & GREEKS"])
            
            with t1:
                st.markdown("<div class='ibkr-header'>KALMAN FILTER DEVIATION (R=1000)</div>", unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data['hist'].index, open=data['hist']['Open'], high=data['hist']['High'], low=data['hist']['Low'], close=data['hist']['Close'], name='Price'))
                # NEON YELLOW LINE
                fig.add_trace(go.Scatter(x=data['hist'].index, y=data['kalman'], line=dict(color='#FFFF00', width=3), name='Kalman Trend'))
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<div class='ibkr-header'>FUTURE PATHWAYS (MONTE CARLO)</div>", unsafe_allow_html=True)
                fig_mc = go.Figure()
                for i in range(50): fig_mc.add_trace(go.Scatter(y=data['mc'][i], mode='lines', line=dict(color='rgba(0, 255, 255, 0.1)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=np.mean(data['mc'], axis=0), line=dict(color='#FFFFFF', width=4), name='Mean'))
                fig_mc.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_mc, use_container_width=True)
            
            with t2:
                st.markdown("### SCENARIO WAR ROOM (BETA ADJUSTED)")
                st.caption("Scenarios now adjusted by stock's sensitivity (Beta) to the market.")
                cols = st.columns(4)
                idx = 0
                for k, v in data['war_room'].items():
                    with cols[idx]:
                        st.metric(k, f"${v['price']:.2f}", f"PnL: ${v['pnl']:.0f}")
                    idx += 1
            
            with t3:
                st.markdown("### MACRO SENSITIVITY")
                if data['macro_matrix']:
                    st.markdown("<table class='aladdin-table'><tr><th>FACTOR</th><th>CORRELATION</th></tr>", unsafe_allow_html=True)
                    for k, v in data['macro_matrix'].items():
                        risk_class = "risk-high" if abs(v) > 0.5 else "risk-low"
                        st.markdown(f"<tr><td>{k}</td><td class='{risk_class}'>{v:.2f}</td></tr>", unsafe_allow_html=True)
                    st.markdown("</table>", unsafe_allow_html=True)
                st.markdown("### FACTOR DNA")
                if data['factors']:
                    cols = st.columns(len(data['factors']))
                    for i, (k, v) in enumerate(data['factors'].items()):
                        cols[i].metric(k, f"{v:.2f}")

            with t4:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<div class='ibkr-header'>POSITION SIZING</div>", unsafe_allow_html=True)
                    st.metric("FINAL SOVEREIGN SIZE", f"{data['size']['final']:.1f}%")
                    st.caption(f"Includes Bubble Penalty (Z-Score)")
                    st.metric("NEWS SENTIMENT", data['sentiment']['label'], f"{data['sentiment']['score']:.2f}")
                with c2:
                    st.markdown("<div class='ibkr-header'>OPTION GREEKS</div>", unsafe_allow_html=True)
                    g1, g2, g3 = st.columns(3)
                    g1.metric("DELTA", f"{data['greeks']['delta']:.2f}")
                    g2.metric("THETA", f"{data['greeks']['theta']:.3f}")
                    g3.metric("GAMMA", f"{data['greeks']['gamma']:.4f}")

elif menu == "ALADDIN PORTFOLIO":
    st.markdown("## PORTFOLIO FORTRESS")
    c1, c2 = st.columns([3, 1])
    tickers = c1.text_input("ASSETS", "NVDA, AAPL, MSFT, GLD, TLT")
    max_alloc = c2.slider("MAX ALLOC", 0.1, 1.0, 0.25)
    
    if st.button("BUILD PORTFOLIO"):
        res = optimize_portfolio([x.strip() for x in tickers.split(',')], max_alloc)
        stress = run_stress_test([x.strip() for x in tickers.split(',')], res['weights'] if res else {})
        
        if res:
            m1, m2, m3 = st.columns(3)
            m1.metric("EXP. RETURN", f"{res['return']*100:.2f}%")
            m2.metric("PORTFOLIO RISK", f"{res['risk']*100:.2f}%")
            m3.metric("SHARPE RATIO", f"{res['sharpe']:.2f}")
            
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown("### OPTIMAL WEIGHTS")
                st.plotly_chart(px.pie(values=list(res['weights'].values()), names=list(res['weights'].keys()), hole=0.6, template="plotly_dark"), use_container_width=True)
            with c_p2:
                st.markdown("### STRESS TEST (HISTORICAL CRASHES)")
                for k, v in stress.items():
                    st.metric(k, f"{v:.2f}%", help=f"Drawdown during {k}")

elif menu == "CFD WAR ROOM":
    st.markdown("## CFD WAR ROOM")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("<div class='ibkr-card'>", unsafe_allow_html=True)
        t = st.text_input("TICKER", "TSLA").upper()
        cap = st.number_input("MARGIN ($)", 100.0)
        lev = st.slider("LEVERAGE", 1, 50, 10)
        side = st.radio("SIDE", ["LONG", "SHORT"], horizontal=True)
        days = st.number_input("DAYS", 7)
        st.markdown("</div>", unsafe_allow_html=True)
    if t:
        d = get_full_analysis(t)
        if d:
            res = calc_cfd_metrics(d['price'], cap, lev, 0.05, 0.02, days, 0.0, side=="SHORT")
            with c2:
                m1, m2 = st.columns(2)
                m1.metric("LIQUIDATION PRICE", f"${res['Liq']:.2f}", delta=f"{res['Dist']:.2f}% Away", delta_color="inverse")
                m2.metric("TOTAL EXPOSURE", f"${res['Pos']:,.0f}")
                st.markdown("<div class='ibkr-card'>", unsafe_allow_html=True)
                c_1, c_2 = st.columns(2)
                c_1.metric("SWAP FEES", f"${res['Swap']:.2f}")
                c_2.metric("NET P&L DRAG", f"${res['Swap']+res['Div']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

elif menu == "TIME MACHINE":
    st.markdown("## TIME MACHINE")
    c1, c2, c3 = st.columns(3)
    t = c1.text_input("TICKER", "AAPL").upper()
    d = c2.date_input("START", datetime.date(2022, 1, 1))
    c = c3.number_input("CAPITAL", 10000.0)
    if st.button("SIMULATE"):
        res = run_backtest(t, d, c)
        if res:
            st.metric("TOTAL ROI", f"{res['ret']:.2f}%", f"${res['val']-c:,.0f}")
            st.line_chart(res['curve'])
            st.markdown("### FUTURE PROJECTION")
            st.line_chart(res['mc_future'].T)

elif menu == "DEAL MECHANICS":
    st.markdown("## DEAL MECHANICS")
    c1, c2 = st.columns(2)
    c1.metric("PPP (Profit Per Partner)", f"${(st.number_input('REV', 1e7)*0.4)/st.number_input('PARTNERS', 5):,.0f}")
    c2.metric("GP CARRY", f"${(st.number_input('EXIT', 2e7)-st.number_input('FUND', 1e7))*0.2:,.0f}")

elif menu == "COMPLIANCE":
    st.markdown("## KYC REGISTRY")
    with st.form("kyc"):
        if st.form_submit_button("REGISTER ENTITY"): st.success("ENTITY REGISTERED")
