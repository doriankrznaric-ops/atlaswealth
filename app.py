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

# ---------------------------------------------------------
# 1. SYSTEM CONFIGURATION
# ---------------------------------------------------------
FIRM_NAME = "ATLAS QUANTUM PRIVATE WEALTH"
SYSTEM_VERSION = "SENTINEL CORE V39.0 UNCHAINED"

st.set_page_config(page_title=FIRM_NAME, layout="wide")

# ---------------------------------------------------------
# 2. DESIGN: "THE ARCHITECT" (PRO UI, NO EMOJIS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* GLOBAL RESET */
    .stApp { background-color: #000000; color: #e5e5e5; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #1c1c1e; }
    
    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        padding: 15px;
        border-radius: 4px;
        border: 1px solid #2c2c2e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        transition: border 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #555;
    }
    div[data-testid="stMetricLabel"] { font-size: 10px; color: #888; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #fff; font-family: 'SF Mono', 'Consolas', monospace; font-weight: 400; }
    div[data-testid="stMetricDelta"] { font-size: 12px; font-weight: 600; font-family: monospace; }

    /* SIGNAL BANNERS */
    .signal-container { 
        padding: 15px; 
        border-radius: 4px; 
        margin-bottom: 25px; 
        text-align: center; 
        font-weight: 800; 
        letter-spacing: 3px; 
        text-transform: uppercase; 
        font-size: 16px; 
        border-left: 5px solid;
        background-color: #0a0a0a;
    }
    .sig-buy { background: rgba(6, 78, 59, 0.2); border-color: #10b981; color: #34d399; border: 1px solid #10b981; border-left-width: 5px; }
    .sig-sell { background: rgba(127, 29, 29, 0.2); border-color: #ef4444; color: #f87171; border: 1px solid #ef4444; border-left-width: 5px; }
    .sig-neut { background: #18181b; border-color: #71717a; color: #a1a1aa; border: 1px solid #71717a; border-left-width: 5px; }
    .sig-crash { background: #450a0a; border-color: #ff0000; color: #ff0000; border: 2px solid #ff0000; animation: pulse 2s infinite; }
    
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }

    /* CUSTOM CARDS */
    .ibkr-card { background-color: #0e0e0e; border: 1px solid #27272a; border-radius: 4px; padding: 20px; margin-bottom: 20px; }
    .ibkr-header { font-size: 11px; color: #71717a; font-weight: 700; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #27272a; padding-bottom: 5px; }
    
    /* INPUTS & BUTTONS */
    .stTextInput>div>div>input { background-color: #000; color: #fff; border: 1px solid #333; border-radius: 4px; height: 45px; font-family: 'SF Mono', 'Consolas', monospace; }
    .stNumberInput>div>div>input { background-color: #000; color: #fff; border: 1px solid #333; border-radius: 4px; }
    
    .stButton>button { 
        background-color: #18181b; 
        color: #e4e4e7; 
        border: 1px solid #3f3f46; 
        font-weight: 700; 
        border-radius: 4px; 
        height: 45px; 
        width: 100%; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
        transition: 0.2s; 
    }
    .stButton>button:hover { 
        background-color: #27272a; 
        border-color: #fff; 
        color: #fff; 
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; margin-bottom: 20px; }
    .stTabs [data-baseweb="tab"] { height: 40px; background-color: #09090b; border: 1px solid #27272a; color: #71717a; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #18181b; color: #fff; border-bottom: 2px solid #fff; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. MATHEMATICAL ENGINES
# ---------------------------------------------------------

# --- A. REGIME SWITCHING (LOOSENED) ---
def get_market_regime():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist.empty: return "UNKNOWN", 0
        current_vix = hist['Close'].iloc[-1]
        
        # INCREASED THRESHOLD FOR CRASH TO 40
        if current_vix > 40.0: return "CRASH", current_vix 
        elif current_vix > 25.0: return "VOLATILE", current_vix
        else: return "CALM", current_vix
    except: return "UNKNOWN", 0

# --- B. KALMAN FILTER (R=50) ---
class KalmanFilter:
    def __init__(self, R=50.0, Q=0.0001): 
        self.R = R 
        self.Q = Q 
        self.x = None 
        self.P = 1.0 
        self.K = None 
    def update(self, measurement):
        if self.x is None: 
            self.x = measurement
            return self.x
        self.P = self.P + self.Q
        self.K = self.P / (self.P + self.R)
        self.x = self.x + self.K * (measurement - self.x)
        self.P = (1 - self.K) * self.P
        return self.x

def apply_kalman(series): 
    kf = KalmanFilter()
    return [kf.update(x) for x in series]

# --- C. FACTOR EXPOSURE ---
def get_factor_exposure(ticker_returns):
    factors = {"MARKET (SPY)": "SPY", "TECH (QQQ)": "QQQ", "SMALL (IWM)": "IWM", "VALUE (DIA)": "DIA"}
    factor_corrs = {}
    try:
        tickers = list(factors.values())
        factor_data = yf.download(tickers, period="1y", progress=False)['Close']
        factor_rets = factor_data.pct_change().dropna()
        aligned = pd.concat([ticker_returns, factor_rets], axis=1, join='inner').dropna()
        if not aligned.empty:
            target = aligned.iloc[:, 0]
            for name, ticker in factors.items():
                if ticker in aligned.columns: factor_corrs[name] = target.corr(aligned[ticker])
        return factor_corrs
    except: return {}

# --- D. STRESS TESTING ---
def run_stress_test(tickers, weights):
    if not tickers: return None
    scenarios = {
        "2008 GFC": ("2008-09-01", "2008-11-20"),
        "2020 COVID": ("2020-02-19", "2020-03-23"),
        "2022 TECH": ("2021-12-27", "2022-06-16")
    }
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

# --- E. OPTIMIZER ---
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
        return {"weights": dict(zip(tickers, opt.x)), "return": np.sum(mean_returns*opt.x), "risk": np.sqrt(np.dot(opt.x.T, np.dot(cov_matrix, opt.x))), "sharpe": -opt.fun, "corr": returns.corr()}
    except: return None

# --- F. RISK & ALGOS ---
def get_hurst_exponent(ts):
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    if len(tau) < 2: return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0 

def cornish_fisher_var(ret, conf=0.95):
    z = norm.ppf(1 - conf)
    s = skew(ret); k = kurtosis(ret)
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * (s**2) / 36
    return ret.mean() + z_cf * ret.std()

# --- G. CFD CALCULATOR ---
def calc_cfd_metrics(entry, cap, lev, s_long, s_short, days, div, is_short):
    pos = cap * lev
    swap = (pos * (s_short if is_short else s_long)) / 365 * days
    div_imp = (pos * div / 365) * days * (-1 if is_short else 1) if div > 0 else 0
    liq = entry * (1 + (1/lev)) if is_short else entry * (1 - (1/lev))
    return {"Pos": pos, "Swap": swap, "Div": div_imp, "Liq": liq, "Dist": abs(entry-liq)/entry*100}

# --- H. MASTER ANALYSIS ENGINE (UNCHAINED) ---
def get_full_analysis(ticker, external_regime="CALM"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: return None
        info = stock.info
        price = hist['Close'].iloc[-1]
        ret = hist['Close'].pct_change().dropna()
        sigma = ret.std() * np.sqrt(252)
        
        # Algos
        hurst = get_hurst_exponent(hist['Close'].values)
        kalman = apply_kalman(hist['Close'].values)
        cf_var = cornish_fisher_var(ret)
        factors = get_factor_exposure(ret)
        
        # INTERNAL REGIME (NO KILL SWITCH)
        residuals = hist['Close'] - kalman
        std_resid = residuals.rolling(window=50).std().iloc[-1]
        current_resid = abs(residuals.iloc[-1])
        
        internal_regime = "CALM"
        if current_resid > (4.0 * std_resid): internal_regime = "EXTREME" # Was CRASH
        elif current_resid > (2.0 * std_resid): internal_regime = "VOLATILE"
        
        # FINAL LOGIC
        final_regime = "CALM"
        regime_penalty = 1.0
        
        # ONLY EXTERNAL VIX CAN KILL THE TRADE
        if external_regime == "CRASH":
            final_regime = "CRASH"
            regime_penalty = 0.0 
        elif internal_regime == "EXTREME":
            final_regime = "EXTREME INTERNAL"
            regime_penalty = 0.2 # 20% size, not 0%
        elif external_regime == "VOLATILE" or internal_regime == "VOLATILE":
            final_regime = "VOLATILE"
            regime_penalty = 0.5
            
        # Monte Carlo
        dt = 1/252; N = 252; sims = 1000
        paths = np.zeros((sims, N + 1)); paths[:, 0] = price
        mu_hist = ret.mean() * 252
        for i in range(sims):
            shocks = np.random.normal(0, 1, N)
            paths[i, 1:] = price * np.exp(np.cumsum((mu_hist - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks))
        
        # Liquidity Check
        avg_vol = hist['Volume'].rolling(20).mean().iloc[-1] * price
        liq_pen = 0.5 if avg_vol < 10000000 else 1.0
        if avg_vol < 1000000: liq_pen = 0.1
        
        # Smart Kelly 2.0
        mc_ret = (np.mean(paths[:, -1]) - price) / price
        k_raw = (mc_ret - 0.045) / (sigma**2)
        kelly = max(0, min(k_raw * 0.25, 0.50)) * 100 * liq_pen * regime_penalty
        
        # Greeks & Z
        d1 = (np.log(price/price)+(0.045+0.5*sigma**2))/(sigma)
        delta = norm.cdf(d1)
        theta = (-price*norm.pdf(d1)*sigma/2 - 0.045*price*np.exp(-0.045)*norm.cdf(d1-sigma))/365
        z = (price - hist['Close'].rolling(200).mean().iloc[-1]) / hist['Close'].rolling(200).std().iloc[-1]

        return {
            "symbol": ticker, "name": info.get('longName', ticker), "price": price, 
            "div": info.get('dividendYield', 0), "sigma": sigma, "z": z, 
            "hurst": hurst, "kalman": kalman, "var": cf_var, "kelly": kelly,
            "delta": delta, "theta": theta, "gamma": norm.pdf(d1)/(price*sigma), 
            "mc": paths, "hist": hist, "factors": factors, "regime": final_regime, 
            "resid_z": current_resid/std_resid if std_resid else 0, "liq_pen": liq_pen
        }
    except: return None

# --- I. BACKTESTER ---
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

# ---------------------------------------------------------
# 4. DATABASE
# ---------------------------------------------------------
DB_FILE = "atlas_v39.json"
def load_db(): 
    if not os.path.exists(DB_FILE): return {}
    with open(DB_FILE, "r") as f: return json.load(f)
def save_db(data): 
    with open(DB_FILE, "w") as f: json.dump(data, f, indent=4)
db = load_db()

# ---------------------------------------------------------
# 5. UI LAYOUT
# ---------------------------------------------------------
ext_regime, vix = get_market_regime()

st.sidebar.markdown(f"### {FIRM_NAME}")
st.sidebar.markdown(f"<div style='color:#666; font-size:10px; margin-bottom:15px;'>{SYSTEM_VERSION}</div>", unsafe_allow_html=True)
st.sidebar.markdown("**SENTINEL STATUS**")
if ext_regime == "CRASH": st.sidebar.error(f"CRASH (VIX {vix:.2f})")
elif ext_regime == "VOLATILE": st.sidebar.warning(f"VOLATILE (VIX {vix:.2f})")
else: st.sidebar.success(f"CALM (VIX {vix:.2f})")

menu = st.sidebar.radio("MODULES", ["QUANTUM LAB", "ALADDIN PORTFOLIO", "CFD WAR ROOM", "TIME MACHINE", "DEAL MECHANICS", "COMPLIANCE"])

# --- MODULE A: QUANTUM LAB ---
if menu == "QUANTUM LAB":
    st.markdown("## ATLAS QUANTUM CORE")
    
    c1, c2 = st.columns([3, 1])
    t = c1.text_input("ASSET TICKER", "NVDA").upper()
    if c2.button("INITIATE SCAN"):
        with st.spinner("Processing Algorithmic & Regime Models..."):
            data = get_full_analysis(t, ext_regime)
        
        if data:
            # SIGNAL BANNER
            regime = data['regime']
            
            if regime == "CRASH":
                st.markdown(f"<div class='signal-container sig-crash'>TRADING HALTED | MARKET CRASH (VIX)</div>", unsafe_allow_html=True)
            elif regime == "EXTREME INTERNAL":
                st.markdown(f"<div class='signal-container sig-neut'>CAUTION | EXTREME DEVIATION (Z-RESID: {data['resid_z']:.2f})</div>", unsafe_allow_html=True)
            elif data['z'] < -2.0:
                st.markdown(f"<div class='signal-container sig-buy'>STRONG BUY DETECTED | Z-SCORE: {data['z']:.2f}</div>", unsafe_allow_html=True)
            elif data['z'] > 2.0:
                st.markdown(f"<div class='signal-container sig-sell'>STRONG SELL DETECTED | Z-SCORE: {data['z']:.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='signal-container sig-neut'>NEUTRAL / HOLD | Z-SCORE: {data['z']:.2f}</div>", unsafe_allow_html=True)

            # METRICS GRID
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SPOT PRICE", f"${data['price']:.2f}", help="Current market execution price.")
            
            k_delta = data['price'] - data['kalman'][-1]
            m2.metric("KALMAN FAIR VALUE", f"${data['kalman'][-1]:.2f}", f"{k_delta:.2f}", delta_color="inverse", help="Noise-filtered trend value (R=50).")
            
            h_state = "TRENDING" if data['hurst'] > 0.55 else "MEAN REV"
            m3.metric("HURST EXPONENT", f"{data['hurst']:.2f}", h_state, help=">0.5 = Trending, <0.5 = Mean Reverting.")
            
            m4.metric("1-DAY VaR (95%)", f"{data['var']*100:.2f}%", help="Maximum expected loss in 1 day with 95% confidence.")

            # TABS
            t1, t2, t3 = st.tabs(["CHART ANALYSIS", "MONTE CARLO", "RISK MATRIX"])
            
            with t1:
                st.markdown("<div class='ibkr-header'>KALMAN FILTER DEVIATION</div>", unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data['hist'].index, open=data['hist']['Open'], high=data['hist']['High'], low=data['hist']['Low'], close=data['hist']['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=data['hist'].index, y=data['kalman'], line=dict(color='#fbbf24', width=2), name='Kalman Trend'))
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with t2:
                st.markdown("<div class='ibkr-header'>1000 FUTURE PATHWAYS</div>", unsafe_allow_html=True)
                c_mc1, c_mc2, c_mc3 = st.columns(3)
                c_mc1.metric("BEAR CASE (5%)", f"${np.percentile(data['mc'][:,-1], 5):.2f}", help="Worst 5% outcome.")
                c_mc2.metric("MEAN EXPECTATION", f"${np.mean(data['mc'][:,-1]):.2f}", help="Average outcome.")
                c_mc3.metric("BULL CASE (95%)", f"${np.percentile(data['mc'][:,-1], 95):.2f}", help="Best 5% outcome.")
                
                fig_mc = go.Figure()
                for i in range(50): fig_mc.add_trace(go.Scatter(y=data['mc'][i], mode='lines', line=dict(color='rgba(74, 222, 128, 0.05)'), showlegend=False))
                fig_mc.add_trace(go.Scatter(y=np.mean(data['mc'], axis=0), line=dict(color='#4ade80', width=3), name='Mean'))
                fig_mc.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_mc, use_container_width=True)
                
            with t3:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<div class='ibkr-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='ibkr-header'>POSITION SIZING (KELLY)</div>", unsafe_allow_html=True)
                    k_val = data['kelly']
                    st.metric("QUARTER KELLY", f"{k_val:.1f}%", help="Recommended size (Fractional 0.25).")
                    if data['liq_pen'] < 1.0: st.caption(f"LIQUIDITY PENALTY APPLIED")
                    
                    if regime == "CRASH":
                        st.caption("⛔ REGIME PENALTY: MARKET CRASH (SIZE = 0)")
                    elif regime == "EXTREME INTERNAL":
                        st.caption("⚠️ REGIME PENALTY: EXTREME DEVIATION (SIZE = 20%)")
                    elif regime == "VOLATILE":
                        st.caption("⚠️ REGIME PENALTY: VOLATILE (SIZE HALVED)")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with c2:
                    st.markdown("<div class='ibkr-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='ibkr-header'>OPTION GREEKS</div>", unsafe_allow_html=True)
                    g1, g2, g3 = st.columns(3)
                    g1.metric("DELTA", f"{data['delta']:.2f}", help="Sensitivity to price.")
                    g2.metric("THETA", f"{data['theta']:.3f}", help="Time decay.")
                    g3.metric("GAMMA", f"{data['gamma']:.4f}", help="Acceleration.")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("### FACTOR DNA")
                if data['factors']:
                    cols = st.columns(len(data['factors']))
                    for i, (k, v) in enumerate(data['factors'].items()):
                        cols[i].metric(k, f"{v:.2f}", help=f"Correlation with {k}")

# --- MODULE B: ALADDIN PORTFOLIO ---
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

# --- MODULE C: CFD WAR ROOM ---
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
            res = calc_cfd_metrics(d['price'], cap, lev, 0.05, 0.02, days, d['div'], side=="SHORT")
            with c2:
                m1, m2 = st.columns(2)
                m1.metric("LIQUIDATION PRICE", f"${res['Liq']:.2f}", delta=f"{res['Dist']:.2f}% Away", delta_color="inverse", help="Price at which you lose everything.")
                m2.metric("TOTAL EXPOSURE", f"${res['Pos']:,.0f}", help="Total position value.")
                
                st.markdown("<div class='ibkr-card'>", unsafe_allow_html=True)
                st.markdown("### COST ANALYSIS")
                c_1, c_2 = st.columns(2)
                c_1.metric("SWAP FEES", f"${res['Swap']:.2f}", help="Overnight interest.")
                c_2.metric("DIVIDEND ADJ.", f"${res['Div']:.2f}", help="Dividend payment (if Short) or credit (if Long).")
                st.divider()
                st.metric("NET P&L DRAG", f"${res['Swap']+res['Div']:.2f}", help="Total cost to hold.")
                st.markdown("</div>", unsafe_allow_html=True)

# --- MODULE D: TIME MACHINE ---
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

# --- MODULE E: DEAL & COMPLIANCE ---
elif menu == "DEAL MECHANICS":
    st.markdown("## DEAL MECHANICS")
    c1, c2 = st.columns(2)
    c1.metric("PPP (Profit Per Partner)", f"${(st.number_input('REV', 1e7)*0.4)/st.number_input('PARTNERS', 5):,.0f}")
    c2.metric("GP CARRY", f"${(st.number_input('EXIT', 2e7)-st.number_input('FUND', 1e7))*0.2:,.0f}")

elif menu == "COMPLIANCE":
    st.markdown("## KYC REGISTRY")
    with st.form("kyc"):
        if st.form_submit_button("REGISTER ENTITY"): st.success("ENTITY REGISTERED")