"""
GreenLeaf CEA — Precision Agriculture Intelligence App
=======================================================
A Streamlit app for the SFU Beedie Business Analytics Hackathon 2026.

Run: streamlit run app.py
Requires: OPENAI_API_KEY env var for the Ask GreenLeaf page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GreenLeaf CEA — Decision Brief",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color palette
C = {
    'green': '#2d5f4e',
    'green_light': '#4d8870',
    'red': '#c84a3e',
    'gold': '#b88a2e',
    'ink': '#1a1a1a',
    'muted': '#595950',
    'bg': '#faf8f4',
    'panel': '#ffffff',
    'rule': '#d9d6cc',
    'blue': '#3a6b8c',
}

TREATMENT_COLORS = {
    'Control': '#888888',
    'High Light': '#2d5f4e',
    'High N': '#4d8870',
    'Integrated Pest': '#3a6b8c',
    'Low N': '#b88a2e',
    'Reduced Pest': '#c87a3e',
    'Shade': '#c84a3e',
}

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
    
    .stApp { background-color: #faf8f4; }
    
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    
    h1 { font-family: 'DM Serif Display', serif !important; color: #1a1a1a; font-size: 2.2rem !important; }
    h2 { font-family: 'DM Serif Display', serif !important; color: #1a1a1a; font-size: 1.5rem !important; }
    h3 { font-family: 'DM Serif Display', serif !important; color: #2d5f4e; font-size: 1.15rem !important; }
    
    /* KPI card styling */
    .kpi-card {
        background: white;
        border: 1px solid #d9d6cc;
        border-radius: 8px;
        padding: 18px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .kpi-card.accent {
        border-color: #2d5f4e;
        border-width: 2px;
        background: #f0f7f4;
    }
    .kpi-card.warn {
        border-color: #c84a3e;
        border-width: 2px;
        background: #fdf4f3;
    }
    .kpi-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #595950;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'DM Serif Display', serif;
        font-size: 32px;
        font-weight: 400;
        color: #1a1a1a;
        line-height: 1.1;
    }
    .kpi-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        color: #595950;
        margin-top: 6px;
        line-height: 1.4;
    }
    
    /* Story banner */
    .story-banner {
        background: white;
        border-left: 4px solid #2d5f4e;
        padding: 18px 24px;
        margin: 12px 0 20px;
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        line-height: 1.65;
        color: #1a1a1a;
        border-radius: 0 8px 8px 0;
    }
    .story-banner strong { color: #2d5f4e; }
    
    /* Perspective banner */
    .rbc-banner {
        background: #1a3a5c;
        color: white;
        padding: 14px 24px;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .rbc-banner strong { color: #7eb8e0; }
    .farmer-banner {
        background: #2d5f4e;
        color: white;
        padding: 14px 24px;
        border-radius: 8px;
        font-size: 15px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .farmer-banner strong { color: #a8dbc5; }
    
    /* Honest limits banner */
    .limits-banner {
        background: #1a1a1a;
        color: #faf8f4;
        padding: 18px 24px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.65;
        margin-top: 20px;
    }
    .limits-banner strong { color: #c84a3e; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { background: #1a1a1a; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #faf8f4 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        font-family: 'DM Sans', sans-serif !important; 
        font-size: 15px !important; 
        font-weight: 600 !important;
        padding: 10px 24px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    from pathlib import Path
    DATA = str(Path(__file__).parent / 'data')
    farms = pd.read_csv(f'{DATA}/farms.csv')
    greenhouses = pd.read_csv(f'{DATA}/greenhouses.csv')
    plots = pd.read_csv(f'{DATA}/plots.csv')
    sensor = pd.read_csv(f'{DATA}/daily_sensor_readings.csv', parse_dates=['date'])
    costs = pd.read_csv(f'{DATA}/daily_input_costs.csv', parse_dates=['date'])
    apps = pd.read_csv(f'{DATA}/input_applications.csv', parse_dates=['date'])
    scout = pd.read_csv(f'{DATA}/scouting_observations.csv', parse_dates=['date'])
    prices = pd.read_csv(f'{DATA}/market_and_input_prices.csv', parse_dates=['date'])
    season = pd.read_csv(f'{DATA}/season_summary.csv')
    
    # Enriched season data
    season_plus = season.merge(plots, on='plot_id').merge(greenhouses[['greenhouse_id','structure_type','heating_system']], on='greenhouse_id')
    
    # Control baselines
    ctrl_profit = season_plus[season_plus.treatment=='Control'].groupby('crop')['season_profit_cad'].mean().to_dict()
    season_plus['profit_lift'] = season_plus.apply(
        lambda r: r['season_profit_cad'] - ctrl_profit.get(r['crop'], np.nan), axis=1
    )
    
    return {
        'farms': farms, 'greenhouses': greenhouses, 'plots': plots,
        'sensor': sensor, 'costs': costs, 'apps': apps,
        'scout': scout, 'prices': prices, 'season': season,
        'season_plus': season_plus,
    }

data = load_data()


# ─────────────────────────────────────────────
# SIDEBAR + NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌱 GreenLeaf CEA")
    st.markdown("**Precision Agriculture Intelligence**")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["Overview", "Triage", "ROI Story", "Forecast (ARIMA)", "Next Dollar (Part C)", "Ask GreenLeaf (AI)"],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    
    # Global filters
    st.markdown("### Filters")
    crops = ['All'] + sorted(data['plots']['crop'].unique().tolist())
    selected_crop = st.selectbox("Crop", crops)
    
    treatments = ['All'] + sorted(data['plots']['treatment'].unique().tolist())
    selected_treatment = st.selectbox("Treatment", treatments)
    
    st.markdown("---")
    st.caption("SFU Beedie Hackathon 2026")
    st.caption("RBC × BCCAI × GreenLeaf CEA")

# Apply global filters
sp = data['season_plus'].copy()
sensor = data['sensor'].copy()
costs = data['costs'].copy()

if selected_crop != 'All':
    sp = sp[sp.crop == selected_crop]
    sensor = sensor[sensor.crop == selected_crop]
    costs = costs[costs.plot_id.isin(sp.plot_id)]
if selected_treatment != 'All':
    sp = sp[sp.treatment == selected_treatment]
    sensor = sensor[sensor.plot_id.isin(sp.plot_id)]
    costs = costs[costs.plot_id.isin(sp.plot_id)]


# ─────────────────────────────────────────────
# HELPER: KPI card
# ─────────────────────────────────────────────
def kpi(label, value, sub="", accent=False, warn=False):
    cls = "kpi-card accent" if accent else ("kpi-card warn" if warn else "kpi-card")
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ''
    st.markdown(f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════
if page == "Overview":
    st.markdown("# The Precision Agriculture Decision Brief")
    
    st.markdown("""
    <div class="story-banner">
        <strong>THE QUESTION:</strong> Does precision agriculture actually pay back?<br>
        <strong>THE EVIDENCE:</strong> Across 120 experimental plots at GreenLeaf CEA, precision practices 
        added <strong>$47,033</strong> in profit over one growing season. But the lift is not uniform. 
        One treatment crushes it. One bleeds money. Most are statistical noise.<br>
        <strong>USE THE SIDEBAR FILTERS</strong> to explore by crop and treatment.
    </div>
    """, unsafe_allow_html=True)
    
    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi("Season Revenue", f"${sp.season_revenue_cad.sum():,.0f}", f"{len(sp)} plots")
    with k2: kpi("Season Cost", f"${sp.total_cost_cad.sum():,.0f}", "from season_summary")
    with k3: kpi("Season Profit", f"${sp.season_profit_cad.sum():,.0f}", f"ROI {sp.season_roi.mean():.1%}")
    with k4: kpi("Precision Benefit", f"${sp.precision_benefit_cad.sum():,.0f}", 
                  f"positive on {(sp.precision_benefit_cad > 0).sum()} plots", accent=True)
    with k5: kpi("Avg Yield", f"{sp.season_yield_kg_m2.mean():.1f} kg/m²", 
                  f"marketable {sp.marketable_ratio.mean():.0%}")
    
    st.markdown("&nbsp;")
    
    # Two charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### ROI by Treatment")
        tx_roi = sp.groupby('treatment').agg(
            roi=('season_roi', 'mean'),
            n=('plot_id', 'count'),
        ).reset_index().sort_values('roi', ascending=True)
        fig = px.bar(tx_roi, y='treatment', x='roi', orientation='h',
                     color='treatment', color_discrete_map=TREATMENT_COLORS,
                     text=tx_roi.roi.apply(lambda x: f'{x:.1%}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Mean ROI',
                         plot_bgcolor='white', paper_bgcolor='#faf8f4',
                         margin=dict(l=0,r=40,t=10,b=30), height=350,
                         xaxis=dict(tickformat='.0%'))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Precision Benefit by Treatment")
        tx_ben = sp.groupby('treatment').agg(
            benefit=('precision_benefit_cad', 'mean'),
        ).reset_index().sort_values('benefit', ascending=True)
        fig = px.bar(tx_ben, y='treatment', x='benefit', orientation='h',
                     color='treatment', color_discrete_map=TREATMENT_COLORS,
                     text=tx_ben.benefit.apply(lambda x: f'${x:,.0f}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Mean Precision Benefit ($)',
                         plot_bgcolor='white', paper_bgcolor='#faf8f4',
                         margin=dict(l=0,r=60,t=10,b=30), height=350,
                         xaxis=dict(tickprefix='$'))
        st.plotly_chart(fig, use_container_width=True)
    
    # Honest limits
    st.markdown("""
    <div class="limits-banner">
        <strong>WHAT THIS DASHBOARD DOES NOT TELL YOU:</strong> Pepper has zero Control plots — Lift cannot 
        be computed for Pepper treatments. Of 16 treatment-crop combinations, only 2 have 95% CIs excluding 
        zero (Strawberry High Light at +$271/plot and Strawberry Shade at −$290/plot). Daily costs and 
        season costs disagree by 2.4×; we use season_summary as the financial source of truth. This is one 
        season at one operator.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: TRIAGE
# ═════════════════════════════════════════════
elif page == "Triage":
    st.markdown("# 01 · Triage")
    st.markdown("*Where should the crew go first this morning?*")
    
    # Compute urgency per plot
    plot_urgency = sensor.groupby('plot_id').agg(
        total_alerts=('alert_flag', 'sum'),
        avg_stress=('plant_stress_index', 'mean'),
        no_response=('alert_flag', lambda x: ((x==1) & (sensor.loc[x.index, 'action_taken']==0)).sum()),
    ).reset_index()
    
    # Add delay
    alert_delay = sensor[sensor.alert_flag==1].groupby('plot_id')['action_delay_days'].mean().reset_index()
    alert_delay.columns = ['plot_id', 'avg_delay']
    plot_urgency = plot_urgency.merge(alert_delay, on='plot_id', how='left').fillna(0)
    
    # Normalize and score
    def n01(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0
    
    plot_urgency['urgency'] = (
        0.40 * n01(plot_urgency.avg_stress) +
        0.25 * n01(plot_urgency.no_response) +
        0.20 * n01(plot_urgency.total_alerts) +
        0.15 * n01(plot_urgency.avg_delay)
    )
    plot_urgency = plot_urgency.merge(
        data['plots'][['plot_id','crop','treatment','greenhouse_id']], on='plot_id'
    ).sort_values('urgency', ascending=False)
    
    # KPI row
    total_alerts = sensor.alert_flag.sum()
    worsening = ((sensor.alert_flag==1) & (sensor.action_taken==1) & (sensor.post_action_stress_delta_3d > 0)).sum()
    total_actions = ((sensor.alert_flag==1) & (sensor.action_taken==1)).sum()
    same_day = ((sensor.alert_flag==1) & (sensor.action_delay_days==0) & (sensor.action_taken==1)).sum()
    responded = ((sensor.alert_flag==1) & (sensor.action_taken==1)).sum()
    
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Total Alerts", f"{total_alerts:,}", "this season")
    with k2: kpi("Same-Day Response", f"{same_day/responded:.0%}" if responded > 0 else "N/A", "of responded alerts")
    with k3: kpi("Worsening Events", f"{worsening:,}", f"{worsening/total_actions:.0%} of actions" if total_actions > 0 else "")
    with k4: kpi("Avg Response Delay", f"{sensor[sensor.alert_flag==1].action_delay_days.mean():.1f}d", "on alert days")
    
    st.markdown("&nbsp;")
    
    # Urgent plots chart + table
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown("### Top 15 Urgent Plots")
        top = plot_urgency.head(15)
        fig = px.bar(top, y='plot_id', x='urgency', orientation='h',
                     color='crop', text=top.urgency.apply(lambda x: f'{x:.3f}'),
                     hover_data=['treatment', 'total_alerts', 'no_response', 'avg_stress'])
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis=dict(categoryorder='total ascending', title=''),
                         xaxis_title='Urgency Score (0-1)',
                         plot_bgcolor='white', paper_bgcolor='#faf8f4',
                         margin=dict(l=0,r=40,t=10,b=30), height=500,
                         legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Plot Detail")
        selected_plot = st.selectbox("Select a plot to drill down", plot_urgency.plot_id.tolist())
        
        plot_sensor = sensor[sensor.plot_id == selected_plot].sort_values('date')
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           subplot_titles=['Plant Stress Index', 'Alert & Action Timeline'],
                           vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=plot_sensor.date, y=plot_sensor.plant_stress_index,
                                mode='lines', name='Stress', line=dict(color=C['green'])), row=1, col=1)
        
        alerts = plot_sensor[plot_sensor.alert_flag==1]
        actions = plot_sensor[plot_sensor.action_taken==1]
        fig.add_trace(go.Scatter(x=alerts.date, y=[1]*len(alerts), mode='markers',
                                name='Alert', marker=dict(color=C['red'], size=6, symbol='triangle-up')), row=2, col=1)
        fig.add_trace(go.Scatter(x=actions.date, y=[0.5]*len(actions), mode='markers',
                                name='Action', marker=dict(color=C['green'], size=6, symbol='circle')), row=2, col=1)
        
        fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='#faf8f4',
                         margin=dict(l=0,r=0,t=30,b=0), showlegend=True,
                         legend=dict(orientation='h', yanchor='bottom', y=1.08))
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, 1.5], showticklabels=False, row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE: ROI STORY
# ═════════════════════════════════════════════
elif page == "ROI Story":
    st.markdown("# 02 · ROI Story")
    st.markdown("*Is the precision system worth financing?*")
    
    st.markdown("""
    <div class="story-banner">
        <strong>THE STATISTICAL TEST:</strong> For each treatment-crop pair, we compute the profit lift versus 
        the mean Control plot profit in the same crop, with 95% confidence intervals via Student's t-distribution. 
        A treatment "works" only if the entire CI excludes zero.
    </div>
    """, unsafe_allow_html=True)
    
    # Compute lift with CIs
    lift_rows = []
    for crop in sp.crop.unique():
        ctrl = sp[(sp.crop==crop) & (sp.treatment=='Control')]
        if len(ctrl) == 0:
            continue
        ctrl_mean = ctrl.season_profit_cad.mean()
        for tx in sp.treatment.unique():
            if tx == 'Control':
                continue
            sub = sp[(sp.crop==crop) & (sp.treatment==tx)]
            if len(sub) < 2:
                continue
            lifts = sub.season_profit_cad - ctrl_mean
            n = len(lifts)
            mean_lift = lifts.mean()
            sem = lifts.std() / np.sqrt(n)
            ci_low, ci_high = stats.t.interval(0.95, n-1, mean_lift, sem)
            sig = ci_low * ci_high > 0
            lift_rows.append({
                'crop': crop, 'treatment': tx, 'n': n,
                'mean_lift': mean_lift, 'ci_low': ci_low, 'ci_high': ci_high,
                'significant': sig,
            })
    
    lift_df = pd.DataFrame(lift_rows)
    
    if len(lift_df) > 0:
        # Focus on crops with Controls
        focus_crop = st.selectbox("Focus on crop (with Control plots)", 
                                  sorted(lift_df.crop.unique().tolist()),
                                  index=sorted(lift_df.crop.unique().tolist()).index('Strawberry') 
                                  if 'Strawberry' in lift_df.crop.values else 0)
        
        crop_lift = lift_df[lift_df.crop == focus_crop].sort_values('mean_lift', ascending=True)
        
        c1, c2 = st.columns([3, 2])
        
        with c1:
            st.markdown(f"### {focus_crop}: Profit Lift vs Control")
            
            fig = go.Figure()
            for _, row in crop_lift.iterrows():
                color = C['green'] if row['mean_lift'] > 0 else C['red']
                marker_color = color if row['significant'] else '#888888'
                fig.add_trace(go.Bar(
                    y=[row['treatment'] + (' ★' if row['significant'] else '')],
                    x=[row['mean_lift']],
                    orientation='h',
                    marker_color=marker_color,
                    text=f"${row['mean_lift']:+,.0f}",
                    textposition='outside',
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=[row['ci_high'] - row['mean_lift']],
                        arrayminus=[row['mean_lift'] - row['ci_low']],
                        color='rgba(0,0,0,0.3)',
                        thickness=2,
                    ),
                    showlegend=False,
                ))
            
            fig.add_vline(x=0, line_color='black', line_width=1)
            fig.update_layout(
                plot_bgcolor='white', paper_bgcolor='#faf8f4',
                xaxis_title='Profit Lift vs Control ($)', yaxis_title='',
                margin=dict(l=0, r=80, t=10, b=30), height=400,
                xaxis=dict(tickprefix='$'),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("★ = 95% CI excludes zero (statistically significant). Error bars show 95% CI. Gray bars are not significant.")
        
        with c2:
            st.markdown("### Lift Details")
            display_df = crop_lift[['treatment','n','mean_lift','ci_low','ci_high','significant']].copy()
            display_df.columns = ['Treatment', 'n', 'Mean Lift ($)', 'CI Low', 'CI High', 'Significant']
            display_df['Mean Lift ($)'] = display_df['Mean Lift ($)'].apply(lambda x: f'${x:+,.0f}')
            display_df['CI Low'] = display_df['CI Low'].apply(lambda x: f'${x:+,.0f}')
            display_df['CI High'] = display_df['CI High'].apply(lambda x: f'${x:+,.0f}')
            display_df['Significant'] = display_df['Significant'].apply(lambda x: '★ Yes' if x else 'No')
            st.dataframe(display_df.sort_values('Treatment'), hide_index=True, use_container_width=True)
            
            st.markdown("&nbsp;")
            st.markdown("### Response Timing")
            # Same-day vs delayed recovery
            sa = sensor[(sensor.alert_flag==1) & (sensor.action_taken==1)].copy()
            if len(sa) > 0:
                def bucket(d):
                    if d == 0: return '0. Same-day'
                    if d == 1: return '1. 1-day'
                    return '2. 2+ day'
                sa['bucket'] = sa.action_delay_days.apply(bucket)
                timing = sa.groupby('bucket').agg(
                    pct_improved=('post_action_stress_delta_3d', lambda x: (x<0).mean()*100)
                ).reset_index()
                
                fig = px.bar(timing, x='bucket', y='pct_improved', 
                            text=timing.pct_improved.apply(lambda x: f'{x:.0f}%'),
                            color_discrete_sequence=[C['green']])
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    plot_bgcolor='white', paper_bgcolor='#faf8f4',
                    xaxis_title='', yaxis_title='% Improved',
                    margin=dict(l=0,r=0,t=10,b=30), height=250,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No Control plots available for the selected filter combination.")


# ═════════════════════════════════════════════
# PAGE: FORECAST (ARIMA)
# ═════════════════════════════════════════════
elif page == "Forecast (ARIMA)":
    st.markdown("# 03 · Forecast")
    st.markdown("*Which plots will be in trouble next? (ARIMA time-series modeling)*")
    
    st.markdown("""
    <div class="story-banner">
        <strong>THE MODEL:</strong> We fit an ARIMA(2,1,1) model to each plot's daily plant stress index. 
        The model captures day-to-day persistence (AR), trend (I), and shock response (MA) — the three 
        dynamics that drive greenhouse stress patterns. Forecast horizon: 7 days with 95% confidence bands.
    </div>
    """, unsafe_allow_html=True)
    
    from statsmodels.tsa.arima.model import ARIMA
    
    # Select plots to forecast
    available_plots = sorted(sensor.plot_id.unique().tolist())
    
    c1, c2 = st.columns([1, 3])
    with c1:
        n_plots = st.slider("Number of plots to forecast", 1, 10, 5)
        sort_by = st.selectbox("Rank by", ["Current stress (highest first)", "Recent trend (rising first)"])
    
    # Compute current state for all plots
    last_week = sensor[sensor.date >= sensor.date.max() - pd.Timedelta(days=7)]
    prev_week = sensor[(sensor.date >= sensor.date.max() - pd.Timedelta(days=14)) & 
                       (sensor.date < sensor.date.max() - pd.Timedelta(days=7))]
    
    plot_current = last_week.groupby('plot_id').agg(
        current_stress=('plant_stress_index', 'mean'),
    ).reset_index()
    plot_prev = prev_week.groupby('plot_id').agg(
        prev_stress=('plant_stress_index', 'mean'),
    ).reset_index()
    plot_rank = plot_current.merge(plot_prev, on='plot_id', how='left')
    plot_rank['trend'] = plot_rank['current_stress'] - plot_rank['prev_stress'].fillna(0)
    
    if sort_by.startswith("Current"):
        plot_rank = plot_rank.sort_values('current_stress', ascending=False)
    else:
        plot_rank = plot_rank.sort_values('trend', ascending=False)
    
    target_plots = plot_rank.head(n_plots).plot_id.tolist()
    
    with c2:
        st.markdown(f"### ARIMA(2,1,1) Stress Forecast — Top {n_plots} Plots")
        
        fig = make_subplots(rows=n_plots, cols=1, shared_xaxes=True,
                           subplot_titles=[f'{p}' for p in target_plots],
                           vertical_spacing=0.08)
        
        for i, plot_id in enumerate(target_plots):
            ts = sensor[sensor.plot_id == plot_id].sort_values('date').set_index('date')['plant_stress_index']
            
            # Show last 60 days + 7-day forecast
            ts_recent = ts.iloc[-60:]
            
            try:
                model = ARIMA(ts, order=(2, 1, 1))
                fit = model.fit()
                forecast = fit.get_forecast(steps=7)
                fc_mean = forecast.predicted_mean
                fc_ci = forecast.conf_int(alpha=0.05)
                
                # Historical line
                fig.add_trace(go.Scatter(
                    x=ts_recent.index, y=ts_recent.values,
                    mode='lines', name=f'{plot_id} (actual)',
                    line=dict(color=C['green'], width=1.5),
                    showlegend=(i==0),
                ), row=i+1, col=1)
                
                # Forecast line
                fig.add_trace(go.Scatter(
                    x=fc_mean.index, y=fc_mean.values,
                    mode='lines', name='Forecast',
                    line=dict(color=C['red'], width=2, dash='dash'),
                    showlegend=(i==0),
                ), row=i+1, col=1)
                
                # Confidence band
                fig.add_trace(go.Scatter(
                    x=list(fc_ci.index) + list(fc_ci.index[::-1]),
                    y=list(fc_ci.iloc[:, 1]) + list(fc_ci.iloc[:, 0][::-1]),
                    fill='toself', fillcolor='rgba(200,74,62,0.15)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% CI', showlegend=(i==0),
                ), row=i+1, col=1)
                
            except Exception as e:
                fig.add_annotation(text=f"ARIMA failed: {str(e)[:50]}", 
                                  xref=f'x{i+1}', yref=f'y{i+1}',
                                  x=0.5, y=0.5, showarrow=False)
            
            fig.update_yaxes(range=[0, 1], row=i+1, col=1)
        
        fig.update_layout(
            height=180 * n_plots + 50,
            plot_bgcolor='white', paper_bgcolor='#faf8f4',
            margin=dict(l=0, r=0, t=30, b=30),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Backtest summary
    st.markdown("### Model Validation (Backtest)")
    
    # Risk score approach validation
    sensor_sorted = sensor.sort_values(['plot_id','date'])
    sensor_sorted['vpd_7d'] = sensor_sorted.groupby('plot_id')['vpd_kpa'].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    sensor_sorted['canopy_gap'] = sensor_sorted.canopy_temp_c - sensor_sorted.air_temp_c
    sensor_sorted['high_stress'] = (sensor_sorted.plant_stress_index > 0.7).astype(int)
    
    def n01(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0
    
    sensor_sorted['risk_score'] = (
        0.30 * n01(sensor_sorted.vpd_7d) +
        0.25 * n01(sensor_sorted.canopy_gap) +
        0.25 * sensor_sorted.disease_risk_index +
        0.20 * ((sensor_sorted.alert_flag==1) & (sensor_sorted.action_taken==0)).astype(float)
    )
    
    sensor_sorted['risk_p90'] = sensor_sorted.risk_score > sensor_sorted.risk_score.quantile(0.90)
    sensor_sorted['stress_next_3d'] = sensor_sorted.groupby('plot_id')['high_stress'].shift(-1).rolling(3, min_periods=1).max().reset_index(level=0,drop=True)
    
    p_high = sensor_sorted.loc[sensor_sorted.risk_p90==True, 'stress_next_3d'].mean()
    p_low = sensor_sorted.loc[sensor_sorted.risk_p90==False, 'stress_next_3d'].mean()
    
    b1, b2, b3 = st.columns(3)
    with b1: kpi("Predictive Lift", f"{p_high/p_low:.1f}×" if p_low > 0 else "N/A", 
                 "top-decile risk vs baseline", accent=True)
    with b2: kpi("P(stress | high risk)", f"{p_high*100:.1f}%" if not np.isnan(p_high) else "N/A",
                 "in next 3 days")
    with b3: kpi("P(stress | low risk)", f"{p_low*100:.1f}%" if not np.isnan(p_low) else "N/A",
                 "in next 3 days")


# ═════════════════════════════════════════════
# ═════════════════════════════════════════════
# PAGE: NEXT DOLLAR (PART C — OPEN SCENARIO)
# ═════════════════════════════════════════════
elif page == "Next Dollar (Part C)":
    st.markdown("# Part C · Where Should the Next Dollar Go?")
    st.markdown("*Two perspectives. One question. Which precision investments produce the highest return?*")
    
    # Pull ALL tables
    apps = data['apps']
    scout = data['scout']
    prices = data['prices']
    greenhouses = data['greenhouses']
    farms = data['farms']
    # sp is already sidebar-filtered (defined above); do NOT overwrite it
    
    # Enrich and filter apps to match sidebar selection
    apps_enriched = apps.merge(data['plots'][['plot_id','crop','treatment']], on='plot_id', how='left')
    if selected_crop != 'All':
        apps_enriched = apps_enriched[apps_enriched.crop == selected_crop]
    if selected_treatment != 'All':
        apps_enriched = apps_enriched[apps_enriched.treatment == selected_treatment]
    sp_clim = sp.merge(farms[['farm_id','climate_zone']], on='farm_id', how='left')
    prec_apps = apps_enriched[apps_enriched.is_precision_action == 1]
    
    # ── Two-tab layout: Farmer vs Lender ──
    farmer_tab, lender_tab, recommend_tab = st.tabs(["🌾 Farmer View", "🏦 Lender View (RBC)", "📋 Recommendation"])
    
    # ═══════════════════════════════════════
    # TAB 1: FARMER VIEW
    # ═══════════════════════════════════════
    with farmer_tab:
        st.markdown("""
        <div class="farmer-banner">
            <strong>FOR THE FARMER:</strong> You're deciding where to focus your precision budget next season. 
            These charts show which treatments, crops, and infrastructure produce the best outcomes — and how 
            fast you need to respond to alerts to actually move the needle.
        </div>
        """, unsafe_allow_html=True)
        
        # ── Farmer Section 1: Treatment head-to-head ──
        st.markdown("## Which treatment works best for my crop?")
        
        crop_options = ['All Crops'] + sorted(sp.crop.unique().tolist())
        farmer_crop = st.selectbox("Select your crop:", crop_options, key='farmer_crop')
        
        if farmer_crop == 'All Crops':
            crop_data = sp.copy()
        else:
            crop_data = sp[sp.crop == farmer_crop]
        crop_label = farmer_crop
        
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: kpi("Plots", f"{len(crop_data)}", crop_label)
        with k2: kpi("Best ROI Treatment", 
                     crop_data.groupby('treatment').season_roi.mean().idxmax(),
                     f"{crop_data.groupby('treatment').season_roi.mean().max():.0%} ROI", accent=True)
        with k3: kpi("Worst ROI Treatment",
                     crop_data.groupby('treatment').season_roi.mean().idxmin(),
                     f"{crop_data.groupby('treatment').season_roi.mean().min():.0%} ROI", warn=True)
        with k4: kpi("Avg Profit/Plot", f"${crop_data.season_profit_cad.mean():,.0f}", crop_label)
        with k5: kpi("Avg Yield", f"{crop_data.season_yield_kg_m2.mean():.1f} kg/m²", 
                     f"marketable {crop_data.marketable_ratio.mean():.0%}")
        
        # ── All Crops cross-comparison (only when All Crops selected) ──
        if farmer_crop == 'All Crops':
            st.markdown("&nbsp;")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Profit per Plot by Crop × Treatment")
                crop_tx = sp.groupby(['crop','treatment']).agg(
                    profit=('season_profit_cad','mean'), n=('plot_id','count')
                ).reset_index()
                fig = px.bar(crop_tx, x='crop', y='profit', color='treatment', barmode='group',
                            color_discrete_map=TREATMENT_COLORS,
                            text=crop_tx.profit.apply(lambda x: f'${x:,.0f}'))
                fig.update_traces(textposition='outside', textfont_size=9)
                fig.update_layout(xaxis_title='', yaxis_title='Mean Profit ($)',
                                 yaxis=dict(tickprefix='$'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                                 height=420, margin=dict(l=0,r=0,t=10,b=30),
                                 legend=dict(orientation='h', yanchor='bottom', y=1.02, title=''))
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.markdown("### Yield vs Marketable Ratio — All 120 Plots")
                all_plot = sp.copy()
                all_plot['profit_size'] = all_plot.season_profit_cad.clip(lower=1)
                fig = px.scatter(all_plot, x='season_yield_kg_m2', y='marketable_ratio',
                               color='crop', size='profit_size', hover_data=['plot_id','treatment'],
                               size_max=18)
                fig.update_layout(xaxis_title='Yield (kg/m²)', yaxis_title='Marketable Ratio',
                                 yaxis=dict(tickformat='.0%'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                                 height=420, margin=dict(l=0,r=0,t=10,b=30),
                                 legend=dict(orientation='h', yanchor='bottom', y=1.02, title=''))
                st.plotly_chart(fig, use_container_width=True)
            
            # ROI boxplot by crop
            st.markdown("### ROI Distribution by Crop")
            fig = px.box(sp, x='crop', y='season_roi', color='crop', points='all',
                        hover_data=['plot_id','treatment'])
            fig.update_layout(xaxis_title='', yaxis_title='Season ROI', showlegend=False,
                             yaxis=dict(tickformat='.0%'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=350, margin=dict(l=0,r=0,t=10,b=30))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each dot is one plot. Hover for plot ID and treatment. Strawberry has the widest spread; Pepper has the fewest plots (n=9).")
        
        # ── Single crop treatment comparison (always shown) ──
        st.markdown("&nbsp;")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### Treatment Comparison — Profit per Plot ({crop_label})")
            tx_comp = crop_data.groupby('treatment').agg(
                profit=('season_profit_cad','mean'), n=('plot_id','count'),
                roi=('season_roi','mean'),
            ).reset_index().sort_values('profit')
            fig = px.bar(tx_comp, y='treatment', x='profit', orientation='h',
                        color='treatment', color_discrete_map=TREATMENT_COLORS,
                        text=tx_comp.apply(lambda r: f"${r.profit:,.0f} (n={r.n})", axis=1),
                        hover_data=['roi','n'])
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Mean Profit per Plot ($)',
                             xaxis=dict(tickprefix='$'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=380, margin=dict(l=0,r=100,t=10,b=30))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown(f"### Yield vs Marketable Ratio ({crop_label})")
            crop_data_plot = crop_data.copy()
            crop_data_plot['profit_size'] = crop_data_plot.season_profit_cad.clip(lower=1)
            fig = px.scatter(crop_data_plot, x='season_yield_kg_m2', y='marketable_ratio',
                           color='treatment', color_discrete_map=TREATMENT_COLORS,
                           size='profit_size', hover_data=['plot_id'],
                           size_max=20)
            fig.update_layout(xaxis_title='Yield (kg/m²)', yaxis_title='Marketable Ratio',
                             yaxis=dict(tickformat='.0%'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=380, margin=dict(l=0,r=0,t=10,b=30),
                             legend=dict(orientation='h', yanchor='bottom', y=1.02, title=''))
            st.plotly_chart(fig, use_container_width=True)
        
        # ── Farmer Section 2: Infrastructure ──
        st.markdown("---")
        st.markdown("## Should I upgrade my greenhouse infrastructure?")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Glass vs Polyfilm")
            struct = sp.groupby('structure_type').agg(
                roi=('season_roi','mean'), profit=('season_profit_cad','mean'), n=('plot_id','count')
            ).reset_index()
            fig = px.bar(struct, x='structure_type', y=['roi'], barmode='group',
                        text=struct.roi.apply(lambda x: f'{x:.1%}'),
                        color_discrete_sequence=[C['blue']])
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='Mean ROI',
                             yaxis=dict(tickformat='.0%'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=300, margin=dict(l=0,r=0,t=10,b=30))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("### Heating System")
            heat = sp.groupby('heating_system').agg(
                roi=('season_roi','mean'), n=('plot_id','count')
            ).reset_index().sort_values('roi')
            fig = px.bar(heat, y='heating_system', x='roi', orientation='h',
                        text=heat.roi.apply(lambda x: f'{x:.1%}'),
                        color_discrete_sequence=[C['gold']])
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Mean ROI',
                             xaxis=dict(tickformat='.0%'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=300, margin=dict(l=0,r=40,t=10,b=30))
            st.plotly_chart(fig, use_container_width=True)
        
        with c3:
            st.markdown("### Climate Zone")
            clim = sp_clim.groupby('climate_zone').agg(
                roi=('season_roi','mean'), n=('plot_id','count')
            ).reset_index().sort_values('roi')
            fig = px.bar(clim, y='climate_zone', x='roi', orientation='h',
                        text=clim.roi.apply(lambda x: f'{x:.1%}'),
                        color_discrete_sequence=[C['green']])
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Mean ROI',
                             xaxis=dict(tickformat='.0%'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=300, margin=dict(l=0,r=40,t=10,b=30))
            st.plotly_chart(fig, use_container_width=True)
        
        # ── Farmer Section 3: Response speed ──
        st.markdown("---")
        st.markdown("## How fast do I need to respond?")
        
        sa = data['sensor'][(data['sensor'].alert_flag==1) & (data['sensor'].action_taken==1)].copy()
        def bucket(d):
            if d == 0: return '0. Same-day'
            if d == 1: return '1. Next-day'
            return '2. Two+ days'
        sa['bucket'] = sa.action_delay_days.apply(bucket)
        timing = sa.groupby('bucket').agg(
            improved=('post_action_stress_delta_3d', lambda x: (x<0).mean()*100),
            mean_delta=('post_action_stress_delta_3d', 'mean'),
            n=('plot_id', 'count'),
        ).reset_index()
        
        c1, c2 = st.columns([2,1])
        with c1:
            fig = px.bar(timing, x='bucket', y='improved',
                        text=timing.improved.apply(lambda x: f'{x:.0f}%'),
                        color='improved',
                        color_continuous_scale=[[0, C['red']], [1, C['green']]],)
            fig.update_traces(textposition='outside', textfont_size=18)
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                             xaxis_title='', yaxis_title='% of actions that improved stress',
                             plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=350, margin=dict(l=0,r=0,t=10,b=30))
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            kpi("Same-day Recovery", "87%", "of actions improved stress")
            st.markdown("&nbsp;")
            kpi("2+ Day Recovery", "59%", "barely better than chance", warn=True)
            st.markdown("&nbsp;")
            kpi("The Gap", "28 pts", "this is the value of speed", accent=True)
    
    # ═══════════════════════════════════════
    # TAB 2: LENDER VIEW (RBC)
    # ═══════════════════════════════════════
    with lender_tab:
        st.markdown("""
        <div class="rbc-banner">
            <strong>FOR THE LENDER:</strong> You're evaluating whether to finance precision agriculture equipment. 
            These charts quantify the risk: how variable are returns, which borrowers are safest, and what happens 
            to ROI when input prices spike.
        </div>
        """, unsafe_allow_html=True)
        
        # ── Lender Section 1: Risk Profile ──
        st.markdown("## How risky is the investment?")
        
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("Mean ROI", f"{sp.season_roi.mean():.0%}", "across all 120 plots")
        with k2: kpi("ROI Std Dev", f"{sp.season_roi.std():.0%}", "volatility of returns")
        with k3: 
            pct_positive = (sp.season_profit_cad > 0).mean() * 100
            kpi("% Profitable", f"{pct_positive:.0f}%", "plots with profit > 0", accent=True)
        with k4:
            worst = sp.season_roi.quantile(0.05)
            kpi("5th Percentile ROI", f"{worst:.0%}", "worst-case scenario (1 in 20)", warn=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ROI Distribution — Where's the Risk?")
            fig = px.histogram(sp, x='season_roi', nbins=25, color_discrete_sequence=[C['blue']],
                              marginal='box')
            fig.add_vline(x=0, line_dash='dash', line_color=C['red'], annotation_text='Breakeven')
            fig.add_vline(x=sp.season_roi.mean(), line_dash='dash', line_color=C['green'], 
                         annotation_text=f'Mean: {sp.season_roi.mean():.0%}')
            fig.update_layout(xaxis_title='Season ROI', yaxis_title='Number of Plots',
                             xaxis=dict(tickformat='.0%'), plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=400, margin=dict(l=0,r=0,t=10,b=30), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("### ROI Variance by Treatment — Which Bets Are Stable?")
            tx_risk = sp.groupby('treatment').agg(
                mean_roi=('season_roi','mean'),
                std_roi=('season_roi','std'),
                min_roi=('season_roi','min'),
                n=('plot_id','count'),
            ).reset_index()
            fig = px.scatter(tx_risk, x='mean_roi', y='std_roi', size='n', color='treatment',
                           color_discrete_map=TREATMENT_COLORS, size_max=30,
                           hover_data=['min_roi','n'],
                           text='treatment')
            fig.update_traces(textposition='top center', textfont_size=11)
            fig.update_layout(xaxis_title='Mean ROI (return)', yaxis_title='ROI Std Dev (risk)',
                             xaxis=dict(tickformat='.0%'), yaxis=dict(tickformat='.0%'),
                             plot_bgcolor='white', paper_bgcolor='#faf8f4',
                             height=400, margin=dict(l=0,r=0,t=10,b=30), showlegend=False)
            fig.add_annotation(x=0.55, y=0.05, text="← LOW RISK, HIGH RETURN", showarrow=False,
                             font=dict(size=10, color=C['green']))
            fig.add_annotation(x=0.25, y=0.35, text="HIGH RISK, LOW RETURN →", showarrow=False,
                             font=dict(size=10, color=C['red']))
            st.plotly_chart(fig, use_container_width=True)
        
        # ── Lender Section 2: Sensitivity Analysis ──
        st.markdown("---")
        st.markdown("## What happens when input prices spike?")
        
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("### Price Scenario")
            fert_d = st.slider("Fertilizer", -30, 50, 0, 5, format="%+d%%", key='lender_fert')
            energy_d = st.slider("Energy", -30, 50, 0, 5, format="%+d%%", key='lender_energy')
            labor_d = st.slider("Labor", -20, 40, 0, 5, format="%+d%%", key='lender_labor')
        
        with c2:
            # Compute scenario impact
            cost_by_type = apps_enriched.groupby(['plot_id','input_type'])['total_cost'].sum().unstack(fill_value=0).reset_index()
            for col in ['Fertilizer','Energy','Labor','Pesticide','Water']:
                if col not in cost_by_type.columns:
                    cost_by_type[col] = 0
            
            cost_by_type['cost_change'] = (
                cost_by_type.get('Fertilizer',0) * (fert_d/100) +
                cost_by_type.get('Energy',0) * (energy_d/100) +
                cost_by_type.get('Labor',0) * (labor_d/100)
            )
            scenario = sp.merge(cost_by_type[['plot_id','cost_change']], on='plot_id', how='left').fillna(0)
            scenario['new_profit'] = scenario.season_profit_cad - scenario.cost_change
            scenario['new_roi'] = scenario.new_profit / (scenario.total_cost_cad + scenario.cost_change)
            
            comparison = scenario.groupby('treatment').agg(
                base=('season_roi','mean'), new=('new_roi','mean')
            ).reset_index()
            comparison['change'] = comparison.new - comparison.base
            comparison = comparison.sort_values('new')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(y=comparison.treatment, x=comparison.base, orientation='h',
                                name='Baseline', marker_color='#cccccc',
                                text=comparison.base.apply(lambda x: f'{x:.1%}'), textposition='inside'))
            bar_color = C['green'] if (fert_d + energy_d + labor_d) <= 0 else C['red']
            fig.add_trace(go.Bar(y=comparison.treatment, x=comparison.new, orientation='h',
                                name='Scenario', marker_color=bar_color,
                                text=comparison.new.apply(lambda x: f'{x:.1%}'), textposition='inside'))
            fig.update_layout(barmode='group', plot_bgcolor='white', paper_bgcolor='#faf8f4', height=380,
                             xaxis_title='ROI', xaxis=dict(tickformat='.0%'), yaxis_title='',
                             margin=dict(l=0,r=0,t=10,b=30),
                             legend=dict(orientation='h', yanchor='bottom', y=1.02, title=''))
            st.plotly_chart(fig, use_container_width=True)
            
            total_impact = scenario.cost_change.sum()
            plots_negative = (scenario.new_profit < 0).sum()
            if abs(total_impact) > 0:
                st.markdown(f"""
                <div class="story-banner">
                    <strong>SCENARIO IMPACT:</strong> Total cost delta: <strong>${total_impact:+,.0f}</strong>. 
                    Plots dropping below breakeven: <strong>{plots_negative} of {len(scenario)}</strong>. 
                    Most resilient: <strong>{comparison.iloc[-1].treatment}</strong>. 
                    Most exposed: <strong>{comparison.iloc[0].treatment}</strong>.
                </div>
                """, unsafe_allow_html=True)
        
        # ── Lender Section 3: Treatment × Zone Heatmap ──
        st.markdown("---")
        st.markdown("## Which region-treatment combos should we finance?")
        
        hm_data = sp_clim.groupby(['climate_zone','treatment']).agg(
            roi=('season_roi','mean'), n=('plot_id','count')
        ).reset_index()
        pivot = hm_data.pivot(index='treatment', columns='climate_zone', values='roi')
        pivot_n = hm_data.pivot(index='treatment', columns='climate_zone', values='n').fillna(0)
        
        text_matrix = []
        for tx in pivot.index:
            row = []
            for cz in pivot.columns:
                v = pivot.loc[tx, cz]
                n = int(pivot_n.loc[tx, cz]) if tx in pivot_n.index and cz in pivot_n.columns else 0
                row.append(f'{v:.0%}\n(n={n})' if not pd.isna(v) else '')
            text_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            text=text_matrix, texttemplate='%{text}', textfont=dict(size=12),
            colorscale=[[0, C['red']], [0.5, '#f5f0e0'], [1, C['green']]],
            zmid=0.4, colorbar=dict(title='ROI', tickformat='.0%'),
        ))
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='#faf8f4', height=420,
                         xaxis_title='Climate Zone', yaxis_title='Treatment',
                         margin=dict(l=0,r=0,t=10,b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green cells = high ROI. Red = low. Cells with n < 5 are directional only.")
    
    # ═══════════════════════════════════════
    # TAB 3: RECOMMENDATION
    # ═══════════════════════════════════════
    with recommend_tab:
        st.markdown("## The Data-Driven Recommendation")
        
        # Find best combos
        best = sp_clim.merge(greenhouses[['greenhouse_id','structure_type']], on='greenhouse_id', suffixes=('','_gh'))
        best_agg = best.groupby(['crop','treatment','structure_type_gh','climate_zone']).agg(
            roi=('season_roi','mean'), profit=('season_profit_cad','mean'),
            benefit=('precision_benefit_cad','mean'), n=('plot_id','count'),
        ).reset_index()
        best_agg = best_agg[best_agg.n >= 3].sort_values('roi', ascending=False)
        
        if len(best_agg) > 0:
            top = best_agg.iloc[0]
            k1, k2, k3 = st.columns(3)
            with k1:
                kpi("Best Combination", f"{top.treatment}", f"{top.crop} · {top.structure_type_gh} · {top.climate_zone}", accent=True)
            with k2:
                kpi("Expected ROI", f"{top.roi:.0%}", f"${top.profit:,.0f} profit/plot")
            with k3:
                kpi("Evidence", f"n = {int(top.n)}", f"${top.benefit:,.0f} precision benefit/plot")
        
        st.markdown("&nbsp;")
        st.markdown("### Top Combinations (minimum 3 plots of evidence)")
        
        top10 = best_agg.head(12).copy()
        fig = px.bar(top10, y=top10.apply(lambda r: f"{r.crop} · {r.treatment} · {r.structure_type_gh}", axis=1),
                    x='roi', orientation='h', color='climate_zone',
                    text=top10.apply(lambda r: f"{r.roi:.0%} (n={int(r.n)})", axis=1),
                    hover_data=['profit','benefit','n'])
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis_title='', xaxis_title='Mean ROI', xaxis=dict(tickformat='.0%'),
                         plot_bgcolor='white', paper_bgcolor='#faf8f4', height=500,
                         margin=dict(l=0,r=100,t=10,b=30),
                         legend=dict(orientation='h', yanchor='bottom', y=1.02, title='Zone'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Scouting validation
        st.markdown("---")
        st.markdown("### Ground Truth: Do Scouts Agree with Sensors?")
        
        scout_e = scout.merge(data['plots'][['plot_id','crop','treatment']], on='plot_id', how='left')
        scout_yield = scout_e.groupby('plot_id').agg(
            blossom=('blossom_count','mean'),
            pest_sev=('pest_severity','mean'),
        ).reset_index().merge(sp[['plot_id','season_yield_kg_m2','crop','treatment']], on='plot_id')
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(scout_yield, x='blossom', y='season_yield_kg_m2', color='crop',
                           trendline='ols', hover_data=['plot_id','treatment'])
            fig.update_layout(xaxis_title='Avg Blossom Count (scout observation)',
                             yaxis_title='Actual Season Yield (kg/m²)',
                             plot_bgcolor='white', paper_bgcolor='#faf8f4', height=380,
                             margin=dict(l=0,r=0,t=10,b=30),
                             legend=dict(orientation='h', yanchor='bottom', y=1.02, title=''))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Higher blossom count predicts higher yield. The trendline confirms sensors and scouts tell the same story.")
        
        with c2:
            fig = px.scatter(scout_yield, x='pest_sev', y='season_yield_kg_m2', color='crop',
                           trendline='ols', hover_data=['plot_id','treatment'])
            fig.update_layout(xaxis_title='Avg Pest Severity (scout observation, 0=none, 1=severe)',
                             yaxis_title='Actual Season Yield (kg/m²)',
                             plot_bgcolor='white', paper_bgcolor='#faf8f4', height=380,
                             margin=dict(l=0,r=0,t=10,b=30),
                             legend=dict(orientation='h', yanchor='bottom', y=1.02, title=''))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Higher pest severity correlates with lower yield, as expected. Scout observations validate sensor-driven pest alerts.")
        
        # Final caveats
        st.markdown("""
        <div class="limits-banner">
            <strong>PART C CAVEATS:</strong> Infrastructure comparisons are observational, not randomized — 
            Glass vs Polyfilm differences may reflect management quality, not structure alone. 
            Treatment × zone cells with n &lt; 5 are directional, not conclusive. 
            Price sensitivity uses linear scaling which understates tail risk. 
            Pepper has zero Control plots so cannot appear in Lift analysis. 
            These findings generate hypotheses for next season's experimental design, not deployment-ready loan terms.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: ASK GREENLEAF (LLM)
# ═════════════════════════════════════════════
elif page == "Ask GreenLeaf (AI)":
    st.markdown("# 04 · Ask GreenLeaf")
    st.markdown("*Ask a question. Get a chart and a short answer, not a wall of text.*")
    
    import os, json
    
    # Check for API key: st.secrets (Streamlit Cloud) → env var → manual input
    api_key = ''
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.environ.get('OPENAI_API_KEY', '')
    
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password",
                                help="Set in Streamlit Cloud secrets or OPENAI_API_KEY env var.")
    
    if not api_key:
        st.info("Enter an OpenAI API key to enable AI-powered Q&A.")
        st.stop()
    
    # ── Pre-built chart catalog (uses REAL data, not LLM-generated) ──
    def chart_roi_by_treatment():
        tx = data['season_plus'].groupby('treatment')['season_roi'].mean().reset_index().sort_values('season_roi')
        fig = px.bar(tx, y='treatment', x='season_roi', orientation='h',
                     color='treatment', color_discrete_map=TREATMENT_COLORS,
                     text=tx.season_roi.apply(lambda x: f'{x:.1%}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Mean ROI',
                         plot_bgcolor='white', paper_bgcolor='#faf8f4', height=350,
                         margin=dict(l=0,r=50,t=10,b=30), xaxis=dict(tickformat='.0%'),
                         )
        return fig
    
    def chart_precision_benefit():
        tx = data['season_plus'].groupby('treatment')['precision_benefit_cad'].mean().reset_index().sort_values('precision_benefit_cad')
        fig = px.bar(tx, y='treatment', x='precision_benefit_cad', orientation='h',
                     color='treatment', color_discrete_map=TREATMENT_COLORS,
                     text=tx.precision_benefit_cad.apply(lambda x: f'${x:,.0f}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Mean Precision Benefit ($)',
                         plot_bgcolor='white', paper_bgcolor='#faf8f4', height=350,
                         margin=dict(l=0,r=60,t=10,b=30), xaxis=dict(tickprefix='$'),
                         )
        return fig
    
    def chart_profit_by_crop():
        cr = data['season_plus'].groupby('crop')['season_profit_cad'].mean().reset_index().sort_values('season_profit_cad')
        fig = px.bar(cr, y='crop', x='season_profit_cad', orientation='h',
                     text=cr.season_profit_cad.apply(lambda x: f'${x:,.0f}'),
                     color_discrete_sequence=[C['green']])
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Mean Profit ($)',
                         plot_bgcolor='white', paper_bgcolor='#faf8f4', height=280,
                         margin=dict(l=0,r=60,t=10,b=30), xaxis=dict(tickprefix='$'),
                         )
        return fig
    
    def chart_response_timing():
        sa = data['sensor'][(data['sensor'].alert_flag==1) & (data['sensor'].action_taken==1)].copy()
        def bucket(d):
            if d == 0: return 'Same-day'
            if d == 1: return '1-day delay'
            return '2+ day delay'
        sa['bucket'] = sa.action_delay_days.apply(bucket)
        timing = sa.groupby('bucket').agg(
            pct=('post_action_stress_delta_3d', lambda x: (x<0).mean()*100),
            n=('plot_id', 'count'),
        ).reset_index().sort_values('pct', ascending=False)
        fig = px.bar(timing, x='bucket', y='pct', text=timing.pct.apply(lambda x: f'{x:.0f}%'),
                     color_discrete_sequence=[C['green']])
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='% Stress Improved',
                         plot_bgcolor='white', paper_bgcolor='#faf8f4', height=300,
                         margin=dict(l=0,r=0,t=10,b=30),
                         )
        return fig
    
    def chart_strawberry_lift():
        sdata = data['season_plus']
        ctrl = sdata[(sdata.crop=='Strawberry') & (sdata.treatment=='Control')]
        if len(ctrl) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No Strawberry Control plots in filter", x=0.5, y=0.5, showarrow=False)
            return fig
        ctrl_mean = ctrl.season_profit_cad.mean()
        straw = sdata[(sdata.crop=='Strawberry') & (sdata.treatment!='Control')].copy()
        if len(straw) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No Strawberry treatment plots in filter", x=0.5, y=0.5, showarrow=False)
            return fig
        straw['lift'] = straw.season_profit_cad - ctrl_mean
        tx_lift = straw.groupby('treatment')['lift'].agg(['mean','std','count']).reset_index()
        tx_lift.columns = ['treatment','mean_lift','std_lift','n_plots']
        tx_lift['sem_val'] = tx_lift['std_lift'] / np.sqrt(tx_lift['n_plots'])
        tx_lift = tx_lift.sort_values('mean_lift')
        
        fig = go.Figure()
        for _, r in tx_lift.iterrows():
            color = C['green'] if r['mean_lift'] > 0 else C['red']
            ci_w = r['sem_val'] * stats.t.ppf(0.975, r['n_plots']-1) if r['n_plots'] > 1 else 0
            fig.add_trace(go.Bar(
                y=[r['treatment']], x=[r['mean_lift']], orientation='h',
                marker_color=color, text=f"${r['mean_lift']:+,.0f}", textposition='outside',
                error_x=dict(type='data', symmetric=True, array=[ci_w], color='rgba(0,0,0,0.3)', thickness=2),
                showlegend=False,
            ))
        fig.add_vline(x=0, line_color='black', line_width=1)
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='#faf8f4', height=300,
                         xaxis_title='Profit Lift vs Control ($)', yaxis_title='',
                         margin=dict(l=0,r=80,t=10,b=30), xaxis=dict(tickprefix='$'),
                         )
        return fig
    
    def chart_alert_types():
        alerts = data['sensor'][data['sensor'].alert_flag==1]
        at = alerts.groupby('alert_type').agg(
            total=('alert_flag','sum'),
            ignored=('action_taken', lambda x: (x==0).sum()),
        ).reset_index()
        at['ignore_rate'] = at.ignored / at.total * 100
        at = at.sort_values('total', ascending=True)
        fig = px.bar(at, y='alert_type', x='total', orientation='h',
                     text=at.apply(lambda r: f"{r.total:,} ({r.ignore_rate:.0f}% ignored)", axis=1),
                     color_discrete_sequence=[C['blue']])
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='Alert Count',
                         plot_bgcolor='white', paper_bgcolor='#faf8f4', height=280,
                         margin=dict(l=0,r=120,t=10,b=30),
                         )
        return fig
    
    def chart_top_urgent():
        pu = sensor.groupby('plot_id').agg(
            alerts=('alert_flag','sum'),
            stress=('plant_stress_index','mean'),
            ignored=('alert_flag', lambda x: ((x==1) & (sensor.loc[x.index,'action_taken']==0)).sum()),
        ).reset_index()
        def n01(s):
            r = s.max() - s.min()
            return (s - s.min()) / r if r > 0 else s * 0
        pu['urgency'] = 0.4*n01(pu.stress) + 0.25*n01(pu.ignored) + 0.2*n01(pu.alerts) + 0.15*0.5
        pu = pu.merge(data['plots'][['plot_id','crop','treatment']], on='plot_id')
        top = pu.nlargest(10, 'urgency')
        fig = px.bar(top, y='plot_id', x='urgency', orientation='h', color='crop',
                     text=top.urgency.apply(lambda x: f'{x:.3f}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis=dict(categoryorder='total ascending', title=''),
                         xaxis_title='Urgency Score', plot_bgcolor='white', paper_bgcolor='#faf8f4',
                         height=350, margin=dict(l=0,r=40,t=10,b=30),
                         
                         legend=dict(orientation='h', yanchor='bottom', y=1.02))
        return fig
    
    def chart_cost_breakdown():
        cost_roll = data['costs'].groupby('plot_id').agg(
            precision=('daily_precision_cost','sum'),
            routine=('daily_routine_cost','sum'),
        ).reset_index()
        merged = cost_roll.merge(data['plots'][['plot_id','treatment']], on='plot_id')
        by_tx = merged.groupby('treatment')[['precision','routine']].mean().reset_index()
        by_tx = by_tx.melt(id_vars='treatment', var_name='type', value_name='cost')
        fig = px.bar(by_tx, x='treatment', y='cost', color='type', barmode='stack',
                     color_discrete_map={'precision': C['green'], 'routine': '#aaa'},
                     text=by_tx.cost.apply(lambda x: f'${x:,.0f}'))
        fig.update_traces(textposition='inside')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='#faf8f4', height=350,
                         xaxis_title='', yaxis_title='Mean Cost ($)', yaxis=dict(tickprefix='$'),
                         margin=dict(l=0,r=0,t=10,b=30),
                         
                         legend=dict(orientation='h', yanchor='bottom', y=1.02))
        return fig
    
    CHART_CATALOG = {
        'roi_by_treatment': {'fn': chart_roi_by_treatment, 'desc': 'ROI by treatment (bar)', 'title': 'ROI by Treatment'},
        'precision_benefit': {'fn': chart_precision_benefit, 'desc': 'Precision benefit by treatment (bar)', 'title': 'Precision Benefit by Treatment'},
        'profit_by_crop': {'fn': chart_profit_by_crop, 'desc': 'Profit by crop (bar)', 'title': 'Profit by Crop'},
        'response_timing': {'fn': chart_response_timing, 'desc': 'Stress recovery by response speed (bar)', 'title': 'Stress Recovery by Response Speed'},
        'strawberry_lift': {'fn': chart_strawberry_lift, 'desc': 'Strawberry profit lift vs Control with CIs (bar)', 'title': 'Strawberry: Lift vs Control (95% CI)'},
        'alert_types': {'fn': chart_alert_types, 'desc': 'Alert type distribution and ignore rates (bar)', 'title': 'Alerts by Type (with Ignore Rate)'},
        'top_urgent': {'fn': chart_top_urgent, 'desc': 'Top 10 urgent plots by urgency score (bar)', 'title': 'Top 10 Urgent Plots'},
        'cost_breakdown': {'fn': chart_cost_breakdown, 'desc': 'Precision vs routine spend by treatment (stacked bar)', 'title': 'Precision vs Routine Spend'},
    }
    
    # ── System prompt: visual-first, structured JSON output ──
    @st.cache_data
    def build_visual_context():
        sp = data['season_plus']
        s = data['sensor']
        
        chart_list = "\n".join([f'  - "{k}": {v["desc"]}' for k, v in CHART_CATALOG.items()])
        
        return f"""You are a visual-first agricultural data analyst for GreenLeaf CEA. 
Your audience is business people who prefer charts over text.

RESPOND ONLY WITH VALID JSON (no markdown, no backticks, no preamble). Format:
{{
  "headline": {{"label": "SHORT_LABEL", "value": "BIG_NUMBER", "context": "one line"}},
  "insight": "2-3 SHORT sentences maximum. Be direct. Cite specific dollar amounts.",
  "charts": ["chart_id_1", "chart_id_2"]
}}

AVAILABLE CHARTS (pick 1-2 most relevant):
{chart_list}

RULES:
- "insight" must be 2-3 sentences MAX. No bullet points. No lists.
- "headline" should be the single most important number for the question.
- "charts" should be 1-2 chart IDs from the list above. Pick the ones that ANSWER the question visually.
- If the question is about treatments, include "roi_by_treatment" or "precision_benefit" or "strawberry_lift".
- If the question is about response speed or alerts, include "response_timing" or "alert_types".
- If the question is about costs, include "cost_breakdown".
- If the question is about urgency or triage, include "top_urgent".

DATASET FACTS:
- 120 plots, 7 treatments, 4 crops (Tomato, Strawberry, Cucumber, Pepper)
- Pepper has ZERO Control plots (cannot compute lift)
- Total precision benefit: ${sp.precision_benefit_cad.sum():,.0f} across 120 plots
- Mean ROI: {sp.season_roi.mean():.1%}
- Strawberry High Light: +$271/plot lift vs Control (significant, CI $21-$522)
- Strawberry Shade: -$290/plot loss vs Control (significant, CI -$486 to -$95)
- Same-day response improves stress 87% of the time vs 59% for 2-day delay
- 1,108 of 4,820 actions worsened stress (23%)
- Total alerts: {s.alert_flag.sum():,}
- High Pest Pressure: 6,410 alerts (25% ignored). Environmental alerts: 35 total (80-100% ignored).
- ROI by treatment: High Light 58.5%, Low N 45.3%, Control 43.0%, High N 41.4%, Integrated Pest 38.1%, Reduced Pest 31.0%, Shade 31.0%
- Only 2 of 16 treatment-crop pairs are statistically significant"""
    
    visual_context = build_visual_context()
    
    # ── Chat interface ──
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'visual_history' not in st.session_state:
        st.session_state.visual_history = []
    
    # Display history
    for i, msg in enumerate(st.session_state.messages):
        if msg['role'] == 'user':
            with st.chat_message('user'):
                st.markdown(msg['content'])
        else:
            # Assistant text only in bubble
            safe = msg['content'].replace('$', '\\$') if msg['content'] else ''
            st.markdown(safe)
            # Charts rendered at full width outside any bubble
            if i < len(st.session_state.visual_history):
                vis = st.session_state.visual_history[i]
                if vis and 'charts' in vis:
                    cols = st.columns(len(vis['charts'])) if len(vis['charts']) > 1 else [st.container()]
                    for j, chart_id in enumerate(vis['charts']):
                        if chart_id in CHART_CATALOG:
                            with cols[j]:
                                st.markdown(f"**{CHART_CATALOG[chart_id]['title']}**")
                                st.plotly_chart(CHART_CATALOG[chart_id]['fn'](), use_container_width=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about the GreenLeaf data..."):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.session_state.visual_history.append(None)
        
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Process the LLM response
        parsed_ok = False
        chart_ids = []
        result = None
        raw = ''
        client = None
        
        with st.spinner("Analyzing data..."):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": visual_context},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=400,
                    temperature=0.2,
                )
                
                raw = response.choices[0].message.content.strip()
                raw = raw.replace('```json', '').replace('```', '').strip()
                
                try:
                    result = json.loads(raw)
                    parsed_ok = True
                    chart_ids = result.get('charts', [])
                    st.session_state.messages.append({'role': 'assistant', 'content': result.get('insight', raw)})
                    st.session_state.visual_history.append({'charts': chart_ids})
                except json.JSONDecodeError:
                    st.session_state.messages.append({'role': 'assistant', 'content': raw})
                    st.session_state.visual_history.append(None)
            
            except Exception as e:
                st.error(f"OpenAI error: {e}")
        
        # ── Render everything OUTSIDE the chat bubble at full page width ──
        if parsed_ok and result:
            # Headline KPI card
            if 'headline' in result and result['headline']:
                h = result['headline']
                st.markdown(f"""
                <div class="kpi-card accent" style="margin-bottom: 16px; max-width: 500px;">
                    <div class="kpi-label">{h.get('label', '')}</div>
                    <div class="kpi-value">{h.get('value', '')}</div>
                    <div class="kpi-sub">{h.get('context', '')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Insight text
            if 'insight' in result:
                safe_text = result['insight'].replace('$', '\\$')
                st.markdown(safe_text)
            
            # Charts at full width
            if chart_ids:
                st.markdown("&nbsp;")
                cols = st.columns(len(chart_ids)) if len(chart_ids) > 1 else [st.container()]
                for j, cid in enumerate(chart_ids):
                    if cid in CHART_CATALOG:
                        with cols[j]:
                            st.markdown(f"**{CHART_CATALOG[cid]['title']}**")
                            st.plotly_chart(CHART_CATALOG[cid]['fn'](), use_container_width=True)
            
            # Blended market context — appears as part of the analysis, not a separate section
            if client:
                st.markdown("---")
                st.markdown("**Connecting to the broader B.C. market:**")
                with st.spinner(""):
                    try:
                        web_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": (
                                    "You are a B.C. agriculture market analyst writing a brief addendum to a data analysis. "
                                    "Given the farmer's question and the GreenLeaf dataset findings, provide 2-3 SHORT bullet points "
                                    "connecting the analysis to the broader B.C. greenhouse market. "
                                    "Focus on: current B.C. produce pricing, input cost trends, relevant provincial or federal programs, "
                                    "and how the findings compare to industry benchmarks. "
                                    "CRITICAL RULES: "
                                    "- NEVER mention dates, years, or time periods for your knowledge. "
                                    "- NEVER say 'as of', 'in 2023', 'in 2024', 'recently', or any temporal qualifier about your sources. "
                                    "- Write as if this is current, established market knowledge. "
                                    "- Do NOT use dollar signs — write 'CAD' instead. "
                                    "- Keep each bullet under 30 words. Start each with a bold label. "
                                    "- Frame everything as context that strengthens or qualifies the dataset findings."
                                )},
                                {"role": "user", "content": f"The farmer asked: '{prompt}'. Connect GreenLeaf's findings to B.C. greenhouse market context."},
                            ],
                            max_tokens=250,
                            temperature=0.3,
                        )
                        web_text = web_response.choices[0].message.content.strip()
                        web_text = web_text.replace('$', '\\$')
                        st.markdown(web_text)
                    except Exception:
                        pass
        
        elif raw:
            st.markdown(raw.replace('$', '\\$'))
    
    # Suggested questions as clickable buttons
    st.markdown("---")
    st.markdown("**Try asking:**")
    suggestions = [
        "Which treatment is most profitable for strawberries?",
        "How quickly should we respond to alerts?",
        "Should a farmer invest in precision agriculture?",
        "What are the most ignored alert types?",
        "Show me the cost breakdown by treatment",
        "Which plots need attention right now?",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            st.caption(f'*"{s}"*')
