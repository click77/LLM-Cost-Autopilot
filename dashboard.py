import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configure the page properties for a clean, broad UI canvas
st.set_page_config(page_title="Cost Autopilot Performance Dashboard", layout="wide")

DB_PATH = "autopilot_analytics.db"

@st.cache_data(ttl=10)
def load_telemetry_data():
    """Reads telemetry from the audit log database and applies business intelligence mappings."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM request_logs", conn)
    
    # Clean and explicitly cast timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.to_period('W').dt.start_time.dt.date
    
    # Back-calculate what the identical traffic would have cost if routed to GPT-4o entirely
    # GPT-4o is ~25x more expensive than GPT-4o-mini and ~30x more expensive than Claude Haiku
    def compute_gpt4o_baseline(row):
        model = row['routed_model'].lower()
        if 'gpt-4o-mini' in model:
            return row['cost'] * 25.0
        elif 'haiku' in model or 'llama' in model:
            return row['cost'] * 30.0
        else:
            return row['cost'] # Already routed to GPT-4o/highest tier
            
    df['gpt4o_baseline_cost'] = df.apply(compute_gpt4o_baseline, axis=1)
    return df

# Initialize UI text layouts
st.title("📊 Cost Autopilot AI Infrastructure Analytics")
st.markdown("Real-time cost reduction, routing topology, and quality guardrail audit analytics.")

try:
    df = load_telemetry_data()
    
    if df.empty:
        st.warning("Telemetry database discovered but contains zero transaction records. Run the seed script to view analytics.")
        st.stop()
        
    # ==============================================================================
    # THE HEADLINE "MONEY SHOT" METRICS
    # ==============================================================================
    total_actual_cost = df['cost'].sum()
    total_baseline_cost = df['gpt4o_baseline_cost'].sum()
    net_savings = total_baseline_cost - total_actual_cost
    savings_percentage = (net_savings / total_baseline_cost) * 100 if total_baseline_cost > 0 else 0

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Traffic volume", value=f"{len(df):,} Requests")
    with col2:
        st.metric(label="Actual API Spend", value=f"${total_actual_cost:.2f}")
    with col3:
        st.metric(label="Unoptimized Spend (GPT-4o Baseline)", value=f"${total_baseline_cost:.2f}")
    with col4:
        st.metric(
            label="🚀 Headline Cost Reduction", 
            value=f"{savings_percentage:.1f}%", 
            delta=f"${net_savings:.2f} Saved",
            delta_color="normal"
        )
    st.markdown("---")

    # Sidebar parameters for time-grain toggle
    st.sidebar.header("Dashboard Filters")
    time_grain = st.sidebar.selectbox("Cost Panel Time Grouping", options=["Daily", "Weekly"])

    # ==============================================================================
    # PANEL 1: TOTAL COST COMPARISON OVER TIME (ACTUAL VS. GPT-4o BASELINE)
    # ==============================================================================
    st.subheader("1. Cost Comparison Over Time vs. Unoptimized Baseline")
    group_col = 'day' if time_grain == "Daily" else 'week'
    
    cost_trend = df.groupby(group_col)[['cost', 'gpt4o_baseline_cost']].sum().reset_index()
    
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x=cost_trend[group_col], y=cost_trend['cost'],
        name='Actual Spend (Autopilot)', marker_color='#1E88E5'
    ))
    fig_cost.add_trace(go.Scatter(
        x=cost_trend[group_col], y=cost_trend['gpt4o_baseline_cost'],
        name='Baseline Spend (All requests to GPT-4o)', line=dict(color='#E53935', width=3, dash='dash')
    ))
    fig_cost.update_layout(
        xaxis_title="Timeline Interval", yaxis_title="Cost in USD ($)",
        barmode='group', hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0)
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    # Secondary split rows layout for specific metric cards
    left_col, right_col = st.columns(2)

    # ==============================================================================
    # PANEL 2: ROUTING DISTRIBUTION PIE CHART
    # ==============================================================================
    with left_col:
        st.subheader("2. Model Routing Topology Distribution")
        model_counts = df['routed_model'].value_counts().reset_index()
        model_counts.columns = ['Model Name', 'Request Count']
        
        fig_pie = px.pie(
            model_counts, values='Request Count', names='Model Name',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # ==============================================================================
    # PANEL 3: QUALITY SCORE DISTRIBUTION HISTOGRAM
    # ==============================================================================
    with right_col:
        st.subheader("3. Verification Quality Score Distribution")
        
        fig_hist = px.histogram(
            df, x='quality_score', nbins=20,
            labels={'quality_score': 'Verifier Quality Score (1.0 - 5.0)'},
            color_discrete_sequence=['#43A047']
        )
        fig_hist.update_layout(
            yaxis_title="Volume Count",
            bargap=0.05,
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ==============================================================================
    # PANEL 4: ESCALATION RATE OVER TIME
    # ==============================================================================
    st.subheader("4. Guardrail Escalation Rate Over Time")
    
    escalation_trend = df.groupby('day').agg(
        total_count=('id', 'count'),
        escalated_count=('was_escalated', 'sum')
    ).reset_index()
    
    escalation_trend['Escalation Rate (%)'] = (escalation_trend['escalated_count'] / escalation_trend['total_count']) * 100
    
    fig_esc = px.line(
        escalation_trend, x='day', y='Escalation Rate (%)',
        markers=True, line_shape='linear'
    )
    fig_esc.update_traces(line_color='#FB8C00', width=2.5)
    fig_esc.update_layout(
        xaxis_title="Date", yaxis_title="Escalation Rate (%)",
        hovermode="x"
    )
    st.plotly_chart(fig_esc, use_container_width=True)

except Exception as ex:
    st.error(f"Error accessing database or rendering analytics panels: {str(ex)}")
