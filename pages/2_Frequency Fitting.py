import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Step 3: Claim Count Modeling (Poisson Distribution)")

# Check that required data exists
if "exposure_data" not in st.session_state:
    st.warning("Please upload exposure data on the home page first.")
    st.stop()

if "final_distribution" not in st.session_state:
    st.warning("Please complete severity distribution fitting on Step 2 first.")
    st.stop()

# Get the data
exposure_data = st.session_state["exposure_data"]
final_severity = st.session_state["final_distribution"]

st.success(f"✅ Severity Distribution: {final_severity['name']} with parameters {final_severity['parameters']}")

# Display exposure data
st.subheader("Exposure Data Overview")
st.dataframe(exposure_data)

# Extract the relevant columns
years = exposure_data["Year"].astype(int).values
projected_exposure = exposure_data["Projected exposure"].values  
original_counts = exposure_data["Projected Claim Count"].values


# --- NEW: Manual entry option with persistence ---
st.subheader("Projected Claim Counts")

# Initialize session state to remember manual entry toggle and data
if "use_manual_counts" not in st.session_state:
    st.session_state["use_manual_counts"] = False
if "manual_counts" not in st.session_state:
    st.session_state["manual_counts"] = original_counts.tolist()

# Checkbox to toggle manual entry
use_manual_counts = st.checkbox(
    "✏️ Manually enter your own projected claim counts instead of using the file values",
    value=st.session_state["use_manual_counts"]
)

# Update session state when checkbox changes
st.session_state["use_manual_counts"] = use_manual_counts

# Show numeric inputs if manual entry is enabled
if st.session_state["use_manual_counts"]:
    st.info("Enter projected claim counts for each year:")
    updated_counts = []
    for i, year in enumerate(years):
        value = st.number_input(
            f"Projected Claims for Year {int(year)}",
            min_value=0.0,
            value=float(st.session_state["manual_counts"][i]),
            step=1.0,
            format="%.0f",
            key=f"manual_count_{i}"
        )
        updated_counts.append(value)
    # Save user input persistently
    st.session_state["manual_counts"] = updated_counts
    projected_counts = np.array(updated_counts)
else:
    projected_counts = np.array(original_counts)


# Simple time series visualization
st.subheader("Annual Claim Count Projections")
st.caption("Using manual entries" if st.session_state["use_manual_counts"] else "Using original exposure data")


# Create subplot with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add projected claim counts (line)
fig.add_trace(
    go.Scatter(x=years, y=projected_counts, mode='lines+markers',
               name='Projected Claims', line=dict(color='blue', width=3),
               marker=dict(size=8)),
    secondary_y=False,
)

# Add projected exposure (bar chart)
fig.add_trace(
    go.Bar(x=years, y=projected_exposure,
           name='Projected Exposure', 
           marker=dict(color='lightcoral', opacity=0.6),
           width=0.4),
    secondary_y=True,
)

# Set x-axis title
fig.update_xaxes(title_text="Year", dtick=1)

# Set y-axes titles
fig.update_yaxes(title_text="Expected Claims", secondary_y=False)
fig.update_yaxes(title_text="Projected Exposure ($)", secondary_y=True)

fig.update_layout(
    title="Expected Claim Counts and Projected Exposure by Year",
    height=400,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig, width="stretch")

# Simulation Options
st.subheader("Poisson Simulation Parameters")

col1, col2 = st.columns(2)

with col1:
    num_simulations = st.number_input(
        "Number of Simulations",
        min_value=100,
        max_value=100000,
        value=10000,
        step=100,
        help="Number of Monte Carlo simulations to run"
    )

with col2:
    confidence_level = st.selectbox(
        "Confidence Level",
        options=[0.90, 0.95, 0.99],
        index=1,
        help="Confidence level for percentile calculations"
    )

# Run Simulations
if st.button("🎲 Run Poisson Simulations", type="primary"):
    with st.spinner("Running simulations..."):
        
        # Run simulations for each year
        simulation_results = {}
        total_simulations = np.zeros(num_simulations)
        
        for i, year in enumerate(years):
            lambda_param = projected_counts[i]
            # Generate Poisson random variables
            simulated_counts = np.random.poisson(lambda_param, num_simulations)
            simulation_results[int(year)] = simulated_counts
            total_simulations += simulated_counts
        
        # Store simulation results in session state
        st.session_state["claim_count_simulations"] = {
            "total_simulations": total_simulations,
            "yearly_simulations": simulation_results,
            "num_simulations": num_simulations,
            "confidence_level": confidence_level,
            "years": years.tolist(),
            "projected_counts": projected_counts.tolist(),
            "use_manual_counts": st.session_state["use_manual_counts"]
        }
        
        st.success("✅ Simulations completed for all years!")

if "claim_count_simulations" in st.session_state:
    stored_years = [
        int(y)
        for y in st.session_state["claim_count_simulations"]["years"]
    ]

    current_years = [int(y) for y in years]

    if stored_years != current_years:
        del st.session_state["claim_count_simulations"]
        st.info(
            "Input years changed. Previous claim count simulations "
            "were cleared. Please run the simulation again."
        )

# Display results if simulations exist
if "claim_count_simulations" in st.session_state:
    sim_data = st.session_state["claim_count_simulations"]
    
    st.info(f"💾 Saved: {sim_data['num_simulations']:,} simulations at {sim_data['confidence_level']*100:.0f}% confidence level")
    
    # Results display
    st.subheader("Simulation Results")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🗑️ Clear Simulation Results"):
            del st.session_state["claim_count_simulations"]
            st.rerun()
    
    confidence_level = sim_data['confidence_level']
    
    # Calculate percentiles
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    # Simulation Results Summary - All Years with Years as Columns
    st.subheader("Simulation Results Summary by Year")
    
    # Create statistics with years as columns
    stats_data = {
        "Metric": ["Mean", "Std Dev", f"{confidence_level*100:.0f}% VaR",
                  f"P{lower_percentile:.0f}", f"P{upper_percentile:.0f}", "Min", "Max"]
    }
    
    # Add each year as a column
    for year in years:
        year_results = sim_data['yearly_simulations'][int(year)]
        
        stats_data[f"Year {int(year)}"] = [
            f"{np.mean(year_results):.1f}",
            f"{np.std(year_results):.1f}",
            f"{np.percentile(year_results, upper_percentile):.0f}",
            f"{np.percentile(year_results, lower_percentile):.0f}",
            f"{np.percentile(year_results, upper_percentile):.0f}",
            f"{np.min(year_results):.0f}",
            f"{np.max(year_results):.0f}"
        ]
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, width="stretch")
    
    # Distribution of Claims - All Years in Single Plot
    st.subheader("Distribution of Simulated Claims by Year")
    
    # Create single plot with all years overlaid
    fig_hist = go.Figure()
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, year in enumerate(years):
        year_results = sim_data['yearly_simulations'][year]
        
        fig_hist.add_trace(go.Histogram(
            x=year_results,
            nbinsx=30,
            name=f"Year {int(year)}",
            opacity=0.6,
            marker_color=colors[i % len(colors)],
            histnorm='probability density'  # Normalize to show probability density
        ))
    
    fig_hist.update_layout(
        title="Distribution of Simulated Claims - All Years",
        xaxis_title="Claims",
        yaxis_title="Probability Density",
        height=500,
        barmode='overlay',  # Overlay histograms
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_hist, width="stretch")