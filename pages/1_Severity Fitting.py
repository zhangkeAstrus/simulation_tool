import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


st.title("Step 2: Severity Distribution Fitting")

# Check that loss data exists
if "loss_data" not in st.session_state:
    st.warning("Please upload data on the home page first.")
    st.stop()

loss_data = st.session_state["loss_data"]
loss_values = loss_data[loss_data["Loss"]>0].iloc[:, 0].astype(float).values

# Distributions to test
distributions = {
    "Lognormal": stats.lognorm,
    "Gamma": stats.gamma,
    "Exponential": stats.expon,
    "Inverse Gaussian": stats.invgauss,
}

# Fit and evaluate each distribution
results = []
st.subheader("Fitting distributions...")

for name, dist in distributions.items():
    try:
        # Fit distribution (fix location = 0 for loss modeling)
        params = dist.fit(loss_values, floc=0)
        ks_stat, ks_p = stats.kstest(loss_values, dist.name, args=params)

        # --- NEW: Log-likelihood, AIC, BIC ---
        pdf_values = dist.pdf(loss_values, *params)
        pdf_values = np.clip(pdf_values, 1e-12, None)  # avoid log(0)
        log_likelihood = np.sum(np.log(pdf_values))
        k = len(params)
        n = len(loss_values)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        results.append((name, ks_stat,aic, bic, params))

    except Exception as e:
        results.append((name, None, None, None, str(e)))


# Show results
results_df = pd.DataFrame(
    results,
    columns=["Distribution", "KS Statistic", "AIC", "BIC", "Parameters"]
)


# --- NEW: Clean up numeric formatting ---
results_df["KS Statistic"] = results_df["KS Statistic"].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
results_df["AIC"] = results_df["AIC"].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
results_df["BIC"] = results_df["BIC"].apply(lambda x: round(x, 2) if pd.notnull(x) else x)


# Sort by best fit — you can sort by KS or AIC; here we keep KS default
results_df = results_df.sort_values("KS Statistic")

st.dataframe(results_df)

# Best fit
best_fit = results_df.iloc[0]
st.success(f"Best fit: {best_fit['Distribution']} (KS Statistic = {best_fit['KS Statistic']:.4f})")

st.subheader("Empirical vs Fitted Statistics for All Distributions")

# Empirical statistics
empirical_stats = {
    "Minimum": np.min(loss_values),
    "Maximum": np.max(loss_values),
    "Mean": np.mean(loss_values),
    "Median": np.median(loss_values),
    "Mode": pd.Series(loss_values).mode().iloc[0] if not pd.Series(loss_values).mode().empty else np.nan,
    "Std Dev": np.std(loss_values, ddof=1),
    "Skewness": stats.skew(loss_values),
    "Kurtosis": stats.kurtosis(loss_values),
}
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
empirical_quantiles = {f"Q{int(q*100)}": np.quantile(loss_values, q) for q in quantiles}

# Combine all into a single structure
summary_table = pd.DataFrame()
summary_table["Empirical"] = {**empirical_stats, **empirical_quantiles}

# Loop through fitted distributions
for name, dist in distributions.items():
    # Skip if fitting failed
    fit_row = results_df[results_df["Distribution"] == name].iloc[0]
    if isinstance(fit_row["Parameters"], str):  # Indicates error
        continue
    
    params = fit_row["Parameters"]
    
    try:
        fitted_stats = {
            "Minimum": dist.ppf(0.0001, *params),
            "Maximum": dist.ppf(0.9999, *params),
            "Mean": dist.mean(*params),
            "Median": dist.median(*params),
            "Mode": dist.ppf(0.01, *params),  # Approximate mode
            "Std Dev": dist.std(*params),
            "Skewness": dist.stats(*params, moments='s'),
            "Kurtosis": dist.stats(*params, moments='k'),
        }
        fitted_quantiles = {f"Q{int(q*100)}": dist.ppf(q, *params) for q in quantiles}
        summary_table[name] = {**fitted_stats, **fitted_quantiles}
    except Exception as e:
        st.warning(f"Failed to compute fitted stats for {name}: {e}")

st.dataframe(summary_table)

# Add probability distribution plot with fitted distributions
st.subheader("Probability Distribution of Actual Loss Data with Fitted Distributions")

# Create probability plot using Plotly
try:
    # Create Plotly figure
    fig = go.Figure()
    
    # Add histogram showing probability (not density)
    fig.add_trace(go.Histogram(
        x=loss_values,
        histnorm='probability',
        name='Actual Data',
        opacity=0.7,
        nbinsx=min(15, len(loss_values)//3),  # Fewer bins for small datasets
        marker_color='lightblue',
        marker_line_color='darkblue',
        marker_line_width=1
    ))
    
    # Add fitted distribution curves
    x_min, x_max = np.min(loss_values), np.max(loss_values)
    n_bins = min(15, len(loss_values)//3)
    
    # Create bin edges
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    colors = ['red', 'green', 'orange', 'purple']
    
    for i, (name, dist) in enumerate(distributions.items()):
        # Get fitted parameters for this distribution
        fit_row = results_df[results_df["Distribution"] == name].iloc[0]
        if isinstance(fit_row["Parameters"], str):  # Skip if fitting failed
            continue
            
        params = fit_row["Parameters"]
        
        try:
            # Calculate probability for each bin using CDF differences
            bin_probabilities = []
            for j in range(len(bin_edges) - 1):
                prob = dist.cdf(bin_edges[j+1], *params) - dist.cdf(bin_edges[j], *params)
                bin_probabilities.append(prob)
            
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=bin_probabilities,
                mode='lines+markers',
                name=f'{name}',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                opacity=0.8
            ))
        except Exception as e:
            st.warning(f"Could not plot {name}: {e}")
    
    fig.update_layout(
        title="Probability Distribution: Actual Data vs Fitted Distributions",
        xaxis_title="Loss Amount",
        yaxis_title="Probability",
        showlegend=True,
        height=500,
        yaxis=dict(
            tickformat='.3f',  # Show probabilities with 3 decimal places
            range=[0, 1]  # Fix y-axis scale from 0 to 1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show basic statistics
    st.write(f"**Data Summary:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Count", len(loss_values))
    with col2:
        st.metric("Mean", f"{np.mean(loss_values):,.2f}")
    with col3:
        st.metric("Std Dev", f"{np.std(loss_values, ddof=1):,.2f}")
    with col4:
        st.metric("Max", f"{np.max(loss_values):,.2f}")

except Exception as e:
    st.error(f"Error creating probability plot: {e}")
    # Fallback to simple histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(loss_values, bins=min(15, len(loss_values)//3), density=False, alpha=0.7, color='lightblue', edgecolor='black')
    ax.set_xlabel('Loss Amount')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Loss Data')
    st.pyplot(fig)

# Finalize Distribution Selection
st.subheader("Finalize Distribution Selection")

# Get successfully fitted distributions for selection
successful_fits = results_df[~results_df["Parameters"].astype(str).str.contains("Error|exception", case=False, na=False)]
distribution_names = successful_fits["Distribution"].tolist()

if len(distribution_names) > 0:
    # Select distribution
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_dist_name = st.selectbox(
            "Select Final Distribution:",
            options=distribution_names,
            index=0,  # Default to best fit (first in sorted list)
            help="Choose the distribution you want to use for your analysis"
        )
    
    with col2:
        # Show fit quality for selected distribution
        selected_row = successful_fits[successful_fits["Distribution"] == selected_dist_name].iloc[0]
        st.metric(
            "KS Test Statistic", 
            f"{selected_row['KS Statistic']:.4f}",
            help="Lower values indicate better fit"
        )

        # --- NEW: Show additional fit metrics ---
        if "AIC" in selected_row and not pd.isna(selected_row["AIC"]):
            st.metric("AIC", f"{selected_row['AIC']:.2f}", help="Lower AIC indicates a better fit (adjusted for model complexity)")
        if "BIC" in selected_row and not pd.isna(selected_row["BIC"]):
            st.metric("BIC", f"{selected_row['BIC']:.2f}", help="Lower BIC indicates a better fit (stronger penalty for complexity)")


    
    # Get the selected distribution and its fitted parameters
    selected_dist = distributions[selected_dist_name]
    fitted_params = selected_row["Parameters"]
    
    st.write(f"**Fitted Parameters for {selected_dist_name} Distribution:**")
    
    # Create editable parameter inputs based on distribution type
    if selected_dist_name == "Lognormal":
        # Lognormal: (s, loc, scale) where s=sigma, loc=shift (fixed at 0), scale=exp(mu)
        s_param = st.number_input("Shape (σ)", value=float(fitted_params[0]), format="%.6f", help="Standard deviation of underlying normal")
        st.write("Location: 0.000000 (fixed for loss modeling)")
        scale_param = st.number_input("Scale", value=float(fitted_params[2]), format="%.6f", help="Scale parameter")
        final_params = (s_param, 0.0, scale_param)
        
    elif selected_dist_name == "Gamma":
        # Gamma: (a, loc, scale) where a=shape, loc=shift (fixed at 0), scale=scale
        a_param = st.number_input("Shape (α)", value=float(fitted_params[0]), format="%.6f", help="Shape parameter")
        st.write("Location: 0.000000 (fixed for loss modeling)")
        scale_param = st.number_input("Scale (β)", value=float(fitted_params[2]), format="%.6f", help="Scale parameter")
        final_params = (a_param, 0.0, scale_param)
        
    elif selected_dist_name == "Exponential":
        # Exponential: (loc, scale) where loc=shift (fixed at 0), scale=1/lambda
        st.write("Location: 0.000000 (fixed for loss modeling)")
        scale_param = st.number_input("Scale (1/λ)", value=float(fitted_params[1]), format="%.6f", help="Scale parameter (1/rate)")
        final_params = (0.0, scale_param)
        
    elif selected_dist_name == "Inverse Gaussian":
        # Inverse Gaussian: (mu, loc, scale) where loc=shift (fixed at 0)
        mu_param = st.number_input("Shape (μ)", value=float(fitted_params[0]), format="%.6f", help="Shape parameter")
        st.write("Location: 0.000000 (fixed for loss modeling)")
        scale_param = st.number_input("Scale", value=float(fitted_params[2]), format="%.6f", help="Scale parameter")
        final_params = (mu_param, 0.0, scale_param)
    
    # Display theoretical statistics with current parameters
    try:
        st.write("**Theoretical Statistics with Current Parameters:**")
        theo_mean = selected_dist.mean(*final_params)
        theo_std = selected_dist.std(*final_params)
        theo_median = selected_dist.median(*final_params)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Theoretical Mean", f"{theo_mean:,.2f}")
        with col2:
            st.metric("Theoretical Std Dev", f"{theo_std:,.2f}")
        with col3:
            st.metric("Theoretical Median", f"{theo_median:,.2f}")
            
    except Exception as e:
        st.warning(f"Could not calculate theoretical statistics: {e}")
    
    # Finalize button
    if st.button("🎯 Finalize Distribution Selection", type="primary"):
        # Store the final distribution and parameters in session state
        st.session_state["final_distribution"] = {
            "name": selected_dist_name,
            "distribution": selected_dist,
            "parameters": final_params,
            "ks_statistic": selected_row['KS Statistic'],
        }
        
        st.success(f"✅ **Distribution Finalized!**")
        st.success(f"Selected: **{selected_dist_name}** with parameters {final_params}")
        st.info("You can now proceed to the next step of your analysis.")
        
        # Optional: Show what was saved
        with st.expander("View Saved Distribution Details"):
            st.json({
                "Distribution": selected_dist_name,
                "Parameters": [float(p) for p in final_params],
                "KS Statistic": float(selected_row['KS Statistic']),
                "Theoretical Mean": float(theo_mean) if 'theo_mean' in locals() else None,
                "Theoretical Std Dev": float(theo_std) if 'theo_std' in locals() else None
            })

else:
    st.error("No distributions were successfully fitted. Please check your data.")