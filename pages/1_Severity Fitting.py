import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils import calculate_limited_mean, solve_scale_for_limited_mean

st.title("Step 2: Severity Distribution Fitting")

if "loss_data" not in st.session_state:
    st.warning("Please upload data on the home page first.")
    st.stop()

loss_data = st.session_state["loss_data"]
loss_values = pd.to_numeric(loss_data["Loss"], errors="coerce")
loss_values = loss_values[loss_values > 0].dropna().astype(float).values

if len(loss_values) == 0:
    st.error("No positive loss values were found in the selected loss column.")
    st.stop()

# Distributions to test
distributions = {
    "Lognormal": stats.lognorm,
    "Gamma": stats.gamma,
    "Exponential": stats.expon,
    "Inverse Gaussian": stats.invgauss,
    "Log-Logistic": stats.fisk
}

# Fit and evaluate each distribution
results = []
st.subheader("Fitting distributions...")

for name, dist in distributions.items():
    try:
        params = dist.fit(loss_values, floc=0)
        ks_stat, _ = stats.kstest(loss_values, dist.name, args=params)

        pdf_values = np.clip(dist.pdf(loss_values, *params), 1e-12, None)
        log_likelihood = np.sum(np.log(pdf_values))

        # Location is fixed at zero, so it is not an estimated parameter.
        k = len(params) - 1
        n = len(loss_values)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        results.append((name, ks_stat, aic, bic, params))
    except Exception as e:
        results.append((name, None, None, None, f"Error: {e}"))

results_df = pd.DataFrame(
    results,
    columns=["Distribution", "KS Statistic", "AIC", "BIC", "Parameters"],
)

results_df["KS Statistic"] = results_df["KS Statistic"].apply(
    lambda x: round(x, 4) if pd.notnull(x) else x
)
results_df["AIC"] = results_df["AIC"].apply(
    lambda x: round(x, 2) if pd.notnull(x) else x
)
results_df["BIC"] = results_df["BIC"].apply(
    lambda x: round(x, 2) if pd.notnull(x) else x
)
results_df = results_df.sort_values("KS Statistic", na_position="last")

st.dataframe(results_df, width="stretch")

successful_fits = results_df[results_df["Parameters"].apply(lambda x: isinstance(x, tuple))]

if successful_fits.empty:
    st.error("No distributions were successfully fitted. Please check your data.")
    st.stop()

best_fit = successful_fits.iloc[0]
st.success(
    f"Best fit: {best_fit['Distribution']} "
    f"(KS Statistic = {best_fit['KS Statistic']:.4f})"
)

# Empirical and fitted statistics
st.subheader("Empirical vs Fitted Statistics for All Distributions")

empirical_stats = {
    "Minimum": np.min(loss_values),
    "Maximum": np.max(loss_values),
    "Mean": np.mean(loss_values),
    "Median": np.median(loss_values),
    "Mode": pd.Series(loss_values).mode().iloc[0]
    if not pd.Series(loss_values).mode().empty
    else np.nan,
    "Std Dev": np.std(loss_values, ddof=1),
    "Skewness": stats.skew(loss_values),
    "Kurtosis": stats.kurtosis(loss_values),
}

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
empirical_quantiles = {
    f"Q{int(q * 100)}": np.quantile(loss_values, q) for q in quantiles
}

summary_table = pd.DataFrame()
summary_table["Empirical"] = {**empirical_stats, **empirical_quantiles}

for name, dist in distributions.items():
    fit_rows = successful_fits[successful_fits["Distribution"] == name]
    if fit_rows.empty:
        continue

    params = fit_rows.iloc[0]["Parameters"]
    try:
        fitted_stats = {
            "Minimum": dist.ppf(0.0001, *params),
            "Maximum": dist.ppf(0.9999, *params),
            "Mean": dist.mean(*params),
            "Median": dist.median(*params),
            "Mode": np.nan,
            "Std Dev": dist.std(*params),
            "Skewness": dist.stats(*params, moments="s"),
            "Kurtosis": dist.stats(*params, moments="k"),
        }
        fitted_quantiles = {
            f"Q{int(q * 100)}": dist.ppf(q, *params) for q in quantiles
        }
        summary_table[name] = {**fitted_stats, **fitted_quantiles}
    except Exception as e:
        st.warning(f"Failed to compute fitted stats for {name}: {e}")

st.dataframe(summary_table, width="stretch")

# Probability plot
st.subheader("Probability Distribution of Actual Loss Data with Fitted Distributions")

try:
    fig = go.Figure()
    n_bins = max(1, min(15, len(loss_values) // 3))

    fig.add_trace(
        go.Histogram(
            x=loss_values,
            histnorm="probability",
            name="Actual Data",
            opacity=0.7,
            nbinsx=n_bins,
            marker_color="lightblue",
            marker_line_color="darkblue",
            marker_line_width=1,
        )
    )

    x_min, x_max = np.min(loss_values), np.max(loss_values)
    if x_min == x_max:
        x_max = x_min + 1

    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    colors = ["red", "green", "orange", "purple"]

    for i, (name, dist) in enumerate(distributions.items()):
        fit_rows = successful_fits[successful_fits["Distribution"] == name]
        if fit_rows.empty:
            continue

        params = fit_rows.iloc[0]["Parameters"]
        bin_probabilities = [
            dist.cdf(bin_edges[j + 1], *params) - dist.cdf(bin_edges[j], *params)
            for j in range(len(bin_edges) - 1)
        ]

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=bin_probabilities,
                mode="lines+markers",
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6),
                opacity=0.8,
            )
        )

    fig.update_layout(
        title="Probability Distribution: Actual Data vs Fitted Distributions",
        xaxis_title="Loss Amount",
        yaxis_title="Probability",
        showlegend=True,
        height=500,
        yaxis=dict(tickformat=".3f", range=[0, 1]),
    )
    st.plotly_chart(fig, width="stretch")

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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(loss_values, bins=max(1, min(15, len(loss_values) // 3)))
    ax.set_xlabel("Loss Amount")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Loss Data")
    st.pyplot(fig)

# Final distribution selection
st.subheader("Finalize Distribution Selection")
distribution_names = successful_fits["Distribution"].tolist()

col1, col2 = st.columns([1, 2])
with col1:
    selected_dist_name = st.selectbox(
        "Select Final Distribution:",
        options=distribution_names,
        index=0,
        help="Choose the distribution you want to use for your analysis",
    )

selected_row = successful_fits[
    successful_fits["Distribution"] == selected_dist_name
].iloc[0]

with col2:
    st.metric("KS Test Statistic", f"{selected_row['KS Statistic']:.4f}")
    if not pd.isna(selected_row["AIC"]):
        st.metric("AIC", f"{selected_row['AIC']:.2f}")
    if not pd.isna(selected_row["BIC"]):
        st.metric("BIC", f"{selected_row['BIC']:.2f}")

selected_dist = distributions[selected_dist_name]
fitted_params = selected_row["Parameters"]

st.write(f"**Fitted Parameters for {selected_dist_name} Distribution:**")

if selected_dist_name == "Lognormal":
    shape_param = st.number_input(
        "Shape (σ)", value=float(fitted_params[0]), format="%.6f"
    )
    st.write("Location: 0.000000 (fixed for loss modeling)")
    scale_param = st.number_input(
        "Scale", value=float(fitted_params[2]), format="%.6f"
    )
    manual_params = (shape_param, 0.0, scale_param)

elif selected_dist_name == "Gamma":
    shape_param = st.number_input(
        "Shape (α)", value=float(fitted_params[0]), format="%.6f"
    )
    st.write("Location: 0.000000 (fixed for loss modeling)")
    scale_param = st.number_input(
        "Scale (β)", value=float(fitted_params[2]), format="%.6f"
    )
    manual_params = (shape_param, 0.0, scale_param)

elif selected_dist_name == "Exponential":
    st.write("Location: 0.000000 (fixed for loss modeling)")
    scale_param = st.number_input(
        "Scale (1/λ)", value=float(fitted_params[1]), format="%.6f"
    )
    manual_params = (0.0, scale_param)

elif selected_dist_name == "Log-Logistic":
    shape_param = st.number_input(
        "Shape (c)",
        value=float(fitted_params[0]),
        format="%.6f"
    )

    st.write("Location: 0.000000 (fixed for loss modeling)")

    scale_param = st.number_input(
        "Scale",
        value=float(fitted_params[2]),
        format="%.6f"
    )

    manual_params = (
        shape_param,
        0.0,
        scale_param
    )
else:  # Inverse Gaussian
    shape_param = st.number_input(
        "Shape (μ)", value=float(fitted_params[0]), format="%.6f"
    )
    st.write("Location: 0.000000 (fixed for loss modeling)")
    scale_param = st.number_input(
        "Scale", value=float(fitted_params[2]), format="%.6f"
    )
    manual_params = (shape_param, 0.0, scale_param)

    

# Optional limited-severity calibration
st.divider()
st.subheader("Optional Limited Severity Calibration")
use_limited_calibration = st.checkbox(
    "Calibrate scale to a target per-claim limited severity",
    help=(
        "Keeps the selected distribution's shape parameter(s) unchanged and "
        "solves only for scale."
    ),
)

final_params = manual_params
limited_calibration = None

if use_limited_calibration:
    c1, c2 = st.columns(2)
    with c1:
        calibration_limit = st.number_input(
            "Per-Claim Limit",
            min_value=1.0,
            value=500000.0,
            step=50000.0,
            format="%.0f",
        )
    with c2:
        target_limited_mean = st.number_input(
            "Target Limited Mean Severity",
            min_value=1.0,
            value=40000.0,
            step=1000.0,
            format="%.2f",
        )

    try:
        current_limited_mean = calculate_limited_mean(
            selected_dist, manual_params, calibration_limit
        )
        calibrated_scale = solve_scale_for_limited_mean(
            selected_dist,
            manual_params,
            calibration_limit,
            target_limited_mean,
        )
        final_params = manual_params[:-1] + (calibrated_scale,)
        calibrated_limited_mean = calculate_limited_mean(
            selected_dist, final_params, calibration_limit
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Current Limited Mean", f"${current_limited_mean:,.2f}")
        with m2:
            st.metric("Required Scale", f"{calibrated_scale:,.6f}")
        with m3:
            st.metric("Calibrated Limited Mean", f"${calibrated_limited_mean:,.2f}")

        st.caption("The shape parameter(s) are unchanged; only scale is recalibrated.")

        limited_calibration = {
            "limit": float(calibration_limit),
            "target_mean": float(target_limited_mean),
            "actual_mean": float(calibrated_limited_mean),
        }
    except Exception as e:
        st.error(f"Limited severity calibration failed: {e}")
        final_params = manual_params
        
# Warn about Log-Logistic tail properties
if selected_dist_name == "Log-Logistic":
    c = final_params[0]

    if c <= 1:
        st.warning(
            "The fitted Log-Logistic has shape <= 1, "
            "so the theoretical mean is infinite."
        )
    elif c <= 2:
        st.warning(
            "The fitted Log-Logistic has finite mean "
            "but infinite variance."
        )
# Statistics for the parameters that will actually be finalized
try:
    st.write("**Theoretical Statistics with Final Parameters:**")
    theo_mean = selected_dist.mean(*final_params)
    theo_std = selected_dist.std(*final_params)
    theo_median = selected_dist.median(*final_params)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Theoretical Mean", f"{theo_mean:,.2f}")
    with c2:
        st.metric("Theoretical Std Dev", f"{theo_std:,.2f}")
    with c3:
        st.metric("Theoretical Median", f"{theo_median:,.2f}")
except Exception as e:
    st.warning(f"Could not calculate theoretical statistics: {e}")

if st.button("🎯 Finalize Distribution Selection", type="primary"):
    st.session_state["final_distribution"] = {
        "name": selected_dist_name,
        "distribution": selected_dist,
        "parameters": final_params,
        "ks_statistic": selected_row["KS Statistic"],
        "limited_calibration": limited_calibration,
    }

    st.success("✅ Distribution Finalized!")
    st.success(f"Selected: **{selected_dist_name}** with parameters {final_params}")
    st.info("You can now proceed to the next step of your analysis.")

    with st.expander("View Saved Distribution Details"):
        saved_details = {
            "Distribution": selected_dist_name,
            "Parameters": [float(p) for p in final_params],
            "KS Statistic": float(selected_row["KS Statistic"]),
            "Theoretical Mean": float(theo_mean),
            "Theoretical Std Dev": float(theo_std),
        }
        if limited_calibration:
            saved_details["Limited Calibration"] = limited_calibration
        st.json(saved_details)
