import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go

from utils import calculate_limited_mean

st.title("Step 4: Complete Claims Simulation (Frequency × Severity)")

if "exposure_data" not in st.session_state:
    st.warning("Please upload exposure data on the home page first.")
    st.stop()
if "final_distribution" not in st.session_state:
    st.warning("Please complete severity distribution fitting on Step 2 first.")
    st.stop()
if "claim_count_simulations" not in st.session_state:
    st.warning("Please complete claim count modeling on Step 3 first.")
    st.stop()

final_severity = st.session_state["final_distribution"]
frequency_sims = st.session_state["claim_count_simulations"]

st.subheader("Model Components Summary")
col1, col2 = st.columns(2)
with col1:
    st.write("**Severity Distribution:**")
    st.success(f"✅ {final_severity['name']}")
    st.write(f"Parameters: {final_severity['parameters']}")
    st.write(f"KS Statistic: {final_severity['ks_statistic']:.4f}")
with col2:
    st.write("**Frequency Distribution:**")
    st.success("✅ Poisson")
    freq_confidence = frequency_sims["confidence_level"]
    st.write(f"Simulations: {frequency_sims['num_simulations']:,}")
    st.write(f"Confidence Level: {freq_confidence * 100:.0f}%")

years = np.array(frequency_sims["years"])
projected_counts = np.array(frequency_sims["projected_counts"], dtype=float)
severity_dist = final_severity["distribution"]
severity_params = tuple(final_severity["parameters"])
limited_calibration = final_severity.get("limited_calibration")

unlimited_severity_mean = float(severity_dist.mean(*severity_params))
calibration_limit = None
limited_severity_mean = None

if limited_calibration:
    calibration_limit = float(limited_calibration["limit"])
    limited_severity_mean = calculate_limited_mean(
        severity_dist, severity_params, calibration_limit
    )

st.subheader("Complete Claims Simulation Parameters")
available_frequency_sims = int(frequency_sims["num_simulations"])

col1, col2 = st.columns(2)
with col1:
    num_complete_sims = st.number_input(
        "Number of Complete Simulations",
        min_value=100,
        max_value=available_frequency_sims,
        value=min(10000, available_frequency_sims),
        step=100,
        help="Number of complete simulations (frequency × severity)",
    )
with col2:
    random_seed = st.number_input(
        "Random Seed (Optional)",
        min_value=0,
        max_value=999999,
        value=42,
        help="Set seed for reproducible results (0 = random)",
    )

if st.button("🚀 Run Complete Claims Simulation", type="primary"):
    if random_seed > 0:
        np.random.seed(random_seed)

    with st.spinner("Running complete claims simulation..."):
        complete_results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for year_idx, year in enumerate(years):
            status_text.text(f"Simulating claims for year {int(year)}...")
            progress_bar.progress((year_idx + 1) / len(years))

            freq_sims = frequency_sims["yearly_simulations"][year][:num_complete_sims]

            year_results = []
            limited_year_results = []
            claim_details = []

            for num_claims in freq_sims:
                actual_claims = int(num_claims)

                if actual_claims == 0:
                    year_results.append(0.0)
                    if calibration_limit is not None:
                        limited_year_results.append(0.0)
                    claim_details.append([])
                    continue

                # No artificial cap on the number of claims.
                claim_severities = severity_dist.rvs(
                    *severity_params,
                    size=actual_claims,
                )

                claim_severities = np.maximum(claim_severities, 0.0)
                claim_details.append(claim_severities.tolist())
                year_results.append(float(np.sum(claim_severities)))

                if calibration_limit is not None:
                    limited_claims = np.minimum(claim_severities, calibration_limit)
                    limited_year_results.append(float(np.sum(limited_claims)))

            complete_results[year] = {
                "total_losses": np.array(year_results),
                "claim_details": claim_details,
                "limited_total_losses": (
                    np.array(limited_year_results)
                    if calibration_limit is not None
                    else None
                ),
            }

        progress_bar.empty()
        status_text.empty()

        st.session_state["complete_claims_simulation"] = {
            "results": complete_results,
            "num_simulations": int(num_complete_sims),
            "confidence_level": freq_confidence,
            "years": years.tolist(),
            "severity_info": {
                "name": final_severity["name"],
                "parameters": final_severity["parameters"],
                "unlimited_mean": unlimited_severity_mean,
                "limited_calibration": limited_calibration,
            },
        }

        st.success("✅ Complete claims simulation completed!")

if "complete_claims_simulation" in st.session_state:
    sim_data = st.session_state["complete_claims_simulation"]
    st.info(f"💾 Saved: {sim_data['num_simulations']:,} complete simulations")

    st.subheader("Complete Simulation Results")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Complete Simulation Results"):
            del st.session_state["complete_claims_simulation"]
            st.rerun()

    confidence_level = sim_data["confidence_level"]
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    st.subheader("Detailed Statistics by Year")
    stats_data = {
        "Metric": [
            "Mean",
            "Median",
            "Std Dev",
            "Skewness",
            "Kurtosis",
            f"P{lower_percentile:.0f}",
            f"P{upper_percentile:.0f}",
            f"VaR ({confidence_level * 100:.0f}%)",
            f"TVaR ({confidence_level * 100:.0f}%)",
            "Min",
            "Max",
            "Zero Loss %",
        ]
    }

    for year in years:
        total_losses = sim_data["results"][year]["total_losses"]
        var_threshold = np.percentile(total_losses, upper_percentile)
        tail = total_losses[total_losses >= var_threshold]
        tvar = np.mean(tail) if len(tail) else np.nan

        stats_data[f"Year {int(year)}"] = [
            f"${np.mean(total_losses):,.0f}",
            f"${np.median(total_losses):,.0f}",
            f"${np.std(total_losses):,.0f}",
            f"{stats.skew(total_losses):.3f}",
            f"{stats.kurtosis(total_losses):.3f}",
            f"${np.percentile(total_losses, lower_percentile):,.0f}",
            f"${np.percentile(total_losses, upper_percentile):,.0f}",
            f"${np.percentile(total_losses, upper_percentile):,.0f}",
            f"${tvar:,.0f}",
            f"${np.min(total_losses):,.0f}",
            f"${np.max(total_losses):,.0f}",
            f"{np.mean(total_losses == 0):.1%}",
        ]

    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    # Actuarial reconciliation: theoretical compound mean vs Monte Carlo mean.
    st.subheader("Actuarial Reconciliation")
    reconciliation_rows = []

    for i, year in enumerate(years):
        expected_claims = projected_counts[i]
        simulated_unlimited_mean = float(
            np.mean(sim_data["results"][year]["total_losses"])
        )
        expected_unlimited_total = expected_claims * unlimited_severity_mean

        row = {
            "Year": int(year),
            "Expected Claims": expected_claims,
            "Unlimited Severity Mean": unlimited_severity_mean,
            "Expected Unlimited Total": expected_unlimited_total,
            "Simulated Unlimited Mean": simulated_unlimited_mean,
            "Unlimited Difference %": (
                simulated_unlimited_mean / expected_unlimited_total - 1
                if expected_unlimited_total > 0
                else np.nan
            ),
        }

        if calibration_limit is not None:
            limited_results = sim_data["results"][year]["limited_total_losses"]
            simulated_limited_mean = float(np.mean(limited_results))
            expected_limited_total = expected_claims * limited_severity_mean

            row.update(
                {
                    "Claim Limit": calibration_limit,
                    "Limited Severity Mean": limited_severity_mean,
                    "Expected Limited Total": expected_limited_total,
                    "Simulated Limited Mean": simulated_limited_mean,
                    "Limited Difference %": (
                        simulated_limited_mean / expected_limited_total - 1
                        if expected_limited_total > 0
                        else np.nan
                    ),
                }
            )

        reconciliation_rows.append(row)

    reconciliation_df = pd.DataFrame(reconciliation_rows)
    format_map = {
        "Expected Claims": "{:,.2f}",
        "Unlimited Severity Mean": "${:,.0f}",
        "Expected Unlimited Total": "${:,.0f}",
        "Simulated Unlimited Mean": "${:,.0f}",
        "Unlimited Difference %": "{:.2%}",
        "Claim Limit": "${:,.0f}",
        "Limited Severity Mean": "${:,.0f}",
        "Expected Limited Total": "${:,.0f}",
        "Simulated Limited Mean": "${:,.0f}",
        "Limited Difference %": "{:.2%}",
    }
    format_map = {k: v for k, v in format_map.items() if k in reconciliation_df.columns}
    st.dataframe(
        reconciliation_df.style.format(format_map),
        use_container_width=True,
    )
    st.caption(
        "The simulated means should converge toward Expected Claims × Expected Severity. "
        "Small differences are normal Monte Carlo variation."
    )

    st.subheader("Distribution of Total Losses by Year")
    fig = go.Figure()
    colors = ["blue", "green", "red", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    for i, year in enumerate(years):
        total_losses = sim_data["results"][year]["total_losses"]
        non_zero_losses = total_losses[total_losses > 0]
        if len(non_zero_losses) > 0:
            fig.add_trace(
                go.Histogram(
                    x=non_zero_losses,
                    nbinsx=30,
                    name=f"Year {int(year)}",
                    opacity=0.6,
                    marker_color=colors[i % len(colors)],
                    histnorm="probability density",
                )
            )

    fig.update_layout(
        title="Distribution of Total Losses - All Years",
        xaxis_title="Total Loss ($)",
        yaxis_title="Probability Density",
        height=500,
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show Log Scale", help="Useful for highly skewed loss distributions"):
        fig_log = go.Figure()
        for i, year in enumerate(years):
            total_losses = sim_data["results"][year]["total_losses"]
            log_losses = np.log10(total_losses + 1)
            fig_log.add_trace(
                go.Histogram(
                    x=log_losses,
                    nbinsx=30,
                    name=f"Year {int(year)}",
                    opacity=0.6,
                    marker_color=colors[i % len(colors)],
                    histnorm="probability density",
                )
            )

        fig_log.update_layout(
            title="Log-Scale Distribution of Total Losses - All Years",
            xaxis_title="Log10(Total Loss + 1)",
            yaxis_title="Probability Density",
            height=500,
            barmode="overlay",
        )
        st.plotly_chart(fig_log, use_container_width=True)

    st.subheader("Export Simulation Data")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Aggregated Claim Data**")
        export_data = []
        for year in years:
            year_data = sim_data["results"][year]
            total_losses = year_data["total_losses"]
            limited_losses = year_data.get("limited_total_losses")

            for sim_idx, total_loss in enumerate(total_losses):
                row = {
                    "Projected_Year": int(year),
                    "Simulation": sim_idx,
                    "Total_Claim_Amount": round(float(total_loss), 2),
                }
                if limited_losses is not None:
                    row[f"Total_Limited_at_{calibration_limit:.0f}"] = round(
                        float(limited_losses[sim_idx]), 2
                    )
                export_data.append(row)

        export_df = pd.DataFrame(export_data)
        st.download_button(
            label="📥 Download Aggregated Data CSV",
            data=export_df.to_csv(index=False),
            file_name="aggregated_claims_all_years.csv",
            mime="text/csv",
        )

    with col2:
        st.write("**Claim Level Details**")
        claim_export = []
        for year in years:
            claim_details = sim_data["results"][year]["claim_details"]
            for sim_idx, claims in enumerate(claim_details):
                for claim_idx, claim_value in enumerate(claims, start=1):
                    claim_export.append(
                        {
                            "Projected_Year": int(year),
                            "Simulation": sim_idx,
                            "Claim_Number": claim_idx,
                            "Claim_Amount": round(float(claim_value), 2),
                        }
                    )

        claim_export_df = pd.DataFrame(claim_export)
        st.download_button(
            label="📥 Download Claim Level Data CSV",
            data=claim_export_df.to_csv(index=False),
            file_name="claim_level_details_all_years.csv",
            mime="text/csv",
        )
