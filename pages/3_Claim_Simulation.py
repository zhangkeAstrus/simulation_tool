import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Step 4: Complete Claims Simulation (Frequency × Severity)")

# Check that all required data exists
if "exposure_data" not in st.session_state:
    st.warning("Please upload exposure data on the home page first.")
    st.stop()

if "final_distribution" not in st.session_state:
    st.warning("Please complete severity distribution fitting on Step 2 first.")
    st.stop()

if "claim_count_simulations" not in st.session_state:
    st.warning("Please complete claim count modeling on Step 3 first.")
    st.stop()

# Get the data
exposure_data = st.session_state["exposure_data"]
final_severity = st.session_state["final_distribution"]
frequency_sims = st.session_state["claim_count_simulations"]

# Display current model components
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
    freq_confidence = frequency_sims['confidence_level']
    st.write(f"Simulations: {frequency_sims['num_simulations']:,}")
    st.write(f"Confidence Level: {freq_confidence*100:.0f}%")

# Extract necessary data
years = np.array(frequency_sims['years'])
severity_dist = final_severity['distribution']
severity_params = final_severity['parameters']

# Complete Simulation Parameters
st.subheader("Complete Claims Simulation Parameters")

col1, col2 = st.columns(2)

with col1:
    num_complete_sims = st.number_input(
        "Number of Complete Simulations",
        min_value=100,
        max_value=50000,
        value=10000,
        step=100,
        help="Number of complete simulations (frequency × severity)"
    )

with col2:
    random_seed = st.number_input(
        "Random Seed (Optional)",
        min_value=0,
        max_value=999999,
        value=42,
        help="Set seed for reproducible results (0 = random)"
    )

# Run Complete Simulation
if st.button("🚀 Run Complete Claims Simulation", type="primary"):
    
    # Set random seed if specified
    if random_seed > 0:
        np.random.seed(random_seed)
    
    with st.spinner("Running complete claims simulation..."):
        
        complete_results = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use default claim cap for performance
        max_claims_per_sim = 1000
        
        for year_idx, year in enumerate(years):
            status_text.text(f"Simulating claims for year {int(year)}...")
            progress_bar.progress((year_idx + 1) / len(years))
            
            # Get frequency simulations for this year
            freq_sims = frequency_sims['yearly_simulations'][year][:num_complete_sims]
            
            year_results = []
            claim_details = []  # Store individual claims for export
            
            for sim_idx, num_claims in enumerate(freq_sims):
                if num_claims == 0:
                    # No claims in this simulation
                    year_results.append(0)
                    claim_details.append([])  # Empty list for no claims
                else:
                    # Cap the number of claims for performance
                    actual_claims = min(int(num_claims), max_claims_per_sim)
                    
                    # Generate severity for each claim
                    claim_severities = severity_dist.rvs(*severity_params, size=actual_claims)
                    
                    # Store individual claim severities for this simulation
                    claim_details.append(claim_severities.tolist())
                    
                    # Calculate total loss for this simulation
                    total_loss = np.sum(claim_severities)
                    year_results.append(total_loss)
            
            complete_results[year] = {
                'total_losses': np.array(year_results),
                'claim_details': claim_details
            }
        
        progress_bar.empty()
        status_text.empty()
        
        # Store complete simulation results with frequency confidence level
        st.session_state["complete_claims_simulation"] = {
            "results": complete_results,
            "num_simulations": num_complete_sims,
            "confidence_level": freq_confidence,  # Use confidence level from frequency modeling
            "years": years.tolist(),
            "severity_info": {
                "name": final_severity['name'],
                "parameters": final_severity['parameters']
            },
            "max_claims_cap": max_claims_per_sim
        }
        
        st.success("✅ Complete claims simulation completed!")

# Display results if simulation exists
if "complete_claims_simulation" in st.session_state:
    sim_data = st.session_state["complete_claims_simulation"]
    
    st.info(f"💾 Saved: {sim_data['num_simulations']:,} complete simulations")
    
    # Results display
    st.subheader("Complete Simulation Results")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🗑️ Clear Complete Simulation Results"):
            del st.session_state["complete_claims_simulation"]
            st.rerun()
    
    confidence_level = sim_data['confidence_level']
    
    # Calculate statistics for all years
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    # Detail Statistics Table for All Years - Years as Columns
    st.subheader("Detailed Statistics by Year")
    
    # Create statistics with years as columns
    stats_data = {
        "Metric": ["Mean", "Median", "Std Dev", "Skewness", "Kurtosis",
                  f"P{lower_percentile:.0f}", f"P{upper_percentile:.0f}", 
                  f"VaR ({confidence_level*100:.0f}%)", f"TVaR ({confidence_level*100:.0f}%)",
                  "Min", "Max", "Zero Loss %"]
    }
    
    # Add each year as a column
    for year in years:
        year_data = sim_data['results'][year]
        total_losses = year_data['total_losses']
        
        # Calculate TVaR (Expected Shortfall)
        var_threshold = np.percentile(total_losses, upper_percentile)
        tvar = np.mean(total_losses[total_losses >= var_threshold])
        
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
            f"{np.mean(total_losses == 0):.1%}"
        ]
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)
    
    # Distribution Visualization for All Years - Single Plot
    st.subheader("Distribution of Total Losses by Year")
    
    # Create single plot with all years overlaid
    fig = go.Figure()
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, year in enumerate(years):
        year_data = sim_data['results'][year]
        total_losses = year_data['total_losses']
        
        # Filter out zero losses for better visualization
        non_zero_losses = total_losses[total_losses > 0]
        
        if len(non_zero_losses) > 0:
            fig.add_trace(go.Histogram(
                x=non_zero_losses,
                nbinsx=30,
                name=f"Year {int(year)}",
                opacity=0.6,
                marker_color=colors[i % len(colors)],
                histnorm='probability density'  # Normalize to show probability density
            ))
    
    fig.update_layout(
        title="Distribution of Total Losses - All Years",
        xaxis_title="Total Loss ($)",
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Log scale option for highly skewed distributions
    if st.checkbox("Show Log Scale", help="Useful for highly skewed loss distributions"):
        fig_log = go.Figure()
        
        for i, year in enumerate(years):
            year_data = sim_data['results'][year]
            total_losses = year_data['total_losses']
            non_zero_losses = total_losses[total_losses > 0]
            
            if len(non_zero_losses) > 0:
                fig_log.add_trace(go.Histogram(
                    x=np.log10(non_zero_losses + 1),
                    nbinsx=30,
                    name=f"Year {int(year)} - Log Scale",
                    opacity=0.6,
                    marker_color=colors[i % len(colors)],
                    histnorm='probability density'
                ))
        
        fig_log.update_layout(
            title="Log-Scale Distribution of Total Losses - All Years",
            xaxis_title="Log10(Total Loss + 1)",
            yaxis_title="Probability Density",
            height=500,
            barmode='overlay',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_log, use_container_width=True)
    
    # Export Options
    st.subheader("Export Simulation Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aggregated Claim Data**")
        if st.button("📊 Export Aggregated Data", type="secondary"):
            with st.spinner("Preparing aggregated export..."):
                # Create export data for aggregated totals
                export_data = []
                
                for year in years:
                    year_data = sim_data['results'][year]
                    total_losses = year_data['total_losses']
                    
                    for sim_idx, total_loss in enumerate(total_losses):
                        export_data.append({
                            'Projected_Year': int(year),
                            'Simulation': sim_idx,
                            'Total_Claim_Amount': f"{total_loss:.0f}"
                        })
                
                # Convert to DataFrame and create CSV
                export_df = pd.DataFrame(export_data)
                csv_data = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Aggregated Data CSV",
                    data=csv_data,
                    file_name=f"aggregated_claims_all_years.csv",
                    mime="text/csv",
                    help=f"Contains {len(export_data):,} rows with total claim amounts per simulation"
                )
                
                # Show preview
                st.write("**Preview of Aggregated Export:**")
                preview_df = export_df.head(10)
                st.dataframe(preview_df, use_container_width=True)
                
                st.info(f"📊 Export contains {len(export_data):,} rows with total claim amounts")
    
    with col2:
        st.write("**Claim Level Details**")
        # Check if claim details are available
        sample_year_data = sim_data['results'][years[0]]
        if 'claim_details' not in sample_year_data:
            st.warning("⚠️ Claim level details not available. Please re-run the simulation to generate claim-level data.")
            st.info("The current simulation results don't include individual claim details. Click 'Run Complete Claims Simulation' again to generate exportable claim-level data.")
        else:
            if st.button("📋 Export Claim Level Data", type="secondary"):
                with st.spinner("Preparing claim level export..."):
                    # Find maximum number of claims across all simulations and years
                    max_claims = 0
                    for year in years:
                        year_data = sim_data['results'][year]
                        claim_details = year_data['claim_details']
                        for claims in claim_details:
                            max_claims = max(max_claims, len(claims))
                    
                    # Create export data
                    export_data = []
                    
                    for year in years:
                        year_data = sim_data['results'][year]
                        claim_details = year_data['claim_details']
                        
                        for sim_idx, claims in enumerate(claim_details):
                            # Create row for this simulation
                            row = {
                                'Projected_Year': int(year),
                                'Simulation': sim_idx
                            }
                            
                            # Add individual claims as separate columns
                            for claim_idx, claim_value in enumerate(claims):
                                row[f'Claim_{claim_idx + 1}'] = f"{claim_value:.0f}"
                            
                            # Fill remaining claim columns with empty values if this simulation has fewer claims
                            for claim_idx in range(len(claims), max_claims):
                                row[f'Claim_{claim_idx + 1}'] = ""
                            
                            export_data.append(row)
                    
                    # Convert to DataFrame and create CSV
                    export_df = pd.DataFrame(export_data)
                    csv_data = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Claim Level Details CSV",
                        data=csv_data,
                        file_name=f"claim_level_details_all_years.csv",
                        mime="text/csv",
                        help=f"Contains {len(export_data):,} rows with up to {max_claims} claims per simulation"
                    )
                    
                    # Show preview of the data structure
                    st.write("**Preview of Claim Level Export:**")
                    preview_df = export_df.head(10)  # Show first 10 rows
                    st.dataframe(preview_df, use_container_width=True)
                    
                    st.info(f"📊 Export will contain {len(export_data):,} rows (simulations) with up to {max_claims} claim columns")