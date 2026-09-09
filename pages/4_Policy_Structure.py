import streamlit as st
import numpy as np
import pandas as pd

st.title("Step 5: Policy Structure & Coverage Terms")

# Check that all required data exists
if "complete_claims_simulation" not in st.session_state:
    st.warning("Please complete claims simulation on Step 4 first.")
    st.stop()

# Get the simulation data
sim_data = st.session_state["complete_claims_simulation"]

st.subheader("Policy Structure Configuration")

col1, col2 = st.columns(2)

with col1:
    num_deductibles = st.number_input(
        "Number of Deductible Options",
        min_value=1,
        max_value=10,
        value=st.session_state.get("num_deductibles", 3),
        step=1,
        help="Set how many different deductible scenarios to analyze"
    )
    # Store in session state
    st.session_state["num_deductibles"] = num_deductibles

with col2:
    num_limits = st.number_input(
        "Number of Limit Options",
        min_value=1,
        max_value=10,
        value=st.session_state.get("num_limits", 3),
        step=1,
        help="Set how many different limit scenarios to analyze"
    )
    # Store in session state
    st.session_state["num_limits"] = num_limits

# Generate deductible input boxes
st.subheader("Deductible Values")
deductibles = []

# Initialize deductibles in session state if not exists
if "deductibles" not in st.session_state:
    st.session_state["deductibles"] = [10000 * (i+1) for i in range(num_deductibles)]

# Adjust list size if number of deductibles changed
while len(st.session_state["deductibles"]) < num_deductibles:
    st.session_state["deductibles"].append(10000 * len(st.session_state["deductibles"]))
if len(st.session_state["deductibles"]) > num_deductibles:
    st.session_state["deductibles"] = st.session_state["deductibles"][:num_deductibles]

cols = st.columns(min(num_deductibles, 3))  # Max 3 columns for layout
for i in range(num_deductibles):
    col_idx = i % 3
    with cols[col_idx]:
        deductible = st.number_input(
            f"Deductible {i+1}",
            min_value=0,
            max_value=10000000,
            value=st.session_state["deductibles"][i],
            step=1000,
            help=f"Enter deductible amount #{i+1}",
            key=f"ded_{i}"
        )
        deductibles.append(deductible)
        st.session_state["deductibles"][i] = deductible

# Generate limit input boxes
st.subheader("Limit Values")
limits = []

# Initialize limits in session state if not exists
if "limits" not in st.session_state:
    st.session_state["limits"] = [1000000 * (i+1) for i in range(num_limits)]

# Adjust list size if number of limits changed
while len(st.session_state["limits"]) < num_limits:
    st.session_state["limits"].append(1000000 * len(st.session_state["limits"]))
if len(st.session_state["limits"]) > num_limits:
    st.session_state["limits"] = st.session_state["limits"][:num_limits]

cols = st.columns(min(num_limits, 3))  # Max 3 columns for layout
for i in range(num_limits):
    col_idx = i % 3
    with cols[col_idx]:
        limit = st.number_input(
            f"Limit {i+1}",
            min_value=0,
            max_value=100000000,
            value=st.session_state["limits"][i],
            step=100000,
            help=f"Enter limit amount #{i+1}",
            key=f"lim_{i}"
        )
        limits.append(limit)
        st.session_state["limits"][i] = limit

# Display current selections
st.subheader("Current Configuration")

col1, col2 = st.columns(2)

with col1:
    st.write("**Selected Deductibles:**")
    for i, ded in enumerate(deductibles):
        st.write(f"• Deductible {i+1}: ${ded:,}")

with col2:
    st.write("**Selected Limits:**")
    for i, lim in enumerate(limits):
        st.write(f"• Limit {i+1}: ${lim:,}")

# Apply deductibles and limits to simulation data
if st.button("🔄 Apply Policy Terms to Simulation Data", type="primary"):
    
    # Check if we have claim-level details
    years = sim_data['years']
    sample_year_data = sim_data['results'][years[0]]
    
    if 'claim_details' not in sample_year_data:
        st.error("⚠️ Claim-level details not available. Please re-run the claims simulation in Step 4 to generate individual claim data.")
        st.stop()
    
    with st.spinner("Applying policy terms to simulation data..."):
        
        policy_results = {}
        
        # Process each deductible/limit combination
        for ded_idx, deductible in enumerate(deductibles):
            for lim_idx, limit in enumerate(limits):
                
                combo_key = f"Ded_{ded_idx+1}_Lim_{lim_idx+1}"
                combo_results = {}
                
                # Process each year
                for year in years:
                    year_data = sim_data['results'][year]
                    claim_details = year_data['claim_details']
                    
                    year_results = []
                    
                    # Process each simulation
                    for sim_idx, claims in enumerate(claim_details):
                        
                        if len(claims) == 0:
                            # No claims in this simulation
                            sim_result = {
                                'ground_up_loss': 0,
                                'losses_under_ded': 0,
                                'losses_under_limit': 0,
                                'covered_loss': 0,
                                'losses_over_limit': 0
                            }
                        else:
                            # Apply policy terms to each claim
                            ground_up_claims = np.array(claims)
                            
                            # Calculate components for each claim
                            losses_under_ded = np.minimum(ground_up_claims, deductible)
                            losses_under_limit = np.minimum(ground_up_claims, limit)
                            covered_per_claim = losses_under_limit - losses_under_ded
                            losses_over_limit = np.maximum(ground_up_claims - limit, 0)
                            
                            # Sum across all claims in the simulation
                            sim_result = {
                                'ground_up_loss': np.sum(ground_up_claims),
                                'losses_under_ded': np.sum(losses_under_ded),
                                'losses_under_limit': np.sum(losses_under_limit),
                                'covered_loss': np.sum(covered_per_claim),
                                'losses_over_limit': np.sum(losses_over_limit)
                            }
                        
                        year_results.append(sim_result)
                    
                    combo_results[year] = year_results
                
                policy_results[combo_key] = {
                    'deductible': deductible,
                    'limit': limit,
                    'results': combo_results
                }
        
        # Store results in session state
        st.session_state["policy_results"] = policy_results
        
        st.success("✅ Policy terms applied successfully!")

# Display results if available
if "policy_results" in st.session_state:
    policy_data = st.session_state["policy_results"]
    years = sim_data['years']  # Define years variable here
    
    st.subheader("Policy Analysis Results")
    
    # Show summary of combinations
    st.info(f"📊 Analysis complete for {len(deductibles)} deductible(s) × {len(limits)} limit(s) = {len(policy_data)} combinations")
    
    # Select combination to view
    combo_options = []
    for combo_key, combo_data in policy_data.items():
        ded_val = combo_data['deductible']
        lim_val = combo_data['limit']
        combo_options.append(f"{combo_key}: ${ded_val:,} ded / ${lim_val:,} limit")
    
    selected_combo = st.selectbox(
        "Select Deductible/Limit Combination to View",
        options=combo_options,
        help="Choose which combination to analyze in detail"
    )
    
    if selected_combo:
        combo_key = selected_combo.split(":")[0]
        combo_data = policy_data[combo_key]
        
        st.write(f"**Analysis for {selected_combo}**")
        
        # Export Options
        st.subheader("Export Policy Analysis Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Aggregated Policy Data**")
            
            # Single combination export
            if st.button(f"📊 Export Single Combo - {combo_key}", type="secondary"):
                with st.spinner("Preparing single combination export..."):
                    # Create export data for aggregated totals
                    export_data = []
                    
                    for year in years:
                        year_results = combo_data['results'][year]
                        
                        for sim_idx, result in enumerate(year_results):
                            export_data.append({
                                'Projected_Year': int(year),
                                'Simulation': sim_idx,
                                'Deductible': combo_data['deductible'],
                                'Limit': combo_data['limit'],
                                'Ground_Up_Loss': f"{result['ground_up_loss']:.0f}",
                                'Losses_Under_Ded': f"{result['losses_under_ded']:.0f}",
                                'Losses_Under_Limit': f"{result['losses_under_limit']:.0f}",
                                'Covered_Loss': f"{result['covered_loss']:.0f}",
                                'Losses_Over_Limit': f"{result['losses_over_limit']:.0f}"
                            })
                    
                    # Convert to DataFrame and create CSV
                    export_df = pd.DataFrame(export_data)
                    csv_data = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Single Combination CSV",
                        data=csv_data,
                        file_name=f"policy_aggregated_{combo_key}.csv",
                        mime="text/csv",
                        help=f"Contains {len(export_data):,} rows for {combo_key}"
                    )
                    
                    # Show preview
                    st.write("**Preview of Single Combination Export:**")
                    preview_df = export_df.head(10)
                    st.dataframe(preview_df, use_container_width=True)
                    
                    st.info(f"📊 Export contains {len(export_data):,} rows for selected combination")
            
            # All combinations export
            if st.button("📊 Export All Combinations", type="primary"):
                with st.spinner("Preparing all combinations export..."):
                    # Create export data for all combinations
                    all_export_data = []
                    
                    for combo_key_all, combo_data_all in policy_data.items():
                        for year in years:
                            year_results = combo_data_all['results'][year]
                            
                            for sim_idx, result in enumerate(year_results):
                                all_export_data.append({
                                    'Combination': combo_key_all,
                                    'Projected_Year': int(year),
                                    'Simulation': sim_idx,
                                    'Deductible': combo_data_all['deductible'],
                                    'Limit': combo_data_all['limit'],
                                    'Ground_Up_Loss': f"{result['ground_up_loss']:.0f}",
                                    'Losses_Under_Ded': f"{result['losses_under_ded']:.0f}",
                                    'Losses_Under_Limit': f"{result['losses_under_limit']:.0f}",
                                    'Covered_Loss': f"{result['covered_loss']:.0f}",
                                    'Losses_Over_Limit': f"{result['losses_over_limit']:.0f}"
                                })
                    
                    # Convert to DataFrame and create CSV
                    all_export_df = pd.DataFrame(all_export_data)
                    csv_data = all_export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download All Combinations CSV",
                        data=csv_data,
                        file_name="policy_all_combinations_aggregated.csv",
                        mime="text/csv",
                        help=f"Contains {len(all_export_data):,} rows for all {len(policy_data)} combinations"
                    )
                    
                    # Show preview
                    st.write("**Preview of All Combinations Export:**")
                    preview_df = all_export_df.head(10)
                    st.dataframe(preview_df, use_container_width=True)
                    
                    st.info(f"📊 Export contains {len(all_export_data):,} rows for all {len(policy_data)} combinations")
        
        with col2:
            st.write("**Claim Level Policy Details**")
            # Check if original claim details are available
            sample_year_data = sim_data['results'][years[0]]
            if 'claim_details' not in sample_year_data:
                st.warning("⚠️ Claim level details not available. Please re-run the simulation in Step 4 to generate claim-level data.")
            else:
                if st.button(f"📋 Export Claim Level Data - {combo_key}", type="secondary"):
                    with st.spinner("Preparing claim level policy export..."):
                        # Find maximum number of claims across all simulations and years
                        max_claims = 0
                        for year in years:
                            claim_details = sim_data['results'][year]['claim_details']
                            for claims in claim_details:
                                max_claims = max(max_claims, len(claims))
                        
                        # Create export data with policy terms applied to each claim
                        export_data = []
                        
                        for year in years:
                            claim_details = sim_data['results'][year]['claim_details']
                            
                            for sim_idx, claims in enumerate(claim_details):
                                # Create base row for this simulation
                                row = {
                                    'Projected_Year': int(year),
                                    'Simulation': sim_idx,
                                    'Deductible': combo_data['deductible'],
                                    'Limit': combo_data['limit']
                                }
                                
                                # Add individual claims with policy terms applied
                                for claim_idx, claim_value in enumerate(claims):
                                    # Apply policy terms to individual claim
                                    ground_up = claim_value
                                    under_ded = min(ground_up, combo_data['deductible'])
                                    under_limit = min(ground_up, combo_data['limit'])
                                    covered = under_limit - under_ded
                                    over_limit = max(ground_up - combo_data['limit'], 0)
                                    
                                    # Add columns for this claim
                                    base_col = f'Claim_{claim_idx + 1}'
                                    row[f'{base_col}_Ground_Up'] = f"{ground_up:.0f}"
                                    row[f'{base_col}_Under_Ded'] = f"{under_ded:.0f}"
                                    row[f'{base_col}_Under_Limit'] = f"{under_limit:.0f}"
                                    row[f'{base_col}_Covered'] = f"{covered:.0f}"
                                    row[f'{base_col}_Over_Limit'] = f"{over_limit:.0f}"
                                
                                # Fill remaining claim columns with empty values
                                for claim_idx in range(len(claims), max_claims):
                                    base_col = f'Claim_{claim_idx + 1}'
                                    row[f'{base_col}_Ground_Up'] = ""
                                    row[f'{base_col}_Under_Ded'] = ""
                                    row[f'{base_col}_Under_Limit'] = ""
                                    row[f'{base_col}_Covered'] = ""
                                    row[f'{base_col}_Over_Limit'] = ""
                                
                                export_data.append(row)
                        
                        # Convert to DataFrame and create CSV
                        export_df = pd.DataFrame(export_data)
                        csv_data = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Claim Level Details CSV",
                            data=csv_data,
                            file_name=f"policy_claim_level_{combo_key}.csv",
                            mime="text/csv",
                            help=f"Contains {len(export_data):,} rows with claim-level policy analysis"
                        )
                        
                        # Show preview of the data structure
                        st.write("**Preview of Claim Level Policy Export:**")
                        preview_df = export_df.head(5)  # Show first 5 rows (less due to many columns)
                        st.dataframe(preview_df, use_container_width=True)
                        
                        st.info(f"📊 Export contains {len(export_data):,} rows with up to {max_claims} claims per simulation, each with 5 policy components")