import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
st.set_page_config(
    page_title="TraitMap: Trait-Based Ancestry Explorer", layout="wide")


# --- Distinctive Title Section (Left-Aligned, Accent Bar) ---
st.markdown(
    """
        <style>
            :root{
                --tm-primary: #0f172a;       /* strong title on light */
                --tm-muted: #475569;         /* muted caption */
                --tm-accent: #06b6d4;        /* accent (teal/cyan) */
                --tm-sub: #0f766e;           /* subtitle */
                --tm-card-bg: #f6fff9;       /* flat card background (light) */
                --tm-card-border: #d0e8dd;   /* subtle border */
                --tm-badge-bg: #e6f9ef;      /* flat badge background */
                --tm-badge-text: #1e7f5c;
                --tm-warning: #eab308;
                --tm-info: #0ea5e9;
                --tm-success: #16a34a;
            }
            @media (prefers-color-scheme: dark){
                :root{
                    --tm-primary: #e6fffb;
                    --tm-muted: #cbd5e1;
                    --tm-accent: #7dd3fc;
                    --tm-sub: #7ee7c6;
                    --tm-card-bg: #072726;       /* flat card background (dark) */
                    --tm-card-border: #083b37;
                    --tm-badge-bg: rgba(255,255,255,0.03);
                    --tm-badge-text: #bbf7d0;
                    --tm-warning: #f59e0b;
                    --tm-info: #38bdf8;
                    --tm-success: #4ade80;
                }
            }
        </style>
        <div style='display:flex; flex-direction:column; align-items:flex-start; margin-bottom:1.3em;'>
            <div>
                <div style='font-size:2.45rem; font-weight:900; color:var(--tm-primary); letter-spacing:-1px; margin-bottom:0.13em;'>TraitMap</div>
                <div style='font-size:1.13rem; color:var(--tm-sub); font-weight:600; margin-bottom:0.18em;'>Explore ancestry patterns using trait-associated SNPs.</div>
                <div style='font-size:1.01rem; color:var(--tm-muted); font-weight:400;'>Upload your genotype data or browse reference populations.</div>
            </div>
        </div>
        """,
    unsafe_allow_html=True
)


# --- Info Section: Warm Pastel Card, Pill Button, Modern Icons ---
st.markdown(
    """
        <div style='margin-bottom:1.5em; display:flex; justify-content:center;'>
                <div style='background:var(--tm-card-bg); border:1.5px solid var(--tm-card-border); border-radius:1.3em; box-shadow:0 2px 12px rgba(0,0,0,0.06); padding:2.1em 2.5em 2.1em 2.5em; min-width:340px; max-width:720px;'>
                    <div style='display:flex; align-items:center; margin-bottom:1.1em;'>
                        <span style='font-size:1.7em; margin-right:0.7em;'>🌐</span>
                        <span style='font-size:1.22em; font-weight:800; color:var(--tm-primary);'>Reference samples: <span style='color:var(--tm-primary); font-weight:600;'>1000 Genomes, SGDP, HGDP</span></span>
                        <a href='https://docs.google.com/spreadsheets/d/1086enwO19h-ruj61SxqLnvn-IXtMRbG50VOCuR-5xg0/edit?gid=1970300111#gid=1970300111' target='_blank' style='margin-left:1.1em; text-decoration:none;'>
                            <span style='background:var(--tm-badge-bg); color:var(--tm-badge-text); font-weight:700; border-radius:1.2em; padding:0.18em 1.1em; font-size:0.98em; border:1px solid rgba(0,0,0,0.04); box-shadow:0 1px 4px rgba(0,0,0,0.04);'>View Data ↗</span>
                        </a>
                    </div>
                    <div style='margin:0.7em 0 0.3em 0; font-size:1.09em; color:var(--tm-warning); font-weight:700;'>⚠️ <span style='color:var(--tm-primary); font-weight:400;'>Only 128 trait SNPs are analyzed. Not a full ancestry test.</span></div>
                    <div style='margin:0.3em 0 0.3em 0; font-size:1.09em; color:var(--tm-info); font-weight:700;'>🔬 <span style='color:var(--tm-primary); font-weight:400;'>Results are experimental and for education/entertainment only.</span></div>
                    <div style='margin:0.3em 0 0.1em 0; font-size:1.09em; color:var(--tm-success); font-weight:700;'>🔒 <span style='color:var(--tm-primary); font-weight:400;'>Data is processed in your browser and never stored.</span></div>
                </div>
            </div>
        """,
    unsafe_allow_html=True
)


# Load the genotype and admixture data


def encode_genotype_dosage(geno, ref_allele=None):
    """
    Encode genotype as allele dosage (number of alternate alleles):
    - If ref_allele is provided, count non-ref alleles (0, 1, 2)
    - If not, use the most common allele in the column as reference
    Returns np.nan for missing/invalid genotypes.
    """
    import numpy as np
    if pd.isnull(geno) or not isinstance(geno, str) or len(geno) != 2:
        return np.nan
    if ref_allele is None:
        ref_allele = geno[0]
    return sum(1 for a in geno if a != ref_allele)


# File uploader for user data
uploaded_file = st.file_uploader(
    'Upload your genotype file to plot with existing data', type=['txt', 'csv'])
st.caption(
    'ℹ️ Genotype upload is currently limited to 23andMe and AncestryDNA raw data formats.')

# Always load the reference (existing) data and get its SNP columns


def load_reference():
    df = pd.read_csv('merged_genotypes.csv', low_memory=False)
    df['__source__'] = 'existing'
    return df


# Admixture calculator selection
st.sidebar.title("Analysis Settings")
admixture_file = st.sidebar.selectbox(
    "Admixture calculator:",
    ["Globe13 (default)", "puntDNAL"],
    index=0,
    key="admixture_select"
)
if admixture_file == "puntDNAL":
    admixture_path = 'merged_puntDNAL.csv'
else:
    admixture_path = 'merged_globe13.csv'


def load_admixture():
    return pd.read_csv(admixture_path, low_memory=False)


df_ref = load_reference()
admix_data = load_admixture()


# Define ancestry components for filtering and coloring, based on admixture file
if admixture_file == "puntDNAL":
    ancestry_components = [
        'African HG', 'Amerinidian', 'Anatolian Neolithic', 'EHG-Steppe',
        'East Eurasian', 'Iran Neolithic', 'Natufian HG', 'Oceanian',
        'Siberian', 'South Eurasian', 'Sub-Saharan', 'Western HG'
    ]
else:
    ancestry_components = [
        'Amerindian', 'Artic', 'Australasian', 'East African', 'East Asian',
        'Mediterranean', 'North European', 'Palaeo African', 'Siberian',
        'South Asian', 'Southwest Asian', 'West African', 'West Asian'
    ]


# --- SNP Category Selection ---
snps_cat_path = 'snps_cat.csv'
if os.path.exists(snps_cat_path):
    snps_cat_df = pd.read_csv(snps_cat_path, low_memory=False)
    available_categories = sorted(snps_cat_df['Category'].unique())
    selected_category = st.sidebar.selectbox(
        'SNP Category:',
        options=['All'] + available_categories,
        index=0,
        key='snp_category_select'
    )
    if selected_category != 'All':
        snps_in_category = snps_cat_df[snps_cat_df['Category']
                                       == selected_category]['rsID'].tolist()
    else:
        snps_in_category = None
else:
    selected_category = 'All'
    snps_in_category = None


# --- Analysis Method Selection and Per-Method Options ---
dimred_method = st.sidebar.radio(
    "Analysis method:",
    ["PCA", "Population Distance"],
    index=0,
    key="viz_method_radio"
)

# PCA options grouped under its own expander
with st.sidebar.expander("PCA Options", expanded=(dimred_method == "PCA")):
    if dimred_method == "PCA":
        show_dominant_ancestry = st.checkbox(
            "Color by dominant ancestry", value=False, key="color_by_ancestry")
    else:
        show_dominant_ancestry = False

# Population Distance options grouped under its own expander
with st.sidebar.expander("Population Distance Options", expanded=(dimred_method == "Population Distance")):
    if dimred_method == "Population Distance":
        compare_individuals = st.checkbox(
            "Compare Individuals (not Groups)", value=False,
            help="Toggle to compare individual samples instead of population groups.",
            key="compare_individuals_checkbox")
    else:
        compare_individuals = False

st.sidebar.caption(
    "Tip: Use the controls above to explore and filter the data.")

# If user uploads, align their data to reference SNPs
if uploaded_file is not None:
    try:
        # Try reading as 23andMe/Ancestry raw data format first without header
        df_raw = pd.read_csv(uploaded_file, sep='\t',
                             comment='#', header=None, low_memory=False)

        # Check if it has 4 columns (typical 23andMe format: rsid, chr, pos, genotype)
        if df_raw.shape[1] == 4:
            df_raw.columns = ['rsid', 'chromosome', 'position', 'genotype']
            st.success("✅ Detected 23andMe format")
        elif df_raw.shape[1] == 5:
            df_raw.columns = ['rsid', 'chromosome',
                              'position', 'allele1', 'allele2']
            # Combine allele1 and allele2 into genotype for AncestryDNA format
            df_raw['genotype'] = df_raw['allele1'].astype(
                str) + df_raw['allele2'].astype(str)
            st.success("✅ Detected AncestryDNA format")
        else:
            # Try with header
            df_raw = pd.read_csv(uploaded_file, sep='\t',
                                 comment='#', low_memory=False)

            # Check if AncestryDNA format with allele1/allele2 columns
            if 'allele1' in df_raw.columns and 'allele2' in df_raw.columns:
                df_raw['genotype'] = df_raw['allele1'].astype(
                    str) + df_raw['allele2'].astype(str)
                st.success("✅ Detected AncestryDNA format")

        # Check if it looks like 23andMe/Ancestry format
        if 'rsid' in df_raw.columns and 'genotype' in df_raw.columns:
            # Convert raw data to match reference format
            meta_cols = ['source', 'group', 'group_full', 'individual']
            snp_cols = [
                col for col in df_ref.columns if col not in meta_cols and col != '__source__']

            # Create empty dataframe with same structure as reference
            df_up = pd.DataFrame(columns=meta_cols + snp_cols + ['__source__'])

            # Fill in one row with genotypes from raw data
            new_row = {}
            for col in meta_cols:
                new_row[col] = None  # Will be filled with defaults
            new_row['__source__'] = 'uploaded'

            # Extract genotypes for SNPs that exist in reference data
            snps_found = 0
            for snp in snp_cols:
                if snp in df_raw['rsid'].values:
                    genotype = df_raw[df_raw['rsid']
                                      == snp]['genotype'].iloc[0]
                    new_row[snp] = genotype
                    snps_found += 1
                else:
                    new_row[snp] = None  # Missing SNP

            st.info(
                f"📊 Found {snps_found} matching SNPs out of {len(snp_cols)} reference SNPs")
            df_up = pd.DataFrame([new_row])

        else:
            st.warning("⚠️ Format not recognized - trying as regular CSV")
            # Try as regular CSV format
            df_up = df_raw.copy()
            df_up['__source__'] = 'uploaded'

    except Exception as e:
        st.error(
            f"Error reading uploaded file: {str(e)}. Please upload 23andMe or Ancestry raw data file.")
        st.stop()

    # Set individual name and group if not present
    if 'individual' not in df_up.columns or df_up['individual'].isna().all():
        df_up['individual'] = 'YOU'
    if 'group' not in df_up.columns or df_up['group'].isna().all():
        df_up['group'] = 'YOU'

    # Combine with reference data
    df = pd.concat([df_ref, df_up], ignore_index=True)
else:
    df = df_ref.copy()


meta_cols = ['source', 'group', 'group_full', 'individual', '__source__']
all_geno_cols = [col for col in df.columns if col not in meta_cols]
# Filter geno_cols by selected SNP category if set
if snps_in_category:
    geno_cols = [col for col in all_geno_cols if col in snps_in_category]
else:
    geno_cols = all_geno_cols


# Always train PCA on reference data only
df_ref_only = df[df['__source__'] == 'existing'].copy()
# Re-filter geno_cols to ensure only selected SNPs are used for PCA
if snps_in_category:
    geno_cols = [col for col in df_ref_only.columns if col in snps_in_category]
else:
    geno_cols = [col for col in df_ref_only.columns if col not in [
        'source', 'group', 'group_full', 'individual', '__source__']]


# Vectorized genotype dosage encoding
ref_alleles = {}
for col in geno_cols:
    if not df_ref_only[col].isna().all():
        alleles = df_ref_only[col].dropna().astype(str).str.cat()
        if alleles:
            ref_allele = max(set(alleles), key=alleles.count)
            ref_alleles[col] = ref_allele
        else:
            ref_alleles[col] = None
    else:
        ref_alleles[col] = None


def encode_geno_vec(geno_col, ref_allele):
    # geno_col: pd.Series of genotypes (str)
    # ref_allele: str or None
    arr = geno_col.values
    # Only process non-null and string of length 2
    mask = pd.notnull(arr) & (pd.Series(arr).astype(str).str.len() == 2).values
    out = np.full(arr.shape, np.nan)
    if ref_allele is None:
        # Use first allele if available
        ref_allele = None
    for i, val in enumerate(arr):
        if mask[i]:
            ra = ref_allele if ref_allele is not None else str(val)[0]
            out[i] = sum(1 for a in str(val) if a != ra)
    return out


df_ref_encoded = df_ref_only.copy()
for col in geno_cols:
    df_ref_encoded[col] = encode_geno_vec(df_ref_only[col], ref_alleles[col])

# Impute and scale reference data
df_ref_pca = df_ref_encoded[geno_cols].copy()
# Remove columns that are all NaN
valid_cols = [col for col in geno_cols if not df_ref_pca[col].isna().all()]
df_ref_pca = df_ref_pca[valid_cols]

imputer = SimpleImputer(strategy='mean')
df_ref_imputed = pd.DataFrame(
    imputer.fit_transform(df_ref_pca), columns=valid_cols)

scaler = StandardScaler()
df_ref_scaled = pd.DataFrame(
    scaler.fit_transform(df_ref_imputed), columns=valid_cols)

# Show quick metrics about the dataset
ref_sample_count = df_ref_only.shape[0]
valid_snp_count = len(valid_cols)
col1, col2 = st.columns(2)
col1.metric("Reference Samples", f"{ref_sample_count}")
col2.metric("SNPs Used", f"{valid_snp_count}")


# --- Dimensionality reduction: PCA or Dendrogram (only one runs) ---
n_components = 2
ref_pca_result = None
if dimred_method == "PCA":
    pca = PCA(n_components=n_components)
    ref_pca_result = pca.fit_transform(df_ref_scaled)

# Create base dataframe for both methods
base_df = pd.DataFrame({
    'group': df_ref_only.apply(
        lambda row: row['group_full'] if pd.notna(row.get('group_full', None)) and str(
            row.get('group_full', '')).strip() != '' else row['group'],
        axis=1
    ),
    'individual': df_ref_only['individual'].values,
    'source': df_ref_only['__source__'].values
})

# Add PCA coordinates (only fill the one in use)
if ref_pca_result is not None:
    base_df['PC1'] = ref_pca_result[:, 0]
    base_df['PC2'] = ref_pca_result[:, 1]
else:
    base_df['PC1'] = np.nan
    base_df['PC2'] = np.nan

# Merge with admixture data
base_df = base_df.merge(
    admix_data[['individual'] + ancestry_components],
    on='individual',
    how='left'
)

# Calculate dominant ancestry for coloring


def get_dominant_ancestry(row):
    ancestry_values = {comp: row[comp]
                       for comp in ancestry_components if pd.notna(row[comp])}
    if ancestry_values:
        return max(ancestry_values.items(), key=lambda x: x[1])[0]
    return 'Unknown'


base_df['dominant_ancestry'] = base_df.apply(get_dominant_ancestry, axis=1)

# Format admixture data for hover tooltip (show all non-zero components)


def format_admixture_tooltip(row):
    ancestry_values = []
    for comp in ancestry_components:
        if pd.notna(row[comp]) and row[comp] > 0:
            ancestry_values.append((comp, row[comp]))
    ancestry_values.sort(key=lambda x: x[1], reverse=True)
    tooltip_parts = [f"{comp}: {pct:.1f}%" for comp, pct in ancestry_values]
    return "<br>".join(tooltip_parts)


base_df['admixture_tooltip'] = base_df.apply(format_admixture_tooltip, axis=1)

# If uploaded data exists, project it onto PCA
if uploaded_file is not None and dimred_method == "PCA":
    df_up_only = df[df['__source__'] == 'uploaded'].copy()
    df_up_encoded = df_up_only.copy()
    for col in valid_cols:
        if col in df_up_only.columns:
            df_up_encoded[col] = df_up_only[col].apply(
                lambda g: encode_genotype_dosage(g, ref_alleles.get(col)))
        else:
            df_up_encoded[col] = None
    df_up_pca = df_up_encoded[valid_cols].copy()
    try:
        df_up_imputed = pd.DataFrame(
            imputer.transform(df_up_pca), columns=valid_cols)
        df_up_scaled = pd.DataFrame(scaler.transform(
            df_up_imputed), columns=valid_cols)
    except Exception as e:
        st.error(f"Error processing uploaded sample: {str(e)}")
        st.stop()
    # Project uploaded sample onto PCA
    up_pca_result = pca.transform(df_up_scaled)
    up_df = pd.DataFrame({
        'group': df_up_only['group'].values,
        'individual': df_up_only['individual'].values,
        'source': df_up_only['__source__'].values,
        'PC1': up_pca_result[:, 0],
        'PC2': up_pca_result[:, 1],
        'dominant_ancestry': 'YOU',
        'admixture_tooltip': 'No admixture data'
    })
    plot_df = pd.concat([base_df, up_df], ignore_index=True)
else:
    plot_df = base_df.copy()

if len(plot_df) < 2:
    st.error('Not enough data for visualization.')
else:
    filtered_df = plot_df.copy()
    color_by = 'dominant_ancestry' if show_dominant_ancestry else 'group'
    symbol_by = 'dominant_ancestry' if show_dominant_ancestry else 'group'

    # Choose axes and title based on method
    if dimred_method == "PCA":
        x_col, y_col = 'PC1', 'PC2'
        plot_title = f'PCA: PC1 vs PC2 (Colored by {"Dominant Ancestry" if show_dominant_ancestry else "Population Group"})'
        hover_data = {
            'group': True,
            'admixture_tooltip': True,
            'PC1': ':.2f',
            'PC2': ':.2f'
        }
        fig = px.scatter(
            filtered_df, x=x_col, y=y_col,
            color=color_by,
            symbol=symbol_by,
            hover_name='individual',
            hover_data=hover_data,
            title=plot_title,
            labels={
                'PC1': 'Principal Component 1',
                'PC2': 'Principal Component 2',
                'group': 'Population Group',
                'individual': 'Individual ID',
                'dominant_ancestry': 'Dominant Ancestry',
                'admixture_tooltip': 'All Ancestry Components'
            },
            color_discrete_sequence=['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FF00FF',
                                     '#00FFFF', '#800000', '#000080', '#008000', '#808000',
                                     '#800080', '#008080', '#FFA500', '#A52A2A', '#DDA0DD',
                                     '#98FB98', '#F0E68C', '#DEB887', '#5F9EA0', '#7FFF00',
                                     '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#B8860B']
        )
        # Make uploaded sample stand out
        for trace in fig.data:
            if 'you' in str(trace.name).lower():
                trace.marker.size = 18
                trace.marker.symbol = 'star'
                trace.marker.color = 'gold'
            else:
                trace.marker.size = 8
        fig.update_layout(
            legend=dict(
                title=color_by.replace('_', ' ').title(),
                tracegroupgap=0,
            ),
            showlegend=True,
            height=600
        )

        marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up',
                          'triangle-down', 'pentagon', 'hexagon', 'star', 'triangle-left',
                          'triangle-right', 'star-triangle-up', 'star-square', 'asterisk']
        for i, trace in enumerate(fig.data):
            trace_name = str(trace.name)
            trace.marker.symbol = marker_symbols[i % len(marker_symbols)]
            if 'you' in trace_name.lower():
                trace.marker.size = 15
                trace.marker.symbol = 'star'
                trace.marker.color = 'gold'
            else:
                trace.marker.size = 8
        fig.update_layout(
            legend=dict(
                title=color_by.replace('_', ' ').title(),
                tracegroupgap=0,
            ),
            showlegend=True,
            height=600
        )
        # 'Reference Samples' metric already shows total samples; avoid redundant message
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, config={"responsive": True}, key="pca_plot")

    elif dimred_method == "Population Distance":
        import plotly.express as px
        from scipy.spatial.distance import cdist, mahalanobis
        st.header("Population Distance Explorer")

        # Ensure uploaded data is included for Population Distance
        if uploaded_file is not None:
            # Include uploaded data similar to PCA section
            df_up_only = df[df['__source__'] == 'uploaded'].copy()
            up_df = pd.DataFrame({
                'group': df_up_only['group'].values,
                'individual': df_up_only['individual'].values,
                'source': df_up_only['__source__'].values,
                'PC1': np.nan,  # Not used for population distance
                'PC2': np.nan,  # Not used for population distance
                'dominant_ancestry': 'YOU',
                'admixture_tooltip': 'No admixture data'
            })
            # Combine base_df with uploaded data
            pop_df_full = pd.concat([base_df, up_df], ignore_index=True)
        else:
            pop_df_full = base_df.copy()

        pop_df = pop_df_full.copy()
        pop_df_reset = pop_df.reset_index(drop=True)

        # --- Comparison Mode Selection ---
        if compare_individuals:
            # Individual mode: show individuals grouped by their population group
            groups_order = pop_df_reset['group'].fillna('Unknown').unique()
            ind_labels = []
            ind_map = {}
            for g in groups_order:
                members = pop_df_reset[pop_df_reset['group'].fillna(
                    'Unknown') == g]['individual'].dropna().unique().tolist()
                if not members:
                    continue
                for ind in members:
                    label = f"{g} — {ind}"
                    ind_labels.append(label)
                    ind_map[label] = ind

            # Default: select uploaded individual if present, else first
            default_idx = 0
            you_label = next(
                (lbl for lbl, iid in ind_map.items() if iid == 'YOU'), None)
            if uploaded_file is not None and you_label:
                default_idx = ind_labels.index(you_label)

            selected_label = st.selectbox(
                "Select an individual:",
                ind_labels,
                index=default_idx,
                key="popdist_individual_select"
            )
            selected_ind = ind_map[selected_label]
            selected_group = None
        else:
            # Group mode (original)
            # Count sample size for each group
            group_sizes = pop_df_reset.groupby('group').size().to_dict()
            pop_groups = pop_df_reset['group'].dropna().unique()
            # Create labels: show (n=) only for non-uploaded groups
            group_labels = []
            for g in pop_groups:
                if g == 'YOU':
                    group_labels.append(g)
                else:
                    group_labels.append(f"{g} (n={group_sizes.get(g, 0)})")
            group_map = dict(zip(group_labels, pop_groups))
            # Auto-select "YOU" if it exists, otherwise first group
            you_label = next(
                (label for label, g in group_map.items() if g == 'YOU'), None)
            if uploaded_file is not None and you_label:
                default_idx = group_labels.index(you_label)
            else:
                default_idx = 0
            selected_label = st.selectbox(
                "Select a population group:",
                group_labels,
                index=default_idx,
                key="popdist_group_select"
            )
            selected_group = group_map[selected_label]

        # Distance metric selection
        metric_options = {
            "Euclidean": "euclidean",
            "Manhattan (Cityblock)": "cityblock",
        }
        dist_metric_label = st.selectbox(
            "Distance metric:",
            list(metric_options.keys()),
            index=0,
            key="popdist_metric_select"
        )
        dist_metric = metric_options[dist_metric_label]

        # Prepare scaled data for all populations/individuals
        pop_encoded = pop_df_reset.copy()

        # Add genotype columns from the original data
        ref_mask = pop_df_reset['source'] == 'existing'
        if ref_mask.any():
            ref_individuals = pop_df_reset[ref_mask]['individual'].values
            ref_data_subset = df_ref_only[df_ref_only['individual'].isin(
                ref_individuals)]
            for col in valid_cols:
                if col in ref_data_subset.columns:
                    ind_to_geno = dict(
                        zip(ref_data_subset['individual'], ref_data_subset[col]))
                    pop_encoded.loc[ref_mask, col] = pop_df_reset.loc[ref_mask, 'individual'].map(
                        ind_to_geno)

        if uploaded_file is not None:
            up_mask = pop_df_reset['source'] == 'uploaded'
            if up_mask.any():
                df_up_only = df[df['__source__'] == 'uploaded'].copy()
                up_individuals = pop_df_reset[up_mask]['individual'].values
                up_data_subset = df_up_only[df_up_only['individual'].isin(
                    up_individuals)]
                for col in valid_cols:
                    if col in up_data_subset.columns:
                        ind_to_geno = dict(
                            zip(up_data_subset['individual'], up_data_subset[col]))
                        pop_encoded.loc[up_mask, col] = pop_df_reset.loc[up_mask, 'individual'].map(
                            ind_to_geno)

        # Encode all genotypes
        for col in valid_cols:
            if col in pop_encoded.columns:
                pop_encoded[col] = pop_encoded[col].apply(
                    lambda g: encode_genotype_dosage(g, ref_alleles.get(col)))
            else:
                pop_encoded[col] = None

        # Impute and scale using the same scaler as reference
        try:
            pop_imputed = pd.DataFrame(imputer.transform(
                pop_encoded[valid_cols]), columns=valid_cols)
            pop_scaled = pd.DataFrame(scaler.transform(
                pop_imputed), columns=valid_cols)
        except Exception as e:
            st.error(f"Error processing population data: {str(e)}")
            st.stop()

        all_scaled = np.asarray(pop_scaled)

        if compare_individuals:
            # Individual-to-individual distance
            inds = pop_df_reset['individual'].tolist()
            selected_idx = inds.index(selected_ind)
            selected_vec = all_scaled[selected_idx].reshape(1, -1)
            # Compute distance to all other individuals
            from scipy.spatial.distance import cdist
            dists = cdist(selected_vec, all_scaled,
                          metric=dist_metric).flatten()
            dist_df = pd.DataFrame({
                'Individual': inds,
                'Distance': dists,
                'Group': pop_df_reset['group'].tolist()
            })
            # Exclude the selected individual itself
            dist_df = dist_df[dist_df['Individual'] != selected_ind]
            dist_df = dist_df.sort_values('Distance')
            st.markdown(
                """
                <div style='margin:0.8em 0;'>
                  <div style='background:var(--tm-card-bg); border:1px solid var(--tm-card-border); border-radius:0.9em; padding:0.9em 1.1em; max-width:100%'>
                    <div style='font-weight:700; color:var(--tm-primary); margin-bottom:0.35em;'>What is Individual Distance?</div>
                    <div style='color:var(--tm-muted); font-size:0.98em; line-height:1.35;'>
                      <div>• Each row is a single individual (reference or uploaded).</div>
                      <div>• Distance is computed between the selected individual and all others.</div>
                      <div>• Smaller distance = more similar genotype profile.</div>
                      <div style='margin-top:0.45em; color:var(--tm-sub); font-size:0.95em;'>Uses the same trait-associated SNPs as the PCA.</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.dataframe(dist_df, width='stretch')
        else:
            # Group-to-group (original)
            # Map DataFrame indices to numpy array row indices
            idx_map = {idx: i for i, idx in enumerate(pop_df_reset.index)}
            group_indices = {g: [
                idx_map[idx] for idx in pop_df_reset.index[pop_df_reset['group'] == g].tolist()] for g in pop_groups}
            # Compute mean vector for each group
            group_means = {g: all_scaled[group_indices[g], :].mean(
                axis=0) for g in pop_groups}
            selected_vec = group_means[selected_group].reshape(1, -1)
            all_vecs = np.stack([group_means[g] for g in pop_groups])
            if dist_metric != "mahalanobis":
                from scipy.spatial.distance import cdist
                dists = cdist(selected_vec, all_vecs,
                              metric=dist_metric).flatten()
            else:
                from scipy.spatial.distance import mahalanobis
                cov = np.cov(all_scaled, rowvar=False)
                try:
                    inv_cov = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    cov += np.eye(cov.shape[0]) * 1e-8
                    inv_cov = np.linalg.inv(cov)
                dists = np.array([
                    mahalanobis(selected_vec.flatten(), all_vecs[i], inv_cov)
                    for i in range(all_vecs.shape[0])
                ])

            st.markdown(
                """
                <div style='margin:0.8em 0;'>
                  <div style='background:var(--tm-card-bg); border:1px solid var(--tm-card-border); border-radius:0.9em; padding:0.9em 1.1em; max-width:100%'>
                    <div style='font-weight:700; color:var(--tm-primary); margin-bottom:0.35em;'>What is Population Distance?</div>
                    <div style='color:var(--tm-muted); font-size:0.98em; line-height:1.35;'>
                      <div>• Mean genotype per group (SNPs standardized).</div>
                      <div>• Compare group means using the chosen metric (Euclidean or Manhattan).</div>
                      <div>• Populations are ranked by genetic similarity (smaller distance = more similar).</div>
                      <div style='margin-top:0.45em; color:var(--tm-sub); font-size:0.95em;'>Uses the same trait-associated SNPs as the PCA.</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Add sample size for each group
            group_sizes = {g: len(group_indices[g]) for g in pop_groups}
            dist_df = pd.DataFrame({
                'Population': pop_groups,
                'Distance': dists,
                'Sample Size': [group_sizes[g] for g in pop_groups]
            })
            # Exclude the selected population itself
            dist_df = dist_df[dist_df['Population'] != selected_group]
            dist_df = dist_df.sort_values('Distance')
            st.dataframe(dist_df, width='stretch')
