# --- TraitMap Title and Dataset Info ---
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
st.set_page_config(
    page_title="TraitMap: Trait-Based Ancestry Explorer", layout="wide")
st.title("TraitMap: Trait-Based Ancestry Explorer")
st.markdown("""
**Reference samples:** This PCA includes individuals from the 1000 Genomes, SGDP, and HGDP datasets.  
You can find the reference genotype data [here](https://docs.google.com/spreadsheets/d/1086enwO19h-ruj61SxqLnvn-IXtMRbG50VOCuR-5xg0/edit?gid=1970300111#gid=1970300111).

**Important note:** This site analyzes just 128 SNPs, each linked to specific genetic traits. It does not use genome-wide data and cannot provide a complete picture of your ancestry. The results show patterns based on trait-associated markers only, not a full ancestry breakdown.
\n**Privacy notice:** Uploaded genotype data is processed only in your browser session and is not stored on any server. Your data is discarded immediately after analysis and never shared or saved.
""")


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
    df = pd.read_csv('merged_genotypes.csv')
    df['__source__'] = 'existing'
    return df


# Admixture calculator selection
admixture_file = st.sidebar.selectbox(
    "Admixture calculator:",
    ["Globe13 (default)", "puntDNAL"],
    index=0
)
if admixture_file == "puntDNAL":
    admixture_path = 'merged_puntDNAL.csv'
else:
    admixture_path = 'merged_globe13.csv'


def load_admixture():
    return pd.read_csv(admixture_path)


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

# Add interactive filtering controls in sidebar
st.sidebar.header("Display Options")

# Add checkboxes to show/hide by dominant ancestry
show_dominant_ancestry = st.sidebar.checkbox(
    "Color by dominant ancestry", value=False)

# If user uploads, align their data to reference SNPs
if uploaded_file is not None:
    try:
        # Try reading as 23andMe/Ancestry raw data format first without header
        df_raw = pd.read_csv(uploaded_file, sep='\t', comment='#', header=None)

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
            df_raw = pd.read_csv(uploaded_file, sep='\t', comment='#')

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
geno_cols = [col for col in df.columns if col not in meta_cols]

# Always train PCA on reference data only
df_ref_only = df[df['__source__'] == 'existing'].copy()
geno_cols = [col for col in df_ref_only.columns if col not in [
    'source', 'group', 'group_full', 'individual', '__source__']]

# Encode reference genotypes - only for columns with some data
df_ref_encoded = df_ref_only.copy()
ref_alleles = {}
for col in geno_cols:
    if not df_ref_only[col].isna().all():  # Skip columns with all NaN
        alleles = df_ref_only[col].dropna().astype(str).str.cat()
        if alleles:
            ref_allele = max(set(alleles), key=alleles.count)
            ref_alleles[col] = ref_allele
        else:
            ref_alleles[col] = None
        df_ref_encoded[col] = df_ref_only[col].apply(
            lambda g: encode_genotype_dosage(g, ref_allele))
    else:
        ref_alleles[col] = None

# Impute and scale reference data
df_ref_pca = df_ref_encoded[geno_cols].copy()
# Remove columns that are all NaN
valid_cols = [col for col in geno_cols if not df_ref_pca[col].isna().all()]
df_ref_pca = df_ref_pca[valid_cols]

imputer = SimpleImputer(strategy='most_frequent')
df_ref_imputed = pd.DataFrame(
    imputer.fit_transform(df_ref_pca), columns=valid_cols)

scaler = StandardScaler()
df_ref_scaled = pd.DataFrame(
    scaler.fit_transform(df_ref_imputed), columns=valid_cols)

# Train PCA on reference data only
n_components = 2
pca = PCA(n_components=n_components)
ref_pca_result = pca.fit_transform(df_ref_scaled)

# Create PCA dataframe with reference samples
pca_df = pd.DataFrame(ref_pca_result, columns=[
                      f'PC{i+1}' for i in range(n_components)])
pca_df['group'] = df_ref_only['group'].values
pca_df['individual'] = df_ref_only['individual'].values
pca_df['source'] = df_ref_only['__source__'].values

# Merge with admixture data
pca_df = pca_df.merge(
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


pca_df['dominant_ancestry'] = pca_df.apply(get_dominant_ancestry, axis=1)

# Format admixture data for hover tooltip (show all non-zero components)


def format_admixture_tooltip(row):
    # Get all ancestry components > 0 to show complete breakdown
    ancestry_values = []
    for comp in ancestry_components:
        if pd.notna(row[comp]) and row[comp] > 0:
            ancestry_values.append((comp, row[comp]))

    # Sort by percentage (highest first)
    ancestry_values.sort(key=lambda x: x[1], reverse=True)

    tooltip_parts = [f"{comp}: {pct:.1f}%" for comp, pct in ancestry_values]
    return "<br>".join(tooltip_parts)


pca_df['admixture_tooltip'] = pca_df.apply(format_admixture_tooltip, axis=1)

# If uploaded data exists, project it onto the same PCA space
if uploaded_file is not None:
    df_up_only = df[df['__source__'] == 'uploaded'].copy()

    # Encode uploaded sample using same reference alleles
    df_up_encoded = df_up_only.copy()
    for col in valid_cols:  # Only process valid columns
        if col in df_up_only.columns:
            df_up_encoded[col] = df_up_only[col].apply(
                lambda g: encode_genotype_dosage(g, ref_alleles.get(col)))
        else:
            df_up_encoded[col] = None

    # Impute and scale uploaded sample using fitted imputer and scaler
    df_up_pca = df_up_encoded[valid_cols].copy()  # Only use valid columns
    try:
        df_up_imputed = pd.DataFrame(
            imputer.transform(df_up_pca), columns=valid_cols)
        df_up_scaled = pd.DataFrame(
            scaler.transform(df_up_imputed), columns=valid_cols)
    except Exception as e:
        st.error(f"Error processing uploaded sample: {str(e)}")
        st.stop()

    # Project uploaded sample onto existing PCA space
    up_pca_result = pca.transform(df_up_scaled)

    # Add uploaded sample to PCA dataframe
    up_pca_df = pd.DataFrame(up_pca_result, columns=[
                             f'PC{i+1}' for i in range(n_components)])
    up_pca_df['group'] = df_up_only['group'].values
    up_pca_df['individual'] = df_up_only['individual'].values
    up_pca_df['source'] = df_up_only['__source__'].values

    # Add simplified data for uploaded samples
    up_pca_df['dominant_ancestry'] = 'YOU'
    up_pca_df['admixture_tooltip'] = 'No admixture data'

    pca_df = pd.concat([pca_df, up_pca_df], ignore_index=True)

if len(pca_df) < 2:
    st.error('Not enough data for PCA visualization.')
else:
    # Use all data (no filtering)
    filtered_df = pca_df.copy()

    # Choose coloring and shape scheme
    color_by = 'dominant_ancestry' if show_dominant_ancestry else 'group'
    symbol_by = 'dominant_ancestry' if show_dominant_ancestry else 'group'

    # Create scatter plot with diverse shapes and colors
    fig = px.scatter(
        filtered_df, x='PC1', y='PC2',
        color=color_by,
        symbol=symbol_by,  # Use shapes based on dominant ancestry when that mode is enabled
        hover_name='individual',
        hover_data={
            'group': True,
            'admixture_tooltip': True,
            'PC1': ':.2f',
            'PC2': ':.2f'
        },
        title=f'PCA: PC1 vs PC2 (Colored by {"{}".format("Dominant Ancestry" if show_dominant_ancestry else "Population Group")})',
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

    # Create diverse marker symbols
    marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up',
                      'triangle-down', 'pentagon', 'hexagon', 'star', 'triangle-left',
                      'triangle-right', 'star-triangle-up', 'star-square', 'asterisk']

    # Update marker symbols and sizes for each trace
    for i, trace in enumerate(fig.data):
        trace_name = str(trace.name)

        # Set unique symbol for each group/ancestry
        trace.marker.symbol = marker_symbols[i % len(marker_symbols)]

        # Make uploaded samples (YOU) a yellow big star
        if 'you' in trace_name.lower():
            trace.marker.size = 15
            trace.marker.symbol = 'star'
            trace.marker.color = 'gold'
        else:
            trace.marker.size = 8

    # Clean up legend
    fig.update_layout(
        legend=dict(
            title=color_by.replace('_', ' ').title(),
            tracegroupgap=0,
        ),
        showlegend=True,
        height=600
    )

    # Display sample count info
    st.info(f"Showing {len(filtered_df)} samples")

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Load trait data
    trait_df = pd.read_csv('merged_traits.csv')

    # Add selectbox to choose sample for trait data
    st.markdown("---")
    st.markdown("### 🧬 Trait Data Explorer")
    st.markdown("Select a sample to view its predicted genetic traits:")

    # Get available individuals for selection
    available_individuals = sorted(filtered_df['individual'].unique())

    selected_individual = st.selectbox(
        "Choose a sample:",
        options=[''] + available_individuals,
        index=0,
        key="trait_selector"
    )

    # Show trait data for selected individual
    if selected_individual:
        trait_row = trait_df[trait_df['individual'] == selected_individual]
        st.markdown(f"#### Trait predictions for: **{selected_individual}**")
        if not trait_row.empty:
            # Show all trait columns except meta
            meta_cols = ['source', 'group', 'group_full', 'individual']
            trait_data = trait_row.drop(columns=meta_cols, errors='ignore').T
            trait_data.columns = ['Probability (%)']

            # Format trait names for better display
            trait_data.index = trait_data.index.str.replace(
                '_', ' ').str.title()

            # Display as three columns for better layout
            col1, col2, col3 = st.columns(3)

            # Split traits into categories
            eye_traits = trait_data[trait_data.index.str.contains('Eye')]
            hair_traits = trait_data[trait_data.index.str.contains('Hair')]
            skin_traits = trait_data[trait_data.index.str.contains('Skin')]

            with col1:
                if not eye_traits.empty:
                    st.markdown("**👁️ Eye Color**")
                    st.dataframe(eye_traits, use_container_width=True)

            with col2:
                if not hair_traits.empty:
                    st.markdown("**💇 Hair Color**")
                    st.dataframe(hair_traits, use_container_width=True)

            with col3:
                if not skin_traits.empty:
                    st.markdown("**🎨 Skin Color**")
                    st.dataframe(skin_traits, use_container_width=True)
        else:
            st.info("No trait data available for this sample.")
