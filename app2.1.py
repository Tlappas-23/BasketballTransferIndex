import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# --------------------------
# CONFIGURATION & STYLING
# --------------------------
st.set_page_config(page_title="Basketball Transfer Index", layout="wide")

# --------------------------
# HEADER LAYOUT
# --------------------------
col1, col2 = st.columns([3, 1])  # Adjust layout proportions

with col1:
    st.markdown("## Basketball Transfer Index")
    st.markdown("Step 1: Select the Target School")
    st.markdown("Step 2: Select the Target Player")
    st.markdown("_Use filters on the left-hand side to refine by Conference, Stats, and more._")
    st.markdown("Step 3: Review projected player performance at the selected school")

with col2:
    logo_path = "BasketballTransferIndexLogo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.warning("Logo not found.")

# --------------------------
# GSPREAD CLIENT (CACHED)
# --------------------------
@st.cache_resource
def get_gspread_client():
    import json
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_dict = json.loads(st.secrets["GSPREAD_CREDS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

# --------------------------
# LOAD & CACHE DATA
# --------------------------
@st.cache_data
def load_player_data():
    client = get_gspread_client()
    sheet = client.open("Transfer Model 3.0").worksheet("Data")
    data = sheet.get_all_values()

    raw_columns = data[0]
    seen = {}
    columns = []
    for col in raw_columns:
        col_clean = col.strip()
        if col_clean in seen:
            seen[col_clean] += 1
            columns.append(f"{col_clean}.{seen[col_clean]}")
        else:
            seen[col_clean] = 0
            columns.append(col_clean)

    df = pd.DataFrame(data[1:], columns=columns)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_target_schools():
    client = get_gspread_client()
    sheet = client.open("Transfer Model 3.0").worksheet("Target School List")
    schools = sheet.col_values(1)
    return sorted(set(s for s in schools if s.strip() and s != "School"))

# --------------------------
# SAFE CONVERSIONS
# --------------------------
def safe_float(val):
    try:
        return float(str(val).replace('%',''))
    except:
        return 0.0

# --------------------------
# POSITIVE PROJECTION LOGIC
# --------------------------
def adjusted_positive_projection(current, multiplier):
    projected = current * multiplier
    while projected < 0:
        multiplier = 1 + ((multiplier - 1) / 2)
        projected = current * multiplier
    return round(projected, 2)

# --------------------------
# COLUMN & LABEL MAPPINGS
# --------------------------
stat_column_map = {
    "PPG": "PPG_Per40","2P%": "2P%_Per40","3P%": "3P%_Per40",
    "FTA": "FTA_Per40","REB": "RPG_Per40", "AST": "APG_Per40",
    "BLK": "BPG_Per40", "TOV": "TOV_Per40", "2FGM": "2PM_Per40",
    "2FGA": "2PA_Per40", "3PM": "3PM_Per40", "3PA": "3PA_Per40",
    "USG": "Est. Usage %", "GP": "GP","MPG": "MPG"
}

label_map = {
    "PPG": "Pts", "2P%": "2P%", "3P%": "3P%", "FTA": "FTA",
    "REB": "Reb", "AST": "Ast", "BLK": "Blk", "TOV": "TOV",
    "2FGM": "2pt FGM", "2FGA": "2pt FGA", "3PM": "3PM", "3PA": "3PA",
    "USG": "Est. Usage %", "GP": "Games Played", "MPG": "Minutes/Game"
}

projected_keys = [
    ("PPG", "Points Per 40"), ("2P%", "2PT FG%"), ("3P%", "3PT FG %"),
    ("FTA", "FTA per 40"), ("REB", "Reb per 40"), ("AST", "Ast per 40"),
    ("BLK", "Blk per 40"), ("TOV", "TOV per 40"), ("2FGM", "2pt FGM/40"),
    ("2FGA", "2pt FGA/40"), ("3PM", "3PM/40"), ("3PA", "3PA/40"),
    ("USG", "Est. Usage %"), ("GP", "Games Played"), ("MPG", "Minutes/Game")
]

# --------------------------
# LOAD DATA
# --------------------------
try:
    df = load_player_data()
    target_schools = load_target_schools()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# --------------------------
# Stat Basis Dropdown
# --------------------------
st.markdown("### Select Stat Basis:")
stat_basis = st.selectbox("", options=["Per 32", "Per 36", "Per 40"], index=2)

basis_minutes = int(stat_basis.split(" ")[1])
scale_factor = basis_minutes / 40

# --------------------------
# UI: TARGET SCHOOL SELECTION
# --------------------------
st.markdown("### Select Target School:")
selected_school = st.selectbox("", target_schools)

# --------------------------
# UI: PLAYER FILTERS (SIDEBAR)
# --------------------------

# Clean column names just in case
df.columns = df.columns.str.strip()

def apply_filter(series, selected_values):
    if selected_values:
        return series.isin(selected_values)
    else:
        return pd.Series([True] * len(series), index=series.index)
    
# Helper function to keep index alignment
def apply_filter(series, selected_values):
    if selected_values:
        return series.isin(selected_values)
    else:
        return pd.Series([True] * len(series), index=series.index)

st.sidebar.subheader("Player Filters")

# Step 1: Nationality filter
nationality_filter = st.sidebar.multiselect("Nationality", sorted(df['Nationality'].dropna().unique()))
df_nat = df[apply_filter(df['Nationality'], nationality_filter)]

# Step 2: Division options filtered by Nationality
division_options = sorted(df_nat['Division'].dropna().unique())
d1d2_filter = st.sidebar.multiselect("Division", division_options)
df_div = df_nat[apply_filter(df_nat['Division'], d1d2_filter)]

# Step 3: Conference options filtered by Division
conference_options = sorted(df_div['Conference'].dropna().unique())
conference_filter = st.sidebar.multiselect("Conference", conference_options)
df_conf = df_div[apply_filter(df_div['Conference'], conference_filter)]

# Step 4: School options filtered by Conference
school_options = sorted(df_conf['School'].dropna().unique())
school_filter = st.sidebar.multiselect("School", school_options)
df_school = df_conf[apply_filter(df_conf['School'], school_filter)]

# Step 5: Position options filtered by School
position_options = sorted(df_school['Pos'].dropna().unique())
position_filter = st.sidebar.multiselect("Position", position_options)

# Step 6: Points Per {basis_minutes} filter
ppg_column = "PPG_Per40"
df_school[ppg_column] = pd.to_numeric(df_school[ppg_column], errors='coerce')
min_ppg = float(df_school[ppg_column].min(skipna=True))
max_ppg = float(df_school[ppg_column].max(skipna=True))
ppg_threshold = st.sidebar.slider(f"Points Per {basis_minutes} (Minimum)",
    min_value=0.0, max_value=max_ppg, value=min_ppg, step=0.5)

# Step 7: 3PA Per {basis_minutes} filter
threepa_column = "3PA_Per40"
df_school[threepa_column] = pd.to_numeric(df_school[threepa_column], errors='coerce')
min_3pa = float(df_school[threepa_column].min(skipna=True))
max_3pa = float(df_school[threepa_column].max(skipna=True))
threepa_threshold = st.sidebar.slider(f"3PA Per {basis_minutes} (Minimum)",
    min_value=0.0, max_value=max_3pa, value=min_3pa, step=0.5)

# Step 8: Rebounds Per {basis_minutes} filter
reb_column = "RPG_Per40"
df_school[reb_column] = pd.to_numeric(df_school[reb_column], errors='coerce')
min_reb = float(df_school[reb_column].min(skipna=True))
max_reb = float(df_school[reb_column].max(skipna=True))
reb_threshold = st.sidebar.slider(f"Rebounds Per {basis_minutes} (Minimum)",
    min_value=0.0, max_value=max_reb, value=min_reb, step=0.5)

# Step 9: 3P%
threep_column = "3P%_Per40"
df_school[threep_column] = pd.to_numeric(df_school[threep_column], errors='coerce')
min_3p = float(df_school[threep_column].min(skipna=True)) * 100
max_3p = float(df_school[threep_column].max(skipna=True)) * 100
threep_threshold = st.sidebar.slider(
    f"Min 3P% (Per {basis_minutes})",
    min_value=0.0, max_value=max_3p, value=min_3p, step=1.0
)

# Step 10: Minutes per Game filter
mpg_column = "MPG"  # adjust if column name differs
df_school[mpg_column] = pd.to_numeric(df_school[mpg_column], errors='coerce')
min_mpg = float(df_school[mpg_column].min(skipna=True))
max_mpg = float(df_school[mpg_column].max(skipna=True))
mpg_threshold = st.sidebar.slider(
    "Minimum MPG", min_value=0.0, max_value=max_mpg, value=min_mpg, step=1.0
)

# Final filtered DataFrame
filtered_df = df_school[
    (apply_filter(df_school['Nationality'], nationality_filter)) &
    (apply_filter(df_school['Division'], d1d2_filter)) &
    (apply_filter(df_school['Conference'], conference_filter)) &
    (apply_filter(df_school['School'], school_filter)) &
    (apply_filter(df_school['Pos'], position_filter)) &
    (df_school[ppg_column] >= ppg_threshold) &
    (df_school[threepa_column] >= threepa_threshold) &
    (df_school[reb_column] >= reb_threshold) &
    (df_school[threep_column] * 100 >= threep_threshold) &
    (df_school[mpg_column] >= mpg_threshold)
]

# --------------------------
# PLAYER SELECTION
# --------------------------

# Safely identify the correct 'Player' column
player_columns = [col for col in filtered_df.columns if col.strip().lower() == "player"]

if not player_columns:
    st.error("No 'Player' column found in the data.")
    st.stop()

if len(player_columns) > 1:
    st.warning(f"Multiple 'Player' columns found: {player_columns}. Using the first one.")

player_column = player_columns[0]

# Get unique player names
player_names = sorted(filtered_df[player_column].dropna().unique())

# Select Player Section
if player_names:
    st.markdown("### Select Target Player:")
    selected_player = st.selectbox("", player_names, key="player_selectbox")
    player = filtered_df[filtered_df[player_column] == selected_player].iloc[0]
else:
    st.warning("No players match your selected filters. Please adjust your selections.")
    st.stop()

# --------------------------
# WRITE TO ACTIVE SHEET
# --------------------------
try:
    client = get_gspread_client()
    active_sheet = client.open("Transfer Model 3.0").worksheet("Active")

    # Update values
    active_sheet.update_acell('A2', player.get('Pos', ''))
    active_sheet.update_acell('A5', selected_player)
    active_sheet.update_acell('B5', selected_school)
    active_sheet.update_acell('A3', player.get('School', ''))
except Exception as e:
    st.warning(f"Failed to update 'Active' sheet: {e}")
    st.stop()

try:
    # Read values
    target_team_rank = active_sheet.acell('B7').value
    target_team_style = active_sheet.acell('B8').value

    # Fetch Stat Cap separately
    stat_cap_data = active_sheet.get('B13:I13')[0]

    # Fetch projection data with correct header row
    projection_data = active_sheet.get('A14:I18')
    projection_df = pd.DataFrame(projection_data[1:], columns=projection_data[0])

    if 'Summary' not in projection_df.columns:
        raise ValueError("Missing 'Summary' column in projection data")

    total_row = projection_df[projection_df['Summary'] == 'Initial Total'].iloc[0]
    stat_cap_cols = projection_df.columns[1:]  # Exclude 'Summary'
    cap_row = dict(zip(stat_cap_cols, stat_cap_data))
except Exception as e:
    st.warning(f"Failed to read from 'Active' sheet: {e}")
    target_team_rank = target_team_style = "N/A"
    projection_df = pd.DataFrame()
    total_row = {}
    cap_row = {}

# --------------------------
# PROJECTED SEASON PERFORMANCE
# --------------------------
st.markdown(f"### Forecasted Per{basis_minutes} Statistics at Target School")

projected_stat_layout = [
    ["PPG", "FTA", "REB", "AST"],
    ["BLK", "TOV", "2P%", "3P%"]
]

volume_stat_cap_multiplier = 1.6
percent_stat_cap_multiplier = 1.15
lower_is_better = {"TOV"}

target_conf = df[df["School"] == selected_school]["Conference"].values
target_conf = target_conf[0] if len(target_conf) > 0 else None

def context_from_percentile(stat, p):
    if stat == "TOV":
        return "High" if p < 33 else "Average" if p < 67 else "Low"
    else:
        return "Below Average" if p < 33 else "Average" if p < 67 else "Above Average"

for stat_row in projected_stat_layout:
    row_cols = st.columns(4)
    for i, stat in enumerate(stat_row):
        adj_col = dict(projected_keys).get(stat, "")
        col = stat_column_map.get(stat, stat)

        current_val = safe_float(player.get(col, 0))
        raw_multiplier = total_row.get(adj_col, 1)
        multiplier = safe_float(raw_multiplier)

        raw_cap = cap_row.get(adj_col)
        stat_cap = safe_float(raw_cap if raw_cap not in [None, ""] else float('inf'))

        if stat in ["2P%", "3P%"]:
            max_allowed_val = percent_stat_cap_multiplier * stat_cap
            projected_val = adjusted_positive_projection(current_val, multiplier)
            projected_val_capped = min(projected_val, max_allowed_val)

            if target_conf:
                conf_vals = df[df["Conference"] == target_conf][col].astype(float)
                percentile = round((conf_vals < projected_val_capped).mean() * 100)
                context = context_from_percentile(stat, percentile)
                caption = f"{context} for {target_conf} ({percentile} percentile)"
            else:
                caption = ""

            with row_cols[i]:
                st.markdown(f"""
                    <div style="font-size:20px; font-weight:bold;">{label_map[stat]} (Per{basis_minutes})</div>
                    <div style="font-size:28px; margin-top:4px;">{projected_val_capped * 100:.0f}%</div>
                    <div style="font-size:14px; color:gray; margin-top:2px;">{caption}</div>
                """, unsafe_allow_html=True)

        else:
            max_allowed_val = volume_stat_cap_multiplier * stat_cap
            projected_val = adjusted_positive_projection(current_val, multiplier) * scale_factor
            projected_val_capped = min(projected_val, max_allowed_val)

            if target_conf:
                conf_vals = df[df["Conference"] == target_conf][col].astype(float)
                if stat in lower_is_better:
                    percentile = round((conf_vals > projected_val_capped).mean() * 100)
                else:
                    percentile = round((conf_vals < projected_val_capped).mean() * 100)
                context = context_from_percentile(stat, percentile)
                caption = f"{context} for {target_conf} ({percentile} percentile)"
            else:
                caption = ""

            with row_cols[i]:
                st.markdown(f"""
                    <div style="font-size:20px; font-weight:bold;">{label_map[stat]} (Per{basis_minutes})</div>
                    <div style="font-size:28px; margin-top:4px;">{projected_val_capped:.1f}</div>
                    <div style="font-size:14px; color:gray; margin-top:2px;">{caption}</div>
                """, unsafe_allow_html=True)

# --------------------------
# CURRENT SEASON PERFORMANCE
# --------------------------
st.markdown(f"### Current Per{basis_minutes} Statistics")

current_stat_layout = [
    ["PPG", "FTA", "REB", "AST"],
    ["BLK", "TOV", "2P%", "3P%"]
]

player_conf = player.get("Conference", None)

def context_from_percentile(stat, p):
    if stat == "TOV":
        return "High" if p < 33 else "Average" if p < 67 else "Low"
    else:
        return "Below Average" if p < 33 else "Average" if p < 67 else "Above Average"

lower_is_better = {"TOV"}

for stat_row in current_stat_layout:
    row_cols = st.columns(4)
    for i, stat in enumerate(stat_row):
        col_name = stat_column_map.get(stat, stat)
        current_val = safe_float(player.get(col_name, "N/A"))

        if player_conf:
            conf_vals = df[df["Conference"] == player_conf][col_name].astype(float)
            if stat in lower_is_better:
                percentile = round((conf_vals > current_val).mean() * 100)
            else:
                percentile = round((conf_vals < current_val).mean() * 100)
            context = context_from_percentile(stat, percentile)
            caption = f"{context} for {player_conf} ({percentile} percentile)"
        else:
            caption = ""

        with row_cols[i]:
            if stat in ["2P%", "3P%"]:
                st.markdown(f"""
                    <div style="font-size:20px; font-weight:bold;">{label_map[stat]} (Per{basis_minutes})</div>
                    <div style="font-size:28px; margin-top:4px;">{current_val * 100:.0f}%</div>
                    <div style="font-size:14px; color:gray; margin-top:2px;">{caption}</div>
                """, unsafe_allow_html=True)
            else:
                adjusted_val = current_val * scale_factor
                st.markdown(f"""
                    <div style="font-size:20px; font-weight:bold;">{label_map[stat]} (Per{basis_minutes})</div>
                    <div style="font-size:28px; margin-top:4px;">{adjusted_val:.1f}</div>
                    <div style="font-size:14px; color:gray; margin-top:2px;">{caption}</div>
                """, unsafe_allow_html=True)

# --------------------------
# Player Profile 
# --------------------------
st.markdown(f"### Current Per{basis_minutes} Player Profile")

# Base stats
gp = safe_float(player.get("GP", 0))
mpg = safe_float(player.get("MPG", 0))
two_fga = safe_float(player.get("2PA_Per40", 0)) * scale_factor
three_pa = safe_float(player.get("3PA_Per40", 0)) * scale_factor
fta = safe_float(player.get("FTA_Per40", 0)) * scale_factor
ast = safe_float(player.get("APG_Per40", 0)) * scale_factor
tov = safe_float(player.get("TOV_Per40", 0)) * scale_factor

# Conference filter
player_conf = player.get("Conference", None)
df_conf = df[df["Conference"] == player_conf] if player_conf else df

# Calculated stats
total_fga = two_fga + three_pa
two_pt_share = (two_fga / total_fga) * 100 if total_fga > 0 else 0
three_pt_share = (three_pa / total_fga) * 100 if total_fga > 0 else 0
fta_fga_ratio = fta / total_fga if total_fga > 0 else 0
fta_fga_pct = fta_fga_ratio * 100
ast_tov_ratio = (ast / tov) if tov > 0 else 0
ast_fga_ratio = (ast / total_fga) if total_fga > 0 else 0

# Percentiles
fta_values = df_conf["FTA_Per40"].astype(float) / (
    df_conf["2PA_Per40"].astype(float) + df_conf["3PA_Per40"].astype(float)).replace(0, np.nan)
fta_percentile = round((fta_values.dropna() < fta_fga_ratio).mean() * 100)

ast_tov_values = df_conf["APG_Per40"].astype(float) / df_conf["TOV_Per40"].astype(float).replace(0, np.nan)
ast_tov_percentile = round((ast_tov_values.dropna() < ast_tov_ratio).mean() * 100)

ast_fga_values = df_conf["APG_Per40"].astype(float) / (
    df_conf["2PA_Per40"].astype(float) + df_conf["3PA_Per40"].astype(float)).replace(0, np.nan)
ast_fga_percentile = round((ast_fga_values.dropna() < ast_fga_ratio).mean() * 100)

fga_values = df_conf["2PA_Per40"].astype(float) + df_conf["3PA_Per40"].astype(float)
fga_percentile = round((fga_values < total_fga).mean() * 100)

two_pt_percentile = round((df_conf["2PA_Per40"].astype(float) < two_fga).mean() * 100)
three_pt_percentile = round((df_conf["3PA_Per40"].astype(float) < three_pa).mean() * 100)

# Contextual labels
def context_from_percentile(p):
    return "Below Average" if p < 33 else "Average" if p < 67 else "Above Average"

def fga_volume_label(percentile):
    return "Low Volume" if percentile < 20 else "Balanced" if percentile < 67 else "High Volume"

def style_label(percentile):
    return "Shoot First" if percentile < 33 else "Balanced" if percentile < 67 else "Pass First"

# Assign context labels
fga_label = fga_volume_label(fga_percentile)
fta_context = context_from_percentile(fta_percentile)
ast_tov_context = context_from_percentile(ast_tov_percentile)
ast_fga_context = style_label(ast_fga_percentile)
three_pt_context = fga_volume_label(three_pt_percentile)

# Layout
row1 = st.columns(4)
row2 = st.columns(4)

# Row 1
with row1[0]:
    st.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">GP (Season Total)</div>
        <div style="font-size:28px; margin-top:4px;">{int(round(gp))}</div>
    """, unsafe_allow_html=True)

with row1[1]:
    st.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">Shot Mix: 2PTA | 3PTA</div>
        <div style="font-size:28px; margin-top:4px;">{two_pt_share:.0f}% / {three_pt_share:.0f}%</div>
        <div style="font-size:14px; color:gray; margin-top:2px;">2PT: {two_pt_percentile} | 3PT: {three_pt_percentile}</div>
    """, unsafe_allow_html=True)

with row1[2]:
    st.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">FGA (Per{basis_minutes})</div>
        <div style="font-size:28px; margin-top:4px;">{total_fga:.1f}</div>
        <div style="font-size:14px; color:gray; margin-top:2px;">{fga_label} shooter in {player_conf} (Pctl: {fga_percentile})</div>
    """, unsafe_allow_html=True)

with row1[3]:
    st.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">3PTA (Per{basis_minutes})</div>
        <div style="font-size:28px; margin-top:4px;">{three_pa:.1f}</div>
        <div style="font-size:14px; color:gray; margin-top:2px;">{three_pt_context} 3PT shooter in {player_conf} (Pctl: {three_pt_percentile})</div>
    """, unsafe_allow_html=True)

# Row 2
with row2[0]:
    st.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">MPG (Season Avg)</div>
        <div style="font-size:28px; margin-top:4px;">{int(round(mpg))}</div>
    """, unsafe_allow_html=True)

with row2[1]:
    st.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">FTA / FGA (%)</div>
        <div style="font-size:28px; margin-top:4px;">{fta_fga_pct:.0f}%</div>
        <div style="font-size:14px; color:gray; margin-top:2px;">{fta_context} at drawing fouls ({fta_percentile} Pctl)</div>
    """, unsafe_allow_html=True)

with row2[2]:
    st.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">AST / TOV</div>
        <div style="font-size:28px; margin-top:4px;">{ast_tov_ratio:.1f}</div>
        <div style="font-size:14px; color:gray; margin-top:2px;">{ast_tov_context} decision-maker ({ast_tov_percentile} Pctl)</div>
    """, unsafe_allow_html=True)

with row2[3]:
    st.markdown(f"""
        <div style="font-size:20px; font-weight:bold;">AST / FGA (%)</div>
        <div style="font-size:28px; margin-top:4px;">{ast_fga_ratio * 100:.0f}%</div>
        <div style="font-size:14px; color:gray; margin-top:2px;">{ast_fga_context} archetype ({ast_fga_percentile} Pctl)</div>
    """, unsafe_allow_html=True)

# --------------------------
# Player Detail
# --------------------------
st.markdown("### Player Detail")

# --------------------------
# Google Search Link
# --------------------------
search_name = selected_player.replace(" ", "+")
search_url = f"https://www.google.com/search?q={search_name}+basketball+stats"

# Define layout: 4 columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"**Current School:** {player.get('School', 'N/A')}")
    st.markdown(f"**Position:** {player.get('Pos', 'N/A')}")
    st.markdown(f"**Division:** {player.get('Division', 'N/A')}")
    st.markdown(f"**Class Year:** {player.get('Class', 'N/A')}")
    st.markdown(f"[{selected_player} Stat Search]({search_url})", unsafe_allow_html=True)

with col2:
    st.markdown(f"**Target School:** {selected_school}")
    transfer_flag = player.get('Previous Transfer', '0')
    transfer_status = 'Yes' if str(transfer_flag).strip() != '0' else 'No'
    st.markdown(f"**Previous Transfer:** {transfer_status}")

with col3:
    st.markdown(f"**Height:** {player.get('HT', 'N/A')}")
    st.markdown(f"**Weight:** {player.get('WT', 'N/A')} lbs")

with col4:
    st.markdown(f"**Nationality:** {player.get('Nationality', 'N/A')}")
    st.markdown(f"**Birth City:** {player.get('Birth City', 'N/A')}")
    st.markdown(f"**HS / Prep School:** {player.get('High School/Prep School', 'N/A')}")
    st.markdown(f"**Birth Date:** {player.get('Birth Date', 'N/A')}")

# --------------------------
# BAR CHART COMPARISON
# --------------------------
st.markdown(f"### Per{basis_minutes} Stats: Current vs Forecasted")

# Define volume stats to compare
volume_stats = ["PPG", "FTA", "REB", "AST", "BLK", "TOV"]

# Safely extract current values
current_vals = [
    safe_float(player.get(stat_column_map.get(stat, ""), 0)) * scale_factor
    for stat in volume_stats
]

# Get multipliers and project values
adjustments = [safe_float(total_row.get(dict(projected_keys).get(stat, ""), 1)) for stat in volume_stats]
projected_vals = [
    adjusted_positive_projection(cur / scale_factor, mult) * scale_factor
    for cur, mult in zip(current_vals, adjustments)
]

# Labels
labels = [label_map.get(stat, stat) for stat in volume_stats]
x = np.arange(len(labels))
bar_width = 0.35

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

bars_current = ax.bar(x - bar_width/2, current_vals, bar_width,
                      label='Current', color='steelblue', edgecolor='black')
bars_projected = ax.bar(x + bar_width/2, projected_vals, bar_width,
                        label='Forecasted', color='darkorange', edgecolor='black')

# Axes and Title
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(f"Per {basis_minutes} Minutes", fontsize=11)
ax.set_title(f"{selected_player} – Current vs Forecasted Stats", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Y-axis scale with buffer
y_max = max(max(current_vals), max(projected_vals)) * 1.15 if current_vals and projected_vals else 1
ax.set_ylim(0, y_max)

# Annotate bars
for bar in bars_current + bars_projected:
    height = bar.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

fig.tight_layout()
st.pyplot(fig)

# --------------------------
# FOOTER DISCLAIMER
# --------------------------
st.markdown("---")
st.markdown(
    "*Notes:*  \n"
    "Players listed in the dropdown may or may not have eligibility for the upcoming season.  \n"
    "The stats forecasted may or may not reflect actual results.  \n\n"
    "**Disclaimer:**  \n"
    "**THE USE OR RELIANCE OF ANY INFORMATION CONTAINED ON THIS SITE IS SOLELY AT YOUR OWN RISK.**"
)
