import pandas as pd
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker,
    HoverTool, Select, CustomJS, Span, LogColorMapper, CustomJSTickFormatter,
    MultiChoice
)
from bokeh.layouts import column
from bokeh.palettes import Viridis256

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ---------- CONFIG ----------
BASE_PATH = 'datasets/transfermarkt/'
OUTPUT_HTML = 'club_country_transfer_heatmaps.html'
TOP_N_CLUBS = 50
# Season filter: List of seasons to include, e.g., ['20/21', '21/22', '22/23']
# Set to None or empty list to include all seasons
SELECTED_SEASONS = None  # Default: all seasons

# ---------- HELPER FUNCTIONS ----------

def preprocess_clubs(clubs, competitions):
    """
    Attach country info to clubs via domestic_competition_id.
    """
    comp_country = competitions[['competition_id', 'country_name']].drop_duplicates()
    merged = clubs.merge(
        comp_country,
        left_on='domestic_competition_id',
        right_on='competition_id',
        how='left'
    )
    merged['country_name'] = merged['country_name'].fillna('Unknown')
    merged = merged.rename(columns={'country_name': 'club_country'})
    return merged

def build_top_clubs(games, clubs, top_n=50):
    """
    Determine top N clubs by number of games played (home + away).
    """
    club_counts = pd.concat([
        games['home_club_id'],
        games['away_club_id']
    ]).value_counts()
    top_club_ids = club_counts.head(top_n).index.tolist()
    known_ids = set(clubs['club_id'])
    top_club_ids = [cid for cid in top_club_ids if cid in known_ids]
    return top_club_ids

def parse_fee(value):
    """
    Convert Transfermarkt-like transfer_fee strings to numeric EUR.
    - Non-cash markers (free, loan, ?, etc.) -> 0
    - Supports '10m', '500k', '€35m', etc.
    """
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return 0.0
    lower = s.lower()

    non_cash_markers = [
        'free',
        'loan',
        'end of loan',
        '?-option',
        'option to buy',
        'swap',
        '?',
        '-'
    ]
    if any(m in lower for m in non_cash_markers):
        return 0.0

    for token in ['€', 'eur', 'fee']:
        lower = lower.replace(token, '')
    lower = lower.replace(',', '').strip()

    multiplier = 1.0
    if lower.endswith('m'):
        multiplier = 1_000_000
        lower = lower[:-1]
    elif lower.endswith('k'):
        multiplier = 1_000
        lower = lower[:-1]

    try:
        return float(lower) * multiplier
    except ValueError:
        return 0.0

def classify_club_country(club_id, club_name, club_country_map):
    """
    Map club to a 'country bucket' for rows.
    Special handling for Without Club / Retired.
    """
    if pd.notna(club_id) and club_id in club_country_map:
        return club_country_map[club_id]

    if isinstance(club_name, str):
        name = club_name.strip().lower()
        if 'without club' in name or 'no club' in name:
            return 'Without Club'
        if 'retired' in name:
            return 'Retired'

    return 'Unknown'

def build_transfer_enriched(transfers, club_country_map):
    """
    Add:
      - fee_eur
      - from_country (country bucket for from_club)
      - to_country   (country bucket for to_club)
    """
    tf = transfers.copy()
    tf['fee_eur'] = tf['transfer_fee'].apply(parse_fee)

    tf['from_country'] = tf.apply(
        lambda r: classify_club_country(
            r.get('from_club_id'),
            r.get('from_club_name'),
            club_country_map
        ),
        axis=1
    )
    tf['to_country'] = tf.apply(
        lambda r: classify_club_country(
            r.get('to_club_id'),
            r.get('to_club_name'),
            club_country_map
        ),
        axis=1
    )
    return tf

def filter_transfers_by_seasons(transfers, seasons):
    """
    Filter transfers dataframe to only include specified seasons.
    If seasons is empty or None, return all transfers.
    """
    if not seasons or len(seasons) == 0:
        return transfers
    return transfers[transfers['transfer_season'].isin(seasons)]

def ordered_rows(index, extras=('Without Club', 'Retired')):
    """
    Sort rows alphabetically, keep special rows at bottom in given order.
    """
    idx = list(index)
    core = sorted([v for v in idx if v not in extras])
    tail = [v for v in extras if v in idx]
    return core + tail

def sort_seasons_chronologically(seasons):
    """
    Sort seasons in 'YY/YY' format chronologically.
    E.g., ['98/99', '99/00', ..., '09/10', '10/11', ..., '25/26']
    """
    def season_sort_key(season):
        # Extract the first year from 'YY/YY' format
        try:
            year_str = season.split('/')[0]
            year = int(year_str)
            # Assume seasons from 90-99 are 1990s, 00-89 are 2000s+
            # This handles the century transition properly
            if year >= 90:
                return 1900 + year
            else:
                return 2000 + year
        except (ValueError, IndexError):
            # If parsing fails, return a high value to put it at the end
            return 9999
    
    return sorted(seasons, key=season_sort_key)

# ---------- MATRIX BUILDERS ----------

def money_in_matrix(top_club_ids, transfers_enriched):
    """
    Spending per club on incoming players for each country.
    Rows: origin country (from_country)
    Cols: buying club (to_club_id)
    """
    tf = transfers_enriched[
        transfers_enriched['to_club_id'].isin(top_club_ids)
    ].copy()
    tf = tf[tf['fee_eur'] > 0]
    tf = tf[~tf['from_country'].isin(['Without Club', 'Retired', 'Unknown'])]

    if tf.empty:
        return pd.DataFrame(index=[], columns=top_club_ids)

    mat = (
        tf.groupby(['from_country', 'to_club_id'])['fee_eur']
          .sum()
          .unstack(fill_value=0)
    )
    mat = mat.reindex(columns=top_club_ids, fill_value=0)
    mat = mat.reindex(index=ordered_rows(mat.index))
    return mat

def money_out_matrix(top_club_ids, transfers_enriched):
    """
    Incomings (money) per club from outgoing players for each country.
    Rows: destination country (to_country)
    Cols: selling club (from_club_id)
    """
    tf = transfers_enriched[
        transfers_enriched['from_club_id'].isin(top_club_ids)
    ].copy()
    tf = tf[tf['fee_eur'] > 0]
    tf = tf[~tf['to_country'].isin(['Without Club', 'Retired', 'Unknown'])]

    if tf.empty:
        return pd.DataFrame(index=[], columns=top_club_ids)

    mat = (
        tf.groupby(['to_country', 'from_club_id'])['fee_eur']
          .sum()
          .unstack(fill_value=0)
    )
    mat = mat.reindex(columns=top_club_ids, fill_value=0)
    mat = mat.reindex(index=ordered_rows(mat.index))
    return mat

def players_in_matrix(top_club_ids, transfers_enriched):
    """
    Number of players received from each country per club.
    Rows: from_country (+ 'Without Club')
    Cols: to_club_id
    """
    tf = transfers_enriched[
        transfers_enriched['to_club_id'].isin(top_club_ids)
    ].copy()
    if tf.empty:
        return pd.DataFrame(index=[], columns=top_club_ids)

    mat = (
        tf.groupby(['from_country', 'to_club_id'])['player_id']
          .count()
          .unstack(fill_value=0)
    )
    mat = mat.reindex(columns=top_club_ids, fill_value=0)
    mat = mat.reindex(index=ordered_rows(mat.index, extras=('Without Club',)))
    return mat

def players_out_matrix(top_club_ids, transfers_enriched):
    """
    Number of players sent to each country per club.
    Rows: to_country, with extra:
      - 'Without Club' for leaving without new club
      - 'Retired' for retirements
    Cols: from_club_id
    """
    tf = transfers_enriched[
        transfers_enriched['from_club_id'].isin(top_club_ids)
    ].copy()
    if tf.empty:
        return pd.DataFrame(index=[], columns=top_club_ids)

    mat = (
        tf.groupby(['to_country', 'from_club_id'])['player_id']
          .count()
          .unstack(fill_value=0)
    )
    mat = mat.reindex(columns=top_club_ids, fill_value=0)
    mat = mat.reindex(
        index=ordered_rows(mat.index, extras=('Without Club', 'Retired'))
    )
    return mat

def column_percentage_matrix(matrix):
    """
    For each club (column), convert counts to percentages of that club's total.
    Columns summing to 0 become all 0.
    """
    if matrix is None or matrix.empty:
        return matrix

    pct = matrix.astype(float).copy()
    col_sums = pct.sum(axis=0)

    for col in pct.columns:
        total = col_sums[col]
        if total > 0:
            pct[col] = pct[col] / total * 100.0
        else:
            pct[col] = 0.0

    return pct

# ---------- BOKEH HEATMAP FACTORY ----------
from bokeh.palettes import Viridis256

def make_heatmap(matrix, title, club_id_to_name, club_country_map,
                 palette=Viridis256, value_label='Value',
                 pct_matrix=None, is_money=False):
    """
    Build heatmap with:
      - linear + optional log layers for absolute values
      - optional percentage layer (0-100) if pct_matrix is provided

    Returns:
      p,
      lin_rect, log_rect,
      lin_color_bar, log_color_bar,
      pct_rect, pct_color_bar
    """
    if matrix is None or matrix.empty:
        p = figure(
            title=title + ' (no data)',
            x_range=[],
            y_range=[],
            width=800,
            height=300
        )
        return p, None, None, None, None, None, None

    # --- totals for ordering ---
    col_totals = matrix.sum(axis=0).to_dict()
    row_totals = matrix.sum(axis=1).to_dict()

    specials = ['Without Club', 'Retired', 'Unknown']
    core_rows = [r for r in matrix.index if r not in specials]
    extra_rows = [r for r in matrix.index if r in specials]

    core_rows = sorted(core_rows, key=lambda r: (-row_totals.get(r, 0), r))
    rows = core_rows + [r for r in specials if r in extra_rows]

    def get_club_country(cid):
        return club_country_map.get(cid, 'Unknown')

    col_ids = list(matrix.columns)

    # --- order clubs by country & descending metric within that country ---
    ordered_club_ids = []
    seen = set()

    for country in rows:
        in_group = [
            cid for cid in col_ids
            if get_club_country(cid) == country and cid not in seen
        ]
        in_group = sorted(
            in_group,
            key=lambda cid: (-col_totals.get(cid, 0), club_id_to_name.get(cid, str(cid)))
        )
        ordered_club_ids.extend(in_group)
        seen.update(in_group)

    leftovers = [cid for cid in col_ids if cid not in seen]
    leftovers = sorted(
        leftovers,
        key=lambda cid: (
            get_club_country(cid),
            -col_totals.get(cid, 0),
            club_id_to_name.get(cid, str(cid))
        )
    )
    ordered_club_ids.extend(leftovers)

    # --- x-axis labels: "Club (Country)" ---
    display_cols = [
        f"{club_id_to_name.get(cid, str(cid))} ({get_club_country(cid)})"
        for cid in ordered_club_ids
    ]

    # --- absolute value data frame in chosen order ---
    df = matrix[ordered_club_ids].copy()
    df.columns = display_cols
    df = df.reindex(index=rows)

    df_stack = df.stack().rename('value').reset_index()
    df_stack.columns = ['country', 'club', 'value']
    values = df_stack['value']
    source = ColumnDataSource(df_stack)

    # --- color mappers: linear & optional log for absolute values ---
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    lin_mapper = LinearColorMapper(palette=palette, low=vmin, high=vmax)

    positive = values[values > 0]
    if not positive.empty and float(positive.min()) < vmax:
        pos_min = float(positive.min())
        low_log = max(pos_min, 1e-6)
        log_mapper = LogColorMapper(
            palette=palette,
            low=low_log,
            high=vmax,
            nan_color="#0b0820"
        )
    else:
        log_mapper = None

    # y factors top -> bottom
    y_factors = list(rows)[::-1]

    p = figure(
        title=title,
        x_range=display_cols,
        y_range=y_factors,
        x_axis_location='above',
        tools='hover,save,reset,pan,wheel_zoom,box_zoom',
        toolbar_location='right',
        width=1600,
        height=600
    )

    # --- rects: absolute (linear) ---
    lin_rect = p.rect(
        x='club',
        y='country',
        width=1,
        height=1,
        source=source,
        line_color=None,
        fill_color={'field': 'value', 'transform': lin_mapper}
    )

    # --- rects: absolute (log) ---
    if log_mapper is not None:
        log_rect = p.rect(
            x='club',
            y='country',
            width=1,
            height=1,
            source=source,
            line_color=None,
            fill_color={'field': 'value', 'transform': log_mapper}
        )
        log_rect.visible = False
    else:
        log_rect = None

    # --- color bars for abs ---
    # Create custom formatter for money values
    if is_money:
        tick_formatter = CustomJSTickFormatter(code="""
            return (tick / 1000000).toFixed(1) + 'M €';
        """)
    else:
        tick_formatter = None

    lin_color_bar = ColorBar(
        color_mapper=lin_mapper,
        ticker=BasicTicker(desired_num_ticks=10),
        label_standoff=8,
        location=(0, 0)
    )
    if tick_formatter:
        lin_color_bar.formatter = tick_formatter
    p.add_layout(lin_color_bar, 'right')

    if log_mapper is not None:
        log_color_bar = ColorBar(
            color_mapper=log_mapper,
            ticker=BasicTicker(desired_num_ticks=10),
            label_standoff=8,
            location=(0, 0)
        )
        if tick_formatter:
            log_color_bar.formatter = tick_formatter
        log_color_bar.visible = False
        p.add_layout(log_color_bar, 'right')
    else:
        log_color_bar = None

    # --- percentage layer (optional, for players matrices) ---
    pct_rect = None
    pct_color_bar = None
    if pct_matrix is not None:
        # percentage values in same row/col order
        pct_df = pct_matrix.reindex(index=rows, columns=ordered_club_ids, fill_value=0).copy()
        pct_df.columns = display_cols

        pct_stack = pct_df.stack().rename('pct').reset_index()
        pct_stack.columns = ['country', 'club', 'pct']

        # add corresponding absolute values for tooltips
        abs_lookup = df_stack.set_index(['country', 'club'])['value']
        pct_stack['value'] = pct_stack.apply(
            lambda r: abs_lookup.get((r['country'], r['club']), 0),
            axis=1
        )

        pct_source = ColumnDataSource(pct_stack)

        pct_mapper = LinearColorMapper(palette=palette, low=0.0, high=100.0)

        pct_rect = p.rect(
            x='club',
            y='country',
            width=1,
            height=1,
            source=pct_source,
            line_color=None,
            fill_color={'field': 'pct', 'transform': pct_mapper}
        )
        pct_rect.visible = False

        pct_color_bar = ColorBar(
            color_mapper=pct_mapper,
            ticker=BasicTicker(desired_num_ticks=10),
            label_standoff=8,
            location=(0, 0)
        )
        pct_color_bar.visible = False
        p.add_layout(pct_color_bar, 'right')

    # --- hover on absolute values (still makes sense for all modes) ---
    # Hover for absolute values (linear/log)
    abs_renderers = [lin_rect]
    if log_rect is not None:
        abs_renderers.append(log_rect)

    hover_abs = HoverTool(
        renderers=abs_renderers,
        tooltips=[
            ('Country', '@country'),
            ('Club', '@club'),
            (value_label, '@value{0,0}')
        ],
        mode='mouse'
    )
    p.add_tools(hover_abs)

    # Hover for percentage view (if present)
    if pct_rect is not None:
        hover_pct = HoverTool(
            renderers=[pct_rect],
            tooltips=[
                ('Country', '@country'),
                ('Club', '@club'),
                ('Share of club total', '@pct{0.0}%'),
                ('Absolute', '@value{0,0}')
            ],
            mode='mouse'
        )
        p.add_tools(hover_pct)

    p.xaxis.major_label_orientation = 1.0

    # --- compute groups for separators & labels ---
    groups = []
    current_country = None
    start_idx = None

    for idx, cid in enumerate(ordered_club_ids):
        ctry = get_club_country(cid)
        if current_country is None:
            current_country = ctry
            start_idx = idx
        elif ctry != current_country:
            groups.append((current_country, start_idx, idx - 1))
            current_country = ctry
            start_idx = idx

    if current_country is not None:
        groups.append((current_country, start_idx, len(ordered_club_ids) - 1))

    # ----- vertical separators BETWEEN groups -----
    # categories are mapped to 0,1,2,... so boundary between groups is idx - 0.5
    for g_idx in range(1, len(groups)):
        start_idx_of_group = groups[g_idx][1]
        boundary_loc = start_idx_of_group
        separator = Span(
            location=boundary_loc,
            dimension='height',
            line_color='lightgray',
            line_width=1.5,
            line_alpha=0.7
        )
        p.add_layout(separator)

    # ----- optional: add horizontal separators between all row factors -----
    # y_factors are in visual bottom-to-top order; Bokeh maps them to 1..N.
    for i in range(1, len(y_factors)):
        hsep = Span(
            location=i,  # between row i and i+1
            dimension='width',
            line_color='white',
            line_width=0.3,
            line_alpha=0.3
        )
        p.add_layout(hsep)

    for i in range(1, len(display_cols)):
        hsep = Span(
            location=i,  # between row i and i+1
            dimension='height',
            line_color='white',
            line_width=0.3,
            line_alpha=0.3
        )
        p.add_layout(hsep)

    return p, lin_rect, log_rect, lin_color_bar, log_color_bar, pct_rect, pct_color_bar


def build_per_season_data(top_club_ids, transfers_enriched, all_seasons):
    """
    Pre-compute aggregated matrices for each season individually.
    Returns a dictionary mapping season -> matrices for that season.
    """
    per_season = {}
    for season in all_seasons:
        season_transfers = transfers_enriched[transfers_enriched['transfer_season'] == season]
        per_season[season] = {
            'money_in': money_in_matrix(top_club_ids, season_transfers),
            'money_out': money_out_matrix(top_club_ids, season_transfers),
            'players_in': players_in_matrix(top_club_ids, season_transfers),
            'players_out': players_out_matrix(top_club_ids, season_transfers)
        }
    return per_season

def build_matrices_and_heatmaps(top_club_ids, transfers_enriched, club_id_to_name, club_country_map):
    """
    Build all matrices and heatmaps for the given transfers data.
    Returns all plot objects and their components.
    """
    # Build matrices
    mat_money_in = money_in_matrix(top_club_ids, transfers_enriched)
    mat_money_out = money_out_matrix(top_club_ids, transfers_enriched)
    mat_money_in_pct = column_percentage_matrix(mat_money_in)
    mat_money_out_pct = column_percentage_matrix(mat_money_out)
    mat_players_in = players_in_matrix(top_club_ids, transfers_enriched)
    mat_players_out = players_out_matrix(top_club_ids, transfers_enriched)
    mat_players_in_pct = column_percentage_matrix(mat_players_in)
    mat_players_out_pct = column_percentage_matrix(mat_players_out)

    # Build heatmaps
    # Money IN
    p_money_in, mi_lin_rect, mi_log_rect, mi_lin_cb, mi_log_cb, mi_pct_rect, mi_pct_cb = make_heatmap(
        mat_money_in,
        "Spending on incoming players by origin country",
        club_id_to_name,
        club_country_map,
        palette=Viridis256,
        value_label='Spent (€)',
        pct_matrix=mat_money_in_pct,
        is_money=True
    )

    # Money OUT
    p_money_out, mo_lin_rect, mo_log_rect, mo_lin_cb, mo_log_cb, mo_pct_rect, mo_pct_cb = make_heatmap(
        mat_money_out,
        "Income from outgoing players by destination country",
        club_id_to_name,
        club_country_map,
        palette=Viridis256,
        value_label='Received (€)',
        pct_matrix=mat_money_out_pct,
        is_money=True
    )

    # Players (with percentage layers)
    p_players_in, pi_lin_rect, pi_log_rect, pi_lin_cb, pi_log_cb, pi_pct_rect, pi_pct_cb = make_heatmap(
        mat_players_in,
        "Number of players received by origin country (Without Club = free agent)",
        club_id_to_name,
        club_country_map,
        palette=Viridis256,
        value_label='Players in',
        pct_matrix=mat_players_in_pct
    )

    p_players_out, po_lin_rect, po_log_rect, po_lin_cb, po_log_cb, po_pct_rect, po_pct_cb = make_heatmap(
        mat_players_out,
        "Number of players sent by destination country (Without Club = free agent)",
        club_id_to_name,
        club_country_map,
        palette=Viridis256,
        value_label='Players out',
        pct_matrix=mat_players_out_pct
    )

    return {
        'plots': {
            'money_in': p_money_in,
            'money_out': p_money_out,
            'players_in': p_players_in,
            'players_out': p_players_out
        },
        'rects': {
            'lin': [mi_lin_rect, mo_lin_rect, pi_lin_rect, po_lin_rect],
            'log': [mi_log_rect, mo_log_rect, pi_log_rect, po_log_rect],
            'pct': [mi_pct_rect, mo_pct_rect, pi_pct_rect, po_pct_rect]
        },
        'color_bars': {
            'lin': [mi_lin_cb, mo_lin_cb, pi_lin_cb, po_lin_cb],
            'log': [mi_log_cb, mo_log_cb, pi_log_cb, po_log_cb],
            'pct': [mi_pct_cb, mo_pct_cb, pi_pct_cb, po_pct_cb]
        },
        'matrices': {
            'money_in': mat_money_in,
            'money_out': mat_money_out,
            'players_in': mat_players_in,
            'players_out': mat_players_out,
            'money_in_pct': mat_money_in_pct,
            'money_out_pct': mat_money_out_pct,
            'players_in_pct': mat_players_in_pct,
            'players_out_pct': mat_players_out_pct
        }
    }


# ---------- MAIN BUILD PIPELINE ----------

def build_dashboard():
    # Load data
    clubs = load_data(BASE_PATH + 'clubs.csv')
    competitions = load_data(BASE_PATH + 'competitions.csv')
    transfers = load_data(BASE_PATH + 'transfers.csv')
    games = load_data(BASE_PATH + 'games.csv')

    if clubs is None or competitions is None or transfers is None or games is None:
        raise RuntimeError("Failed to load one or more datasets. Check file paths.")

    clubs_enriched = preprocess_clubs(clubs, competitions)

    # Maps
    club_country_map = dict(
        zip(clubs_enriched['club_id'], clubs_enriched['club_country'])
    )
    club_id_to_name = dict(
        zip(clubs_enriched['club_id'], clubs_enriched['name'])
    )

    # Top clubs by games played
    top_club_ids = build_top_clubs(games, clubs_enriched, top_n=TOP_N_CLUBS)

    # Enrich transfers with numeric fees and country buckets
    transfers_enriched = build_transfer_enriched(transfers, club_country_map)
    
    # Get unique seasons sorted chronologically
    all_seasons = sort_seasons_chronologically(transfers_enriched['transfer_season'].dropna().unique().tolist())
    
    # Apply season filter if configured
    if SELECTED_SEASONS and len(SELECTED_SEASONS) > 0:
        transfers_filtered = filter_transfers_by_seasons(transfers_enriched, SELECTED_SEASONS)
        selected_seasons_display = SELECTED_SEASONS
    else:
        transfers_filtered = transfers_enriched
        selected_seasons_display = all_seasons
    
    # Build visualizations with filtered transfers
    result = build_matrices_and_heatmaps(top_club_ids, transfers_filtered, club_id_to_name, club_country_map)
    
    p_money_in = result['plots']['money_in']
    p_money_out = result['plots']['money_out']
    p_players_in = result['plots']['players_in']
    p_players_out = result['plots']['players_out']
    
    mi_lin_rect, mo_lin_rect, pi_lin_rect, po_lin_rect = result['rects']['lin']
    mi_log_rect, mo_log_rect, pi_log_rect, po_log_rect = result['rects']['log']
    mi_pct_rect, mo_pct_rect, pi_pct_rect, po_pct_rect = result['rects']['pct']
    
    mi_lin_cb, mo_lin_cb, pi_lin_cb, po_lin_cb = result['color_bars']['lin']
    mi_log_cb, mo_log_cb, pi_log_cb, po_log_cb = result['color_bars']['log']
    mi_pct_cb, mo_pct_cb, pi_pct_cb, po_pct_cb = result['color_bars']['pct']

    # Layouts per mode
    money_layout = column(p_money_in, p_money_out)
    players_layout = column(p_players_in, p_players_out)
    players_layout.visible = False  # start with Money
    
    # Season filter widget - shows which seasons are included
    season_select = MultiChoice(
        title="Seasons",
        value=selected_seasons_display,
        options=all_seasons,
        width=600
    )
    
    # Pre-compute per-season aggregated data for dynamic filtering
    per_season_data = build_per_season_data(top_club_ids, transfers_enriched, all_seasons)
    
    # Convert per-season matrices to JSON-serializable format for JavaScript
    # Convert float club IDs to integers for consistent keys
    import json
    season_data_js = {}
    for season in all_seasons:
        season_data_js[season] = {}
        for matrix_key in ['money_in', 'money_out', 'players_in', 'players_out']:
            matrix_dict = per_season_data[season][matrix_key].to_dict('index')
            # Convert float keys to int keys
            converted = {}
            for country, clubs_data in matrix_dict.items():
                converted[country] = {int(float(k)): v for k, v in clubs_data.items()}
            season_data_js[season][matrix_key] = converted
       
    from bokeh.models import Div
    mode_select = Select(
        title="View mode",
        value="Money",
        options=["Money", "Players"]
    )

    callback = CustomJS(
        args=dict(
            mode_select=mode_select,
            money_layout=money_layout,
            players_layout=players_layout
        ),
        code="""
        const mode = mode_select.value;
        if (mode === 'Money') {
            money_layout.visible = true;
            players_layout.visible = false;
        } else {
            money_layout.visible = false;
            players_layout.visible = true;
        }
        """
    )
    mode_select.js_on_change('value', callback)

    scale_select = Select(
        title="Scale / representation",
        value="Linear",
        options=["Linear", "Log", "Percentage"]
    )
    scale_callback = CustomJS(
        args=dict(
            scale_select=scale_select,
            # order: [money_in, money_out, players_in, players_out]
            lin_rects=[mi_lin_rect, mo_lin_rect, pi_lin_rect, po_lin_rect],
            log_rects=[mi_log_rect, mo_log_rect, pi_log_rect, po_log_rect],
            lin_cbs=[mi_lin_cb, mo_lin_cb, pi_lin_cb, po_lin_cb],
            log_cbs=[mi_log_cb, mo_log_cb, pi_log_cb, po_log_cb],
            pct_rects=[mi_pct_rect, mo_pct_rect, pi_pct_rect, po_pct_rect],
            pct_cbs=[mi_pct_cb, mo_pct_cb, pi_pct_cb, po_pct_cb]
        ),
        code="""
        const mode = scale_select.value;

        function setVisible(obj, v) {
            if (obj) { obj.visible = v; }
        }

        const n = lin_rects.length;  // 4

        if (mode === 'Percentage') {
            // Show percentage layers everywhere; hide abs (lin/log)
            for (let i = 0; i < n; i++) {
                setVisible(lin_rects[i], false);
                setVisible(lin_cbs[i], false);
                setVisible(log_rects[i], false);
                setVisible(log_cbs[i], false);
                setVisible(pct_rects[i], true);
                setVisible(pct_cbs[i], true);
            }
        } else {
            const useLog = (mode === 'Log');

            // Hide all percentage layers
            for (let i = 0; i < n; i++) {
                setVisible(pct_rects[i], false);
                setVisible(pct_cbs[i], false);
            }

            // Toggle lin/log per plot, falling back to linear if no log
            for (let i = 0; i < n; i++) {
                const hasLog = !!log_rects[i] && !!log_cbs[i];
                if (useLog && hasLog) {
                    setVisible(lin_rects[i], false);
                    setVisible(lin_cbs[i], false);
                    setVisible(log_rects[i], true);
                    setVisible(log_cbs[i], true);
                } else {
                    setVisible(lin_rects[i], true);
                    setVisible(lin_cbs[i], true);
                    setVisible(log_rects[i], false);
                    setVisible(log_cbs[i], false);
                }
            }
        }
        """
    )

    scale_select.js_on_change('value', scale_callback)
    
    # Season filter callback - aggregates selected seasons and updates data sources
    # Convert season_data_js to JSON string for safer serialization
    import json
    season_data_json_string = json.dumps(season_data_js)
    
    season_callback = CustomJS(
        args=dict(
            season_select=season_select,
            season_data_json_string=season_data_json_string,  # Pass as JSON string
            all_seasons=all_seasons,
            # Data sources from all 4 plots (money_in, money_out, players_in, players_out)
            sources=[
                mi_lin_rect.data_source,
                mo_lin_rect.data_source, 
                pi_lin_rect.data_source,
                po_lin_rect.data_source
            ],
            pct_sources=[
                mi_pct_rect.data_source if mi_pct_rect else None,
                mo_pct_rect.data_source if mo_pct_rect else None,
                pi_pct_rect.data_source if pi_pct_rect else None,
                po_pct_rect.data_source if po_pct_rect else None
            ],
            matrices_keys=['money_in', 'money_out', 'players_in', 'players_out'],
            club_ids=[int(cid) for cid in result['matrices']['money_in'].columns],  # Convert to int!
            all_countries=list(result['matrices']['money_in'].index),
            club_display_names=[f"{club_id_to_name.get(cid, str(cid))} ({club_country_map.get(cid, 'Unknown')})" for cid in result['matrices']['money_in'].columns]
        ),
        code="""
        // Parse the JSON string to get season_data_js
        const season_data_js = JSON.parse(season_data_json_string);
        
        // Get selected seasons (if empty, use all)
        let selected = season_select.value;
        if (selected.length === 0) {
            selected = all_seasons;
        }
                
        // Function to aggregate matrices across selected seasons
        function aggregateSeasons(matrix_key) {
            const aggregated = {};
            
            // Initialize aggregated matrix with all countries that might appear
            for (const country of all_countries) {
                aggregated[country] = {};
                for (const club_id of club_ids) {
                    aggregated[country][club_id] = 0;
                }
            }
            
            // Sum up values from selected seasons
            let total_added = 0;
            for (const season of selected) {
                if (!season_data_js[season]) {
                    console.warn('Season data not found for:', season);
                    continue;
                }
                const season_data = season_data_js[season][matrix_key];
                if (!season_data) {
                    console.warn('Matrix data not found for season:', season, 'matrix:', matrix_key);
                    continue;
                }
                
                // Iterate through countries in the season data
                for (const country in season_data) {
                    // Initialize country if not exists (might be a new country not in all_countries)
                    if (!aggregated[country]) {
                        aggregated[country] = {};
                        for (const club_id of club_ids) {
                            aggregated[country][club_id] = 0;
                        }
                    }
                    
                    const country_data = season_data[country];
                    
                    // Iterate through clubs in this country's data
                    for (const club_id_key in country_data) {
                        const value = country_data[club_id_key];
                        
                        // club_id_key might be a string or number, normalize it
                        const club_id_num = Number(club_id_key);
                        
                        // Check if this club is in our club_ids list
                        if (club_ids.includes(club_id_num)) {
                            if (!aggregated[country][club_id_num]) {
                                aggregated[country][club_id_num] = 0;
                            }
                            aggregated[country][club_id_num] += value;
                            if (value > 0) total_added++;
                        }
                    }
                }
            }
            
            return aggregated;
        }
                
        // Function to convert matrix to stacked format for data source
        function matrixToStacked(matrix) {
            const countries = [];
            const clubs = [];
            const values = [];
            
            for (const country of all_countries) {
                if (!matrix[country]) {
                    console.warn('Country not found in matrix:', country);
                    continue;
                }
                
                for (let i = 0; i < club_ids.length; i++) {
                    const club_id = club_ids[i];
                    const club_display = club_display_names[i];
                    countries.push(country);
                    clubs.push(club_display);
                    values.push(matrix[country][club_id] || 0);
                }
            }
            
            return { country: countries, club: clubs, value: values };
        }
        
        // Function to calculate percentages from absolute values
        function calculatePercentages(matrix) {
            const pct_matrix = {};
            
            // Calculate column sums
            const col_sums = {};
            for (const club_id of club_ids) {
                col_sums[club_id] = 0;
                for (const country of all_countries) {
                    if (matrix[country]) {
                        col_sums[club_id] += matrix[country][club_id] || 0;
                    }
                }
            }
            
            // Calculate percentages
            for (const country of all_countries) {
                pct_matrix[country] = {};
                for (const club_id of club_ids) {
                    const total = col_sums[club_id];
                    if (total > 0 && matrix[country]) {
                        pct_matrix[country][club_id] = (matrix[country][club_id] || 0) / total * 100.0;
                    } else {
                        pct_matrix[country][club_id] = 0;
                    }
                }
            }
            
            return pct_matrix;
        }
        
        // Update each visualization
        for (let i = 0; i < matrices_keys.length; i++) {
            const matrix_key = matrices_keys[i];
            const source = sources[i];
            const pct_source = pct_sources[i];
            
            // Aggregate data for this matrix
            const aggregated = aggregateSeasons(matrix_key);
            
            // Update absolute value source
            const stacked = matrixToStacked(aggregated);
            source.data = stacked;
            
            // Update percentage source if it exists
            if (pct_source) {
                const pct_aggregated = calculatePercentages(aggregated);
                const pct_stacked = matrixToStacked(pct_aggregated);
                pct_source.data = {
                    country: pct_stacked.country,
                    club: pct_stacked.club,
                    pct: pct_stacked.value,
                    value: stacked.value  // Include absolute values for tooltip
                };
            }
        }
                """
    )
    
    season_select.js_on_change('value', season_callback)
    
    # Create a debug div that will execute JavaScript to check season_data_js
    from bokeh.models import Div
    debug_div = Div(text="""
        <script>
        // This script will execute when the page loads
        setTimeout(function() {
            console.log('=== Debug from Div script ===');
            console.log('Checking if season_data_js is accessible...');
            // Note: season_data_js might not be directly accessible from here
            // since it's scoped to the CustomJS callback
        }, 1000);
        </script>
        <div style="display:none;">Debug div loaded</div>
    """, visible=False)

    layout = column(mode_select, scale_select, season_select, money_layout, players_layout, debug_div)

    output_file(OUTPUT_HTML, title="Club-Country Transfer Heatmaps")
    show(layout)

if __name__ == "__main__":
    build_dashboard()
