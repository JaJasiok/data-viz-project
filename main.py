import pandas as pd
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker,
    HoverTool, Select, CustomJS, Span, LogColorMapper
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

def ordered_rows(index, extras=('Without Club', 'Retired')):
    """
    Sort rows alphabetically, keep special rows at bottom in given order.
    """
    idx = list(index)
    core = sorted([v for v in idx if v not in extras])
    tail = [v for v in extras if v in idx]
    return core + tail

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
                 pct_matrix=None):
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
    lin_color_bar = ColorBar(
        color_mapper=lin_mapper,
        ticker=BasicTicker(desired_num_ticks=10),
        label_standoff=8,
        location=(0, 0)
    )
    p.add_layout(lin_color_bar, 'right')

    if log_mapper is not None:
        log_color_bar = ColorBar(
            color_mapper=log_mapper,
            ticker=BasicTicker(desired_num_ticks=10),
            label_standoff=8,
            location=(0, 0)
        )
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
        pct_matrix=mat_money_in_pct
    )

    # Money OUT
    p_money_out, mo_lin_rect, mo_log_rect, mo_lin_cb, mo_log_cb, mo_pct_rect, mo_pct_cb = make_heatmap(
        mat_money_out,
        "Income from outgoing players by destination country",
        club_id_to_name,
        club_country_map,
        palette=Viridis256,
        value_label='Received (€)',
        pct_matrix=mat_money_out_pct
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

    # Layouts per mode
    money_layout = column(p_money_in, p_money_out)
    players_layout = column(p_players_in, p_players_out)
    players_layout.visible = False  # start with Money

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

    layout = column(mode_select, scale_select, money_layout, players_layout)

    output_file(OUTPUT_HTML, title="Club-Country Transfer Heatmaps")
    show(layout)

if __name__ == "__main__":
    build_dashboard()
