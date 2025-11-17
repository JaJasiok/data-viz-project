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

from matrix_builders import money_in_matrix, money_out_matrix, column_percentage_matrix, players_in_matrix, \
    players_out_matrix


# ---------- BOKEH HEATMAP FACTORY ----------
def make_heatmap(matrix, title, club_id_to_name, club_country_map,
                 palette=Viridis256, value_label='Value',
                 pct_matrix=None, is_money=False, enable_tap=False):
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

    tools_list = 'hover,save,reset,pan,wheel_zoom,box_zoom'
    if enable_tap:
        tools_list = 'hover,tap,save,reset,pan,wheel_zoom,box_zoom'
    
    p = figure(
        title=title,
        x_range=display_cols,
        y_range=y_factors,
        x_axis_location='above',
        tools=tools_list,
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


def build_per_season_data(club_ids, transfers_enriched, all_seasons):
    """
    Pre-compute aggregated matrices for each season individually.
    Returns a dictionary mapping season -> matrices for that season.
    """
    per_season = {}
    for season in all_seasons:
        season_transfers = transfers_enriched[transfers_enriched['transfer_season'] == season]
        per_season[season] = {
            'money_in': money_in_matrix(club_ids, season_transfers),
            'money_out': money_out_matrix(club_ids, season_transfers),
            'players_in': players_in_matrix(club_ids, season_transfers),
            'players_out': players_out_matrix(club_ids, season_transfers)
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
        is_money=True,
        enable_tap=True
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
