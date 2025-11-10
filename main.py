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

from config import BASE_PATH, TOP_N_CLUBS, SELECTED_SEASONS, OUTPUT_HTML
from data_processing import load_data, preprocess_clubs, build_top_clubs, build_transfer_enriched, \
    sort_seasons_chronologically, filter_transfers_by_seasons, ordered_rows
from heatmap_factory import build_matrices_and_heatmaps, build_per_season_data


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

    # Just above season_callback, build a unified country list:
    all_countries = sorted(set(
        list(result['matrices']['money_in'].index) +
        list(result['matrices']['money_out'].index) +
        list(result['matrices']['players_in'].index) +
        list(result['matrices']['players_out'].index)
    ))

    # If you want special rows at the bottom consistently, and you still have ordered_rows():
    all_countries = ordered_rows(all_countries, extras=('Without Club', 'Retired', 'Unknown'))

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
            all_countries=all_countries,
            club_display_names=[f"{club_id_to_name.get(cid, str(cid))} ({club_country_map.get(cid, 'Unknown')})" for cid in result['matrices']['money_in'].columns],
            # NEW: pass colorbars so we can update their color_mappers
            lin_cbs=[mi_lin_cb, mo_lin_cb, pi_lin_cb, po_lin_cb],
            log_cbs=[mi_log_cb, mo_log_cb, pi_log_cb, po_log_cb],
            pct_cbs=[mi_pct_cb, mo_pct_cb, pi_pct_cb, po_pct_cb]
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
        
            // ---- recompute color scale for absolute values ----
            const lin_cb = lin_cbs[i];
            const log_cb = log_cbs[i];
        
            if (lin_cb) {
                const vals = stacked.value;
                let minv = Infinity;
                let maxv = -Infinity;
                for (let j = 0; j < vals.length; j++) {
                    const v = vals[j];
                    if (v < minv) minv = v;
                    if (v > maxv) maxv = v;
                }
                if (!isFinite(minv)) minv = 0;
                if (!isFinite(maxv) || maxv <= minv) maxv = minv + 1;
        
                // linear mapper
                lin_cb.color_mapper.low = minv;
                lin_cb.color_mapper.high = maxv;
        
                // log mapper (if exists)
                if (log_cb) {
                    const pos = vals.filter(v => v > 0);
                    if (pos.length) {
                        let minPos = pos[0];
                        for (let k = 1; k < pos.length; k++) {
                            if (pos[k] < minPos) minPos = pos[k];
                        }
                        if (minPos <= 0) minPos = 1e-6;
                        log_cb.color_mapper.low = minPos;
                        log_cb.color_mapper.high = maxv;
                    }
                }
            }
        
            // ---- update percentage source & keep its scale (0-100) ----
            if (pct_source) {
                const pct_aggregated = calculatePercentages(aggregated);
                const pct_stacked = matrixToStacked(pct_aggregated);
                pct_source.data = {
                    country: pct_stacked.country,
                    club: pct_stacked.club,
                    pct: pct_stacked.value,
                    value: stacked.value  // absolute values for tooltip
                };
                // pct_cbs[i] already uses 0..100; no need to adjust
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
