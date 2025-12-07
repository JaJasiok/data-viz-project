import pandas as pd
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker,
    HoverTool, Select, CustomJS, Span, LogColorMapper, CustomJSTickFormatter,
    MultiChoice, Div, CheckboxButtonGroup
)
from bokeh.layouts import column, row, Spacer
from bokeh.palettes import Viridis256

from config import BASE_PATH, TOP_N_CLUBS, SELECTED_SEASONS, OUTPUT_HTML
from data_processing import load_data, preprocess_clubs, build_top_clubs, build_transfer_enriched, \
    sort_seasons_chronologically, filter_transfers_by_seasons, ordered_rows, calculate_team_statistics, \
    calculate_per_season_statistics, build_top_spenders
from heatmap_factory import build_matrices_and_heatmaps, build_per_season_data


# ---------- MAIN BUILD PIPELINE ----------

def build_dashboard():
    # Load external CSS first
    try:
        with open('styles.css', 'r') as f:
            css_content = f.read()
        print(f"Loaded styles.css ({len(css_content)} bytes)")
    except Exception as e:
        print(f"Error loading styles.css: {e}")
        css_content = ""

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
    # top_club_ids = build_top_spenders(transfers, clubs_enriched, top_n=TOP_N_CLUBS)

    # Enrich transfers with numeric fees and country buckets
    transfers_enriched = build_transfer_enriched(transfers, club_country_map)

    # Get unique seasons sorted chronologically
    all_seasons = sort_seasons_chronologically(transfers_enriched['transfer_season'].dropna().unique().tolist(), '12/13')

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
    money_layout = column(p_money_in, p_money_out, css_classes=['content-card'], stylesheets=[css_content])
    players_layout = column(p_players_in, p_players_out, css_classes=['content-card'], stylesheets=[css_content])
    players_layout.visible = False  # start with Money
    
    season_title = Div(text="<span class='section-title'>Seasons</span>")

    # Season filter widget - shows which seasons are included
    season_select = CheckboxButtonGroup(
        labels=all_seasons,
        active=[i for i, s in enumerate(all_seasons) if s in selected_seasons_display],
        stylesheets=[css_content]
    )
    
    season_widget_group = column(season_title, season_select)

    # Pre-compute per-season aggregated data for ALL clubs
    all_club_ids = clubs_enriched['club_id'].dropna().unique().tolist()
    per_season_data = build_per_season_data(all_club_ids, transfers_enriched, all_seasons)

    # Calculate team statistics (games won, money spent, money earned) for top clubs
    team_stats = calculate_team_statistics(
        club_ids=top_club_ids,
        games=games,
        transfers_enriched=transfers_enriched,
        selected_seasons=selected_seasons_display
    )
    
    # Convert team stats to a dictionary keyed by club_id for easy lookup
    team_stats_dict = team_stats.set_index('club_id').to_dict('index')

    # Create grouped bar chart for team statistics (initially hidden)
    import json

    team_stats_title = Div(text="<h3>Click on a team column to see statistics</h3>", width=1200)

    # Data source for team stats (money + win%, including per competition)
    chart_source = ColumnDataSource(data=dict(
        season=[],
        spent=[],
        earned=[],
        win_pct=[],
        win_pct_league=[],
        win_pct_domestic_cup=[],
        win_pct_international_cup=[],
        games_played=[],
        games_won=[],
        games_played_league=[],
        games_won_league=[],
        games_played_domestic_cup=[],
        games_won_domestic_cup=[],
        games_played_international_cup=[],
        games_won_international_cup=[],
    ))

    # Create figure with dual y-axes for better visualization
    from bokeh.transform import dodge
    from bokeh.models import LinearAxis, Range1d

    p_team_stats = figure(
        x_range=[],
        height=380,
        width=1200,
        title="Team Statistics by Season – Money vs Performance",
        toolbar_location="right",
        tools="hover,save,reset,pan,wheel_zoom,box_zoom"
    )

    # General styling tweaks
    p_team_stats.background_fill_color = "#f9f9f9"
    p_team_stats.border_fill_color = "#ffffff"
    p_team_stats.outline_line_color = None
    p_team_stats.grid.grid_line_color = "#dfe6e9"
    p_team_stats.grid.grid_line_alpha = 0.4
    p_team_stats.axis.axis_line_color = "#2c3e50"
    p_team_stats.axis.major_tick_line_color = "#2c3e50"
    p_team_stats.axis.major_label_text_font_size = "10pt"
    p_team_stats.title.text_font_size = "14pt"

    # Left y-axis: Money (€ Millions)
    p_team_stats.y_range = Range1d(start=0, end=100)
    p_team_stats.yaxis.axis_label = "Money (€ Millions)"
    p_team_stats.yaxis.axis_label_text_color = "#e74c3c"

    # Money bars on left axis
    spent_bars = p_team_stats.vbar(
        x=dodge('season', -0.2, range=p_team_stats.x_range),
        top='spent', width=0.35, source=chart_source,
        color="#e74c3c", legend_label="Money Spent (€M)", alpha=0.85
    )
    earned_bars = p_team_stats.vbar(
        x=dodge('season', 0.2, range=p_team_stats.x_range),
        top='earned', width=0.35, source=chart_source,
        color="#27ae60", legend_label="Money Earned (€M)", alpha=0.85
    )

    # Right y-axis: Win percentage
    p_team_stats.extra_y_ranges = {"win_game_range": Range1d(start=0, end=70)}
    right_axis = LinearAxis(y_range_name="win_game_range", axis_label="Games won")
    right_axis.axis_label_text_color = "#2980b9"
    p_team_stats.add_layout(right_axis, 'right')

    # Win % lines with different colors for each competition
    # Overall – blue
    win_line = p_team_stats.line(
        x='season', y='games_won', source=chart_source,
        color="#2980b9", line_width=3, legend_label="Games won (overall)",
        y_range_name="win_game_range"
    )
    win_circles = p_team_stats.circle(
        x='season', y='games_won', source=chart_source,
        color="#2980b9", size=8, legend_label="Games won (overall)",
        y_range_name="win_game_range"
    )

    # League – green
    win_line_league = p_team_stats.line(
        x='season', y='games_won_league', source=chart_source,
        color="#27ae60", line_width=2, legend_label="Games won league",
        y_range_name="win_game_range"
    )
    win_circles_league = p_team_stats.circle(
        x='season', y='games_won_league', source=chart_source,
        color="#27ae60", size=7, legend_label="Games won league",
        y_range_name="win_game_range"
    )

    # Domestic cup – orange
    win_line_domestic_cup = p_team_stats.line(
        x='season', y='games_won_domestic_cup', source=chart_source,
        color="#e67e22", line_width=2, legend_label="Games won domestic cup",
        y_range_name="win_game_range"
    )
    win_circles_domestic_cup = p_team_stats.circle(
        x='season', y='games_won_domestic_cup', source=chart_source,
        color="#e67e22", size=7, legend_label="Games won domestic cup",
        y_range_name="win_game_range"
    )

    # International cup – purple
    win_line_international_cup = p_team_stats.line(
        x='season', y='games_won_international_cup', source=chart_source,
        color="#8e44ad", line_width=2, legend_label="Games won international cup",
        y_range_name="win_game_range"
    )
    win_circles_international_cup = p_team_stats.circle(
        x='season', y='games_won_international_cup', source=chart_source,
        color="#8e44ad", size=7, legend_label="Games won international cup",
        y_range_name="win_game_range"
    )

    p_team_stats.xaxis.major_label_orientation = 0.8
    p_team_stats.legend.location = "top_left"
    p_team_stats.legend.click_policy = "hide"

    # Competition selector widget – stays close to the plot
    competition_select = CheckboxButtonGroup(
        labels=["Overall", "Domestic league", "Domestic cup", "International cup"],
        active=[0, 1, 2, 3],
        width=450,
        stylesheets=[css_content]
    )

    competition_callback = CustomJS(
        args=dict(
            competition_select=competition_select,
            win_line=win_line,
            win_circles=win_circles,
            win_line_league=win_line_league,
            win_circles_league=win_circles_league,
            win_line_domestic_cup=win_line_domestic_cup,
            win_circles_domestic_cup=win_circles_domestic_cup,
            win_line_international_cup=win_line_international_cup,
            win_circles_international_cup=win_circles_international_cup
        ),
        code="""
            const active = new Set(competition_select.active);
            function setVisible(glyph, flag) { if (glyph) glyph.visible = flag; }
            setVisible(win_line, active.has(0));
            setVisible(win_circles, active.has(0));
            setVisible(win_line_league, active.has(1));
            setVisible(win_circles_league, active.has(1));
            setVisible(win_line_domestic_cup, active.has(2));
            setVisible(win_circles_domestic_cup, active.has(2));
            setVisible(win_line_international_cup, active.has(3));
            setVisible(win_circles_international_cup, active.has(3));
        """
    )
    competition_select.js_on_change('active', competition_callback)

    # Hover tool: money + overall performance
    hover = p_team_stats.select_one(HoverTool)
    hover.tooltips = [
        ("Season", "@season"),
        ("Money Spent", "€@spent{0.0}M"),
        ("Money Earned", "€@earned{0.0}M"),
        ("Games", "@games_played played, @games_won won"),
        ("Games league", "@games_played_league played, @games_won_league won"),
        ("Games domestic cup", "@games_played_domestic_cup played, @games_won_domestic_cup won"),
        ("Games international cup", "@games_played_international_cup played, @games_won_international_cup won"),
    ]
    hover.mode = 'mouse'

    # Put selector directly under the plot
    team_stats_layout = column(
        team_stats_title,
        p_team_stats,
        competition_select,
        css_classes=['content-card'],
        stylesheets=[css_content]
    )
    # Hide until a club is clicked so it doesn't push the heatmaps down as an empty block
    team_stats_layout.visible = False

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

    mode_select = Select(
        title="View mode",
        value="Money",
        options=["Money", "Players"],
        width=150,
        stylesheets=[css_content]
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
        options=["Linear", "Log", "Percentage"],
        width=150,
        stylesheets=[css_content]
    )

    club_mode_select = Select(
        title="Clubs",
        value="Top 50",
        options=["Top 50", "By country"],
        width=150,
        stylesheets=[css_content]
    )

    # Available countries (from clubs_enriched / club_country_map)
    available_countries = sorted(
        {c for c in clubs_enriched['club_country'].unique()
         if c not in ('Unknown', 'Without Club', 'Retired')}
    )

    country_select = Select(
        title="Country",
        value=available_countries[0] if available_countries else "",
        options=available_countries,
        disabled=True,
        width=200,
        stylesheets=[css_content]
    )

    # Make top controls consistent in width
    mode_select.width = 180
    scale_select.width = 200
    club_mode_select.width = 200
    country_select.width = 220



    club_mode_callback = CustomJS(
        args=dict(
            club_mode_select=club_mode_select,
            country_select=country_select
        ),
        code="""
        const mode = club_mode_select.value;
        country_select.disabled = (mode !== 'By country');
        """
    )
    club_mode_select.js_on_change('value', club_mode_callback)

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

    # Click handler for team statistics
    # Prepare data for all clubs per season
    per_season_stats_data = {}
    for club_id in top_club_ids:
        club_stats = calculate_per_season_statistics(
            club_id=club_id,
            games=games,
            competitions=competitions,
            transfers_enriched=transfers_enriched,
            seasons=all_seasons
        )
        per_season_stats_data[int(club_id)] = club_stats.to_dict('records')
    
    per_season_stats_json = json.dumps(per_season_stats_data)
    club_names_json = json.dumps({int(k): v for k, v in club_id_to_name.items() if k in top_club_ids})
    
    # Get data sources for all rectangle renderers
    mi_source = mi_lin_rect.data_source if mi_lin_rect else None
    mo_source = mo_lin_rect.data_source if mo_lin_rect else None
    pi_source = pi_lin_rect.data_source if pi_lin_rect else None
    po_source = po_lin_rect.data_source if po_lin_rect else None
    
    click_callback = CustomJS(
        args=dict(
            chart_source=chart_source,
            team_stats_title=team_stats_title,
            team_stats_layout=team_stats_layout,
            p_team_stats=p_team_stats,
            season_select=season_select,
            all_seasons=all_seasons,
            mi_source=mi_source,
            mo_source=mo_source,
            pi_source=pi_source,
            po_source=po_source
        ),
        code=f"""
        const perSeasonStats = {per_season_stats_json};
        const clubNames = {club_names_json};
        
        console.log('Callback triggered!');
        
        // Get the source - try cb_obj first, then check all sources
        let source = null;
        
        // Check if cb_obj is the source (from js_on_change)
        if (typeof cb_obj !== 'undefined' && cb_obj && cb_obj.selected && cb_obj.selected.indices) {{
            source = cb_obj;
            console.log('Using cb_obj as source');
        }} else {{
            // Check mo_source specifically for selection/deselection
            if (mo_source) {{
                source = mo_source;
                console.log('Using mo_source');
            }}
        }}
        
        if (!source) {{
            console.log('No source found');
            return;
        }}
        
        // Handle deselection - reset the chart
        if (!source.selected.indices || source.selected.indices.length === 0) {{
            console.log('Deselected - resetting chart');
            team_stats_layout.visible = false;
            window.currentTeamStats = null;
            chart_source.data = {{
                season: [],
                spent: [],
                earned: [],
                win_pct: [],
                win_pct_league: [],
                win_pct_domestic_cup: [],
                win_pct_international_cup: [],
                games_played: [],
                games_won: [],
                games_played_league: [],
                games_won_league: [],
                games_played_domestic_cup: [],
                games_won_domestic_cup: [],
                games_played_international_cup: [],
                games_won_international_cup: [],
            }};

            p_team_stats.x_range.factors = [];
            return;
        }}
        
        const idx = source.selected.indices[0];
        const clickedClub = source.data.club[idx];
        
        console.log('Clicked on:', clickedClub);
        
        // Extract club name from "Club Name (Country)" format
        const match = clickedClub.match(/^(.+?) \\((.+?)\\)$/);
        if (!match) {{
            console.log('Could not parse:', clickedClub);
            team_stats_title.text = '<h3 style=\"color: red;\">Error: Could not parse club name</h3>';
            team_stats_layout.visible = true;
            return;
        }}
        
        const clubName = match[1];
        console.log('Club name:', clubName);
        
        // Find club ID
        let clubId = null;
        for (const [id, name] of Object.entries(clubNames)) {{
            if (name === clubName) {{
                clubId = parseInt(id);
                break;
            }}
        }}
        
        console.log('Club ID:', clubId);
        
        if (!clubId || !perSeasonStats[clubId]) {{
            console.log('No stats for club ID:', clubId);
            team_stats_title.text = '<h3 style=\"color: orange;\">' + clubName + ' - No statistics available</h3>';
            team_stats_layout.visible = true;
            return;
        }}
        
        // Get selected seasons
        const selectedIndices = season_select.active;
        const selectedSeasons = selectedIndices.map(i => all_seasons[i]);
        console.log('Selected seasons:', selectedSeasons);
        
        const stats = perSeasonStats[clubId];
        console.log('All stats for club:', stats);
        
        // Filter stats by selected seasons
        const filteredStats = stats.filter(s => selectedSeasons.includes(s.season));
        console.log('Filtered stats:', filteredStats);
        
        if (filteredStats.length === 0) {{
            team_stats_title.text = '<h3>' + clubName + ' - No data for selected seasons</h3>';
            team_stats_layout.visible = true;
            chart_source.data = {{
                season: [],
                spent: [],
                earned: [],
                win_pct: [],
                win_pct_league: [],
                win_pct_domestic_cup: [],
                win_pct_international_cup: [],
                games_played: [],
                games_won: [],
                games_played_league: [],
                games_won_league: [],
                games_played_domestic_cup: [],
                games_won_domestic_cup: [],
                games_played_international_cup: [],
                games_won_international_cup: [],
            }};
            p_team_stats.x_range.factors = [];
            return;
        }}
        
        console.log('Selected seasons:', selectedSeasons);
        
        // Prepare data for chart (convert money to millions, keep win% as is)
        const seasons = filteredStats.map(s => s.season);
        const spent = filteredStats.map(s => s.money_spent / 1000000);
        const earned = filteredStats.map(s => s.money_earned / 1000000);
        const winPct = filteredStats.map(s => s.win_pct);  // Keep as percentage (0-100)
        const gamesPlayed = filteredStats.map(s => s.games_played);
        const gamesWon = filteredStats.map(s => s.games_won);
        const winPctLeague = filteredStats.map(s => s.win_pct_league);
        const winPctDomesticCup = filteredStats.map(s => s.win_pct_domestic_cup);
        const winPctInternationalCup = filteredStats.map(s => s.win_pct_international_cup);
        const gamesWonLeague = filteredStats.map(s => s.games_won_league);
        console.log("aaa1");
        const gamesPlayedLeague = filteredStats.map(s => s.games_played_league);
        console.log("aaa2");
        const gamesWonDomesticCup = filteredStats.map(s => s.games_won_domestic_cup);
        const gamesPlayedDomesticCup = filteredStats.map(s => s.games_played_domestic_cup);
        const gamesWonInternationalCup = filteredStats.map(s => s.games_won_international_cup);
        const gamesPlayedInternationalCup = filteredStats.map(s => s.games_played_international_cup);
        
        console.log('Chart data - seasons:', seasons);
        console.log('Chart data - spent:', spent);
        console.log('Chart data - earned:', earned);
        console.log('Chart data - winPct:', winPct);
        
        // Update chart
        chart_source.data = {{
            season: seasons,
            spent: spent,
            earned: earned,
            win_pct: winPct,
            win_pct_league: winPctLeague,
            win_pct_domestic_cup: winPctDomesticCup,
            win_pct_international_cup: winPctInternationalCup,
            games_played: gamesPlayed,
            games_won: gamesWon,
            games_played_league: gamesPlayedLeague,
            games_won_league: gamesWonLeague,
            games_played_domestic_cup: gamesPlayedDomesticCup,
            games_won_domestic_cup: gamesWonDomesticCup,
            games_played_international_cup: gamesPlayedInternationalCup,
            games_won_international_cup: gamesWonInternationalCup,
        }};

        // Update x_range
        p_team_stats.x_range.factors = seasons;
        
        // Update y_range dynamically based on data
        const maxMoney = Math.max(...spent, ...earned);
        p_team_stats.y_range.end = Math.max(maxMoney * 1.2, 50);  // At least 50M
        
        // Update title with summary
        const totalGames = filteredStats.reduce((sum, s) => sum + s.games_played, 0);
        const totalWins = filteredStats.reduce((sum, s) => sum + s.games_won, 0);
        const avgWinPct = totalGames > 0 ? (totalWins / totalGames * 100).toFixed(1) : 0;
        const totalSpent = spent.reduce((sum, s) => sum + s, 0);
        const totalEarned = earned.reduce((sum, s) => sum + s, 0);
        const netSpend = totalEarned - totalSpent;
        
        team_stats_title.text = '<div class="section-title">' + clubName + ' - Money vs Performance</div>' +
            '<div style="display: flex; gap: 30px; color: #666; font-size: 14px; margin: 10px 0;">' +
            '<div><strong>Games:</strong> ' + totalGames + ' played, ' + totalWins + ' won (' + avgWinPct + '%)</div>' +
            '<div><strong>Spending:</strong> €' + totalSpent.toFixed(1) + 'M</div>' +
            '<div><strong>Income:</strong> €' + totalEarned.toFixed(1) + 'M</div>' +
            '<div><strong>Net:</strong> <span style="color: ' + (netSpend > 0 ? '#2ecc71' : '#e74c3c') + ';">€' + 
            (netSpend > 0 ? '+' : '') + netSpend.toFixed(1) + 'M</span></div>' +
            '</div>';
        
        // Show the layout
        team_stats_layout.visible = true;
        
        // Store current club info globally for filter updates
        window.currentTeamStats = {{
            clubId: clubId,
            clubName: clubName
        }};
        
        console.log('Chart updated successfully!');
        """
    )
    
    # Connect callback only to money_out source (second heatmap)
    if mo_source:
        mo_source.selected.js_on_change('indices', click_callback)
    
    # Also add TapTool callback to money_out plot directly
    from bokeh.models import TapTool
    tap_tool = p_money_out.select_one(TapTool)
    if tap_tool:
        tap_tool.callback = click_callback
    
    # Create a callback to update team stats when filters change
    update_team_stats_callback = CustomJS(
        args=dict(
            chart_source=chart_source,
            team_stats_title=team_stats_title,
            team_stats_layout=team_stats_layout,
            p_team_stats=p_team_stats,
            season_select=season_select,
            all_seasons=all_seasons,
            club_mode_select=club_mode_select,
            country_select=country_select
        ),
        code=f"""
        const perSeasonStats = {per_season_stats_json};
        const clubNames = {club_names_json};
        const topClubIds = {json.dumps([int(cid) for cid in top_club_ids])};
        const clubCountryMap = {json.dumps({int(k): v for k, v in club_country_map.items()})};
        
        // Check if a team is currently selected
        if (!window.currentTeamStats) {{
            return;  // No team selected, nothing to update
        }}
        
        const clubId = window.currentTeamStats.clubId;
        const clubName = window.currentTeamStats.clubName;
        
        console.log('Updating team stats for:', clubName);
        
        // Check if the club is still visible based on current filters
        const clubMode = club_mode_select.value;
        let clubVisible = false;
        
        if (clubMode === 'Top 50') {{
            clubVisible = topClubIds.includes(clubId);
        }} else if (clubMode === 'By country') {{
            const selectedCountry = country_select.value;
            const clubCountry = clubCountryMap[clubId];
            clubVisible = (clubCountry === selectedCountry);
        }}
        
        if (!clubVisible) {{
            console.log('Club not visible in current view, resetting chart');
            team_stats_layout.visible = false;
            window.currentTeamStats = null;
            return;
        }}
        
        // Get selected seasons
        const selectedIndices = season_select.active;
        const selectedSeasons = selectedIndices.map(i => all_seasons[i]);
        const stats = perSeasonStats[clubId];
        
        if (!stats) {{
            console.log('No stats available');
            team_stats_layout.visible = false;
            window.currentTeamStats = null;
            return;
        }}
        
        // Filter stats by selected seasons
        const filteredStats = stats.filter(s => selectedSeasons.includes(s.season));
        
        if (filteredStats.length === 0) {{
            team_stats_title.text = '<h3>' + clubName + ' - No data for selected seasons</h3>';
            chart_source.data = {{
                season: [],
                spent: [],
                earned: [],
                win_pct: [],
                win_pct_league: [],
                win_pct_domestic_cup: [],
                win_pct_international_cup: [],
                games_played: [],
                games_won: [],
                games_played_league: [],
                games_won_league: [],
                games_played_domestic_cup: [],
                games_won_domestic_cup: [],
                games_played_international_cup: [],
                games_won_international_cup: [],
            }};
            p_team_stats.x_range.factors = [];
            team_stats_layout.visible = true;
            return;
        }}
        
        // Prepare data for chart
        const seasons = filteredStats.map(s => s.season);
        const spent = filteredStats.map(s => s.money_spent / 1000000);
        const earned = filteredStats.map(s => s.money_earned / 1000000);
        const winPct = filteredStats.map(s => s.win_pct);
        const gamesPlayed = filteredStats.map(s => s.games_played);
        const gamesWon = filteredStats.map(s => s.games_won);
        const winPctLeague = filteredStats.map(s => s.win_pct_league);
        const winPctDomesticCup = filteredStats.map(s => s.win_pct_domestic_cup);
        const winPctInternationalCup = filteredStats.map(s => s.win_pct_international_cup);
        const gamesWonLeague = filteredStats.map(s => s.games_won_league);
        console.log("aaa3");
        const gamesPlayedLeague = filteredStats.map(s => s.games_played_league);
        console.log("aaa4");
        const gamesWonDomesticCup = filteredStats.map(s => s.games_won_domestic_cup);
        const gamesPlayedDomesticCup = filteredStats.map(s => s.games_played_domestic_cup);
        const gamesWonInternationalCup = filteredStats.map(s => s.games_won_international_cup);
        const gamesPlayedInternationalCup = filteredStats.map(s => s.games_played_international_cup);

        
        // Update chart
        chart_source.data = {{
            season: seasons,
            spent: spent,
            earned: earned,
            win_pct: winPct,
            win_pct_league: winPctLeague,
            win_pct_domestic_cup: winPctDomesticCup,
            win_pct_international_cup: winPctInternationalCup,
            games_played: gamesPlayed,
            games_won: gamesWon,
            games_played_league: gamesPlayedLeague,
            games_won_league: gamesWonLeague,
            games_played_domestic_cup: gamesPlayedDomesticCup,
            games_won_domestic_cup: gamesWonDomesticCup,
            games_played_international_cup: gamesPlayedInternationalCup,
            games_won_international_cup: gamesWonInternationalCup,
        }};

        
        p_team_stats.x_range.factors = seasons;
        
        const maxMoney = Math.max(...spent, ...earned);
        p_team_stats.y_range.end = Math.max(maxMoney * 1.2, 50);
        
        // Update title
        const totalGames = filteredStats.reduce((sum, s) => sum + s.games_played, 0);
        const totalWins = filteredStats.reduce((sum, s) => sum + s.games_won, 0);
        const avgWinPct = totalGames > 0 ? (totalWins / totalGames * 100).toFixed(1) : 0;
        const totalSpent = spent.reduce((sum, s) => sum + s, 0);
        const totalEarned = earned.reduce((sum, s) => sum + s, 0);
        const netSpend = totalEarned - totalSpent;
        
        team_stats_title.text = '<div class="section-title">' + clubName + ' - Money vs Performance</div>' +
            '<div style="display: flex; gap: 30px; color: #666; font-size: 14px; margin: 10px 0;">' +
            '<div><strong>Games:</strong> ' + totalGames + ' played, ' + totalWins + ' won (' + avgWinPct + '%)</div>' +
            '<div><strong>Spending:</strong> €' + totalSpent.toFixed(1) + 'M</div>' +
            '<div><strong>Income:</strong> €' + totalEarned.toFixed(1) + 'M</div>' +
            '<div><strong>Net:</strong> <span style="color: ' + (netSpend > 0 ? '#2ecc71' : '#e74c3c') + ';">€' + 
            (netSpend > 0 ? '+' : '') + netSpend.toFixed(1) + 'M</span></div>' +
            '</div>';
        
        console.log('Team stats updated');
        """
    )
    
    # Register update_team_stats_callback for all filter changes
    mode_select.js_on_change('value', update_team_stats_callback)
    scale_select.js_on_change('value', update_team_stats_callback)
    club_mode_select.js_on_change('value', update_team_stats_callback)
    country_select.js_on_change('value', update_team_stats_callback)

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

    # JS helpers: mappings for clubs
    top_club_ids_int = [int(cid) for cid in top_club_ids]

    club_country_map_js = {int(k): v for k, v in club_country_map.items()}

    club_display_map_js = {
        int(cid): f"{club_id_to_name.get(cid, str(cid))} ({club_country_map.get(cid, 'Unknown')})"
        for cid in clubs_enriched['club_id']
    }

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
            # NEW: figures so we can update x_range factors
            plots=[p_money_in, p_money_out, p_players_in, p_players_out],
            # club_ids=[int(cid) for cid in result['matrices']['money_in'].columns],  # Convert to int!
            all_countries=all_countries,
            # NEW: club-mode-related data
            club_mode_select=club_mode_select,
            country_select=country_select,
            top_club_ids=top_club_ids_int,
            club_country_map_js=club_country_map_js,
            club_display_map_js=club_display_map_js,
            # club_display_names=[f"{club_id_to_name.get(cid, str(cid))} ({club_country_map.get(cid, 'Unknown')})" for cid in result['matrices']['money_in'].columns],
            # NEW: pass colorbars so we can update their color_mappers
            lin_cbs=[mi_lin_cb, mo_lin_cb, pi_lin_cb, po_lin_cb],
            log_cbs=[mi_log_cb, mo_log_cb, pi_log_cb, po_log_cb],
            pct_cbs=[mi_pct_cb, mo_pct_cb, pi_pct_cb, po_pct_cb]
        ),
        code="""
        // Parse the JSON string to get season_data_js
        const season_data_js = JSON.parse(season_data_json_string);
        
        // Get selected seasons (if empty, use all)
        const selectedIndices = season_select.active;
        let selected = selectedIndices.map(i => all_seasons[i]);
        
        if (selected.length === 0) {
            selected = all_seasons;
        }
        
        // Decide which clubs are currently active
        function getActiveClubs() {
            let ids = [];
        
            if (club_mode_select.value === "Top 50") {
                ids = top_club_ids.slice();
        
            } else {
                const target = country_select.value;
                console.log(target);
        
                // If no country is selected yet, don't wipe everything:
                if (!target) {
                    ids = top_club_ids.slice();
                } else {
                    for (const [id, country] of club_country_map_js) {
                        if (country === target) {
                            ids.push(Number(id));
                        }
                    }
        
                    // If, for some reason, nothing matched, fall back to Top 50
                    if (ids.length === 0) {
                        console.warn("No clubs found for country:", target,
                                     "– falling back to Top 50.");
                        ids = top_club_ids.slice();
                    }
                }
            }
        
            // Sort nicely for display
            ids.sort((a, b) => {
                const na = (club_display_map_js[a] || "").toLowerCase();
                const nb = (club_display_map_js[b] || "").toLowerCase();
                if (na < nb) return -1;
                if (na > nb) return 1;
                return a - b;
            });
        
            const labels = ids.map(id => club_display_map_js.get(id) || String(id));
            return { ids, labels };
        }
        
        const active = getActiveClubs();
        const club_ids = active.ids;
        const club_display_names = active.labels;
        
        // Update x_range factors of all plots to match active clubs
        for (let i = 0; i < plots.length; i++) {
            plots[i].x_range.factors = club_display_names;
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

    season_select.js_on_change('active', season_callback)
    season_select.js_on_change('active', update_team_stats_callback)
    club_mode_select.js_on_change('value', season_callback)
    country_select.js_on_change('value', season_callback)
    country_select.js_on_change('value', update_team_stats_callback)

    # Create a debug div that will execute JavaScript to check season_data_js
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

    page_title = Div(
        text="""
        <div class="dashboard-header">
            <h1>Club–Country Transfer Explorer</h1>
            <p>
                Explore transfer spending, income and performance by club and country.
            </p>
        </div>
        """,
        width=1200
    )

    # Top control bar: view, scale, club mode, country
    top_controls = row(
        mode_select,
        scale_select,
        club_mode_select,
        country_select,
        css_classes=['control-panel'],
        stylesheets=[css_content],
        spacing=20
    )

    # Seasons in their own full-width row
    season_row = row(
        season_widget_group,
        css_classes=['control-panel'],
        stylesheets=[css_content],
        spacing=20
    )

    # Inject CSS directly into the page title Div to ensure it's rendered
    page_title = Div(
        text=f"""
        <style>
        {css_content}
        </style>
        <div class="dashboard-header">
            <h1>Club–Country Transfer Explorer</h1>
            <p>
                Explore transfer spending, income and performance by club and country.
            </p>
        </div>
        """,
        width=1200
    )

    layout = column(
        page_title,
        top_controls,
        season_row,
        money_layout,
        players_layout,
        team_stats_layout,
        debug_div,
        sizing_mode="scale_width"
    )

    output_file(OUTPUT_HTML, title="Club-Country Transfer Heatmaps")
    show(layout)

if __name__ == "__main__":
    build_dashboard()
