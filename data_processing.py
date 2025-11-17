import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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

    # Never used from this down
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


def calculate_team_statistics(club_ids, games, transfers_enriched, selected_seasons=None):
    """
    Calculate statistics for each club in the provided list:
    - Total games played
    - Games won
    - Win percentage
    - Total money spent (incoming transfers)
    - Total money earned (outgoing transfers)
    
    Args:
        club_ids: List of club IDs to calculate stats for
        games: DataFrame with game data
        transfers_enriched: DataFrame with transfer data (including fee_eur)
        selected_seasons: List of seasons to filter by in 'YY/YY' format (None = all seasons)
    
    Returns:
        DataFrame with columns: club_id, games_played, games_won, win_pct, 
                                money_spent, money_earned
    """
    results = []
    
    # Convert season format from 'YY/YY' to year for games filtering
    # E.g., '20/21' -> 2020
    games_years = None
    if selected_seasons and len(selected_seasons) > 0:
        games_years = []
        for season in selected_seasons:
            try:
                year_str = season.split('/')[0]
                year = int(year_str)
                # Handle century transition (90-99 = 1990s, 00-89 = 2000s+)
                if year >= 90:
                    games_years.append(1900 + year)
                else:
                    games_years.append(2000 + year)
            except (ValueError, IndexError):
                pass
    
    # Filter games by year if specified
    games_filtered = games.copy()
    if games_years:
        games_filtered = games_filtered[games_filtered['season'].isin(games_years)]
    
    # Filter transfers by season if specified
    transfers_filtered = transfers_enriched.copy()
    if selected_seasons and len(selected_seasons) > 0:
        transfers_filtered = filter_transfers_by_seasons(transfers_filtered, selected_seasons)
    
    for club_id in club_ids:
        # Calculate games statistics
        home_games = games_filtered[games_filtered['home_club_id'] == club_id]
        away_games = games_filtered[games_filtered['away_club_id'] == club_id]
        
        # Count wins
        home_wins = len(home_games[home_games['home_club_goals'] > home_games['away_club_goals']])
        away_wins = len(away_games[away_games['away_club_goals'] > away_games['home_club_goals']])
        total_wins = home_wins + away_wins
        
        # Total games
        total_games = len(home_games) + len(away_games)
        
        # Win percentage
        win_pct = (total_wins / total_games * 100) if total_games > 0 else 0
        
        # Calculate transfer spending (money IN - buying players)
        money_spent = transfers_filtered[
            transfers_filtered['to_club_id'] == club_id
        ]['fee_eur'].sum()
        
        # Calculate transfer earnings (money OUT - selling players)
        money_earned = transfers_filtered[
            transfers_filtered['from_club_id'] == club_id
        ]['fee_eur'].sum()
        
        results.append({
            'club_id': club_id,
            'games_played': total_games,
            'games_won': total_wins,
            'win_pct': round(win_pct, 2),
            'money_spent': money_spent,
            'money_earned': money_earned
        })
    
    return pd.DataFrame(results)


def calculate_per_season_statistics(club_id, games, transfers_enriched, seasons):
    """
    Calculate statistics for a single club broken down by season.
    
    Args:
        club_id: The club ID to calculate stats for
        games: DataFrame with game data
        transfers_enriched: DataFrame with transfer data (including fee_eur)
        seasons: List of seasons in 'YY/YY' format to include
    
    Returns:
        DataFrame with columns: season, games_played, games_won, win_pct, 
                                money_spent, money_earned
    """
    results = []
    
    for season in seasons:
        # Convert season format from 'YY/YY' to year for games filtering
        try:
            year_str = season.split('/')[0]
            year = int(year_str)
            if year >= 90:
                game_year = 1900 + year
            else:
                game_year = 2000 + year
        except (ValueError, IndexError):
            continue
        
        # Filter games for this season
        season_games = games[games['season'] == game_year]
        home_games = season_games[season_games['home_club_id'] == club_id]
        away_games = season_games[season_games['away_club_id'] == club_id]
        
        # Count wins
        home_wins = len(home_games[home_games['home_club_goals'] > home_games['away_club_goals']])
        away_wins = len(away_games[away_games['away_club_goals'] > away_games['home_club_goals']])
        total_wins = home_wins + away_wins
        
        # Total games
        total_games = len(home_games) + len(away_games)
        
        # Win percentage
        win_pct = (total_wins / total_games * 100) if total_games > 0 else 0
        
        # Filter transfers for this season
        season_transfers = transfers_enriched[transfers_enriched['transfer_season'] == season]
        
        # Calculate transfer spending (money IN - buying players)
        money_spent = season_transfers[
            season_transfers['to_club_id'] == club_id
        ]['fee_eur'].sum()
        
        # Calculate transfer earnings (money OUT - selling players)
        money_earned = season_transfers[
            season_transfers['from_club_id'] == club_id
        ]['fee_eur'].sum()
        
        results.append({
            'season': season,
            'games_played': total_games,
            'games_won': total_wins,
            'win_pct': round(win_pct, 2),
            'money_spent': money_spent,
            'money_earned': money_earned
        })
    
    return pd.DataFrame(results)