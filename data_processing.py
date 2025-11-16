import pandas as pd 
from unidecode import unidecode

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

def standardize_text(text):
    """
    Applies unidecode, converts to lowercase, and strips whitespace.
    """
    if pd.isna(text):
        return text
    return unidecode(str(text)).lower().strip()

def preprocess_additional_clubs(additional_clubs, transfers):
    """
    Retrieve club ids from transfers for additional clubs data.
    """
    
    df_from = transfers[['from_club_id', 'from_club_name']].drop_duplicates().rename(columns={
        'from_club_id': 'club_id',
        'from_club_name': 'club_name'
    })
    df_to = transfers[['to_club_id', 'to_club_name']].drop_duplicates().rename(columns={
        'to_club_id': 'club_id',
        'to_club_name': 'club_name'
    })
    
    club_ids = pd.concat([df_from, df_to]).drop_duplicates(subset=['club_id'])
    
    print(f"Total unique clubs in transfers: {len(club_ids)}")
    print(f"Total additional clubs before merge: {len(additional_clubs)}")
    
    # club_names = set(club_ids['club_name'].dropna().unique())
    # print(club_names)
    
    club_ids['club_name_original'] = club_ids['club_name']
    additional_clubs['club_name_original'] = additional_clubs['club_name']
    
    club_ids['club_name'] = club_ids['club_name'].apply(standardize_text)
    additional_clubs['club_name'] = additional_clubs['club_name'].apply(standardize_text)
    
    merged = additional_clubs.merge(
        club_ids,
        on='club_name',
        how='inner'
    )
    
    merged['club_name'] = merged['club_name_original_y']
    merged = merged.drop(columns=['club_name_original_x', 'club_name_original_y'])
    
    missing_clubs = set(additional_clubs['club_name']) - set(merged['club_name'])
    print(missing_clubs)
                        
    merged = merged.rename(columns={'club_name': 'name'})
    
    print(f"Total merged clubs: {len(merged)}")
        
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