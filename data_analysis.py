import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load the data
base_path = 'datasets/transfermarkt/'
clubs = load_data(base_path + 'clubs.csv')
competitions = load_data(base_path + 'competitions.csv')

print("=" * 80)
print("CLUBS WITHOUT MATCHING DOMESTIC_COMPETITION_ID")
print("=" * 80)

# Get all unique competition_ids from competitions table
valid_competition_ids = set(competitions['competition_id'].unique())

# Find clubs with domestic_competition_id that don't match any competition
clubs_without_match = clubs[~clubs['domestic_competition_id'].isin(valid_competition_ids)].copy()

# Also check for clubs with null domestic_competition_id
clubs_with_null = clubs[clubs['domestic_competition_id'].isna()].copy()

print(f"\n📊 Summary:")
print(f"   Total clubs: {len(clubs)}")
print(f"   Valid competition IDs in competitions table: {len(valid_competition_ids)}")
print(f"   Clubs without matching competition: {len(clubs_without_match)}")
print(f"   Clubs with NULL domestic_competition_id: {len(clubs_with_null)}")
print(f"   Total problematic clubs: {len(clubs_without_match) + len(clubs_with_null)}")

# Show clubs without match (non-null but invalid)
if len(clubs_without_match) > 0:
    print(f"\n🔍 Clubs with INVALID domestic_competition_id:")
    print("-" * 80)
    display_cols = ['club_id', 'name', 'domestic_competition_id']
    if 'pretty_name' in clubs_without_match.columns:
        display_cols.insert(2, 'pretty_name')
    
    print(clubs_without_match[display_cols].to_string(index=False))
    
    # Show the invalid competition IDs
    invalid_comp_ids = clubs_without_match['domestic_competition_id'].unique()
    print(f"\n❌ Invalid competition IDs found: {sorted(invalid_comp_ids)}")

# Show clubs with null values
if len(clubs_with_null) > 0:
    print(f"\n🔍 Clubs with NULL domestic_competition_id:")
    print("-" * 80)
    display_cols = ['club_id', 'name']
    if 'pretty_name' in clubs_with_null.columns:
        display_cols.insert(2, 'pretty_name')
    
    print(clubs_with_null[display_cols].head(20).to_string(index=False))
    if len(clubs_with_null) > 20:
        print(f"   ... and {len(clubs_with_null) - 20} more")

# Check which of these problematic clubs are involved in transfers
transfers = load_data(base_path + 'transfers.csv')
problematic_club_ids = set(clubs_without_match['club_id'].tolist() + clubs_with_null['club_id'].tolist())

transfers_from_problematic = transfers[transfers['from_club_id'].isin(problematic_club_ids)]
transfers_to_problematic = transfers[transfers['to_club_id'].isin(problematic_club_ids)]

print(f"\n💸 Transfer Activity:")
print(f"   Transfers FROM problematic clubs: {len(transfers_from_problematic)}")
print(f"   Transfers TO problematic clubs: {len(transfers_to_problematic)}")
print(f"   Total transfers involving these clubs: {len(transfers_from_problematic) + len(transfers_to_problematic)}")

print("\n" + "=" * 80)

# Check clubs in games table
print("\n" + "=" * 80)
print("CLUBS IN GAMES TABLE")
print("=" * 80)

games = load_data(base_path + 'games.csv')

home_club_ids = set(games['home_club_id'].dropna().unique())
away_club_ids = set(games['away_club_id'].dropna().unique())
all_game_clubs = home_club_ids | away_club_ids

print(f"\n📊 Games Table Statistics:")
print(f"   Total games: {len(games)}")
print(f"   Unique clubs as home: {len(home_club_ids)}")
print(f"   Unique clubs as away: {len(away_club_ids)}")
print(f"   Total unique clubs in games: {len(all_game_clubs)}")

# Check overlap with clubs table
all_club_ids = set(clubs['club_id'].unique())
clubs_in_both = all_game_clubs & all_club_ids
clubs_in_games_not_clubs_table = all_game_clubs - all_club_ids
clubs_in_clubs_table_not_games = all_club_ids - all_game_clubs

print(f"\n🔄 Overlap Analysis:")
print(f"   Clubs in BOTH games and clubs table: {len(clubs_in_both)}")
print(f"   Clubs in games but NOT in clubs table: {len(clubs_in_games_not_clubs_table)}")
print(f"   Clubs in clubs table but NOT in games: {len(clubs_in_clubs_table_not_games)}")

if clubs_in_games_not_clubs_table:
    print(f"\n   Sample club IDs in games but missing from clubs table: {sorted(list(clubs_in_games_not_clubs_table))[:10]}")

print("\n" + "=" * 80)

# Check clubs in transfers table
print("\n" + "=" * 80)
print("CLUBS IN TRANSFERS TABLE")
print("=" * 80)

transfer_from_ids = set(transfers['from_club_id'].dropna().unique())
transfer_to_ids = set(transfers['to_club_id'].dropna().unique())
all_transfer_clubs = transfer_from_ids | transfer_to_ids

print(f"\n📊 Transfers Table Statistics:")
print(f"   Total transfers: {len(transfers)}")
print(f"   Unique clubs as 'from': {len(transfer_from_ids)}")
print(f"   Unique clubs as 'to': {len(transfer_to_ids)}")
print(f"   Total unique clubs in transfers: {len(all_transfer_clubs)}")

# Check overlap with clubs table
all_club_ids = set(clubs['club_id'].unique())
clubs_in_both = all_transfer_clubs & all_club_ids
clubs_in_transfers_not_clubs_table = all_transfer_clubs - all_club_ids
clubs_in_clubs_table_not_transfers = all_club_ids - all_transfer_clubs

print(f"\n🔄 Overlap Analysis:")
print(f"   Clubs in BOTH transfers and clubs table: {len(clubs_in_both)}")
print(f"   Clubs in transfers but NOT in clubs table: {len(clubs_in_transfers_not_clubs_table)}")
print(f"   Clubs in clubs table but NOT in transfers: {len(clubs_in_clubs_table_not_transfers)}")

if clubs_in_transfers_not_clubs_table:
   
    # Count how many transfers involve these missing clubs
    transfers_with_missing_from = transfers[transfers['from_club_id'].isin(clubs_in_transfers_not_clubs_table)]
    transfers_with_missing_to = transfers[transfers['to_club_id'].isin(clubs_in_transfers_not_clubs_table)]

    print(f"   Sample club IDs: {sorted(list(clubs_in_transfers_not_clubs_table))[:10]}")

print("\n" + "=" * 80)
