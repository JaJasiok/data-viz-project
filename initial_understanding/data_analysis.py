import pandas as pd
from unidecode import unidecode
import re  # Added for parsing the new file

def standardize_text(text):
    """
    Applies unidecode, converts to lowercase, and strips whitespace.
    """
    if pd.isna(text):
        return text
    # Apply unidecode (e.g., "Köln" -> "Koln"), convert to lowercase, and strip
    return unidecode(str(text)).lower().strip()

def load_data(file_path, **kwargs):
    """
    Loads a CSV file with flexible keyword arguments for pandas.
    """
    try:
        # Pass any extra arguments (like sep='\t') to read_csv
        data = pd.read_csv(file_path, **kwargs)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def load_clubs_report(file_path):
    """
    Parses the unstructured clubs report file.
    """
    print(f"--- Parsing clubs report: {file_path} ---")
    all_parsed_clubs = []
    # Regex to find the country path line
    path_regex = re.compile(r'^\*\*([a-z\/_.-]+)\.txt\*\*')
    # Regex to clean up markdown-linked club names
    club_clean_regex = re.compile(r'\*\*\[([^\]]+)\]\(.*?\)\*\*')
    
    current_country = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line is a new country header
                path_match = path_regex.search(line)
                
                if path_match:
                    # It's a header line
                    path_str = path_match.group(1)
                    # Get country (e.g., 'congo-dr' from 'africa/congo-dr/cd.clubs')
                    current_country = path_str.split('/')[-2]
                    
                    # Clubs might be on the same line after the colon
                    try:
                        clubs_part = line.split(':', 1)[1]
                    except IndexError:
                        clubs_part = '' # No colon, clubs must be on next line
                
                elif current_country:
                    # It's a continuation line (or a line without a path)
                    clubs_part = line
                
                else:
                    # We are before the first country header (e.g., "98 datafiles...")
                    continue

                # Process the clubs_part string
                club_names_raw = clubs_part.split(' · ')
                
                for raw_name in club_names_raw:
                    # Clean the name:
                    # 1. Remove markdown links
                    clean_name = club_clean_regex.sub(r'\1', raw_name.strip())
                    # 2. Remove any lingering list characters and strip whitespace
                    clean_name = clean_name.strip().lstrip('·').strip()
                    
                    if clean_name:
                        all_parsed_clubs.append({
                            'club_name': clean_name, 
                            'country': current_country
                        })
                        
        print(f"   Successfully parsed {len(all_parsed_clubs)} clubs from report.")
        return pd.DataFrame(all_parsed_clubs)

    except Exception as e:
        print(f"Error loading clubs report from {file_path}: {e}")
        return None

# --- 1. LOAD ALL DATA ---
print("--- Loading all data files ---")
base_path = 'datasets/transfermarkt/'

clubs = load_data(base_path + 'clubs.csv')
# competitions = load_data(base_path + 'competitions.csv') # <-- OMITTED as requested
transfers = load_data(base_path + 'transfers.csv')

# Load the new ranking file
ranking = load_data('datasets/rankings/ranking.csv', sep=',')

# Load the new clubs report file
clubs_report = load_clubs_report('datasets/clubs/SUMMARY.md')

print("--- All files loaded ---\n")


# --- 2. STANDARDIZE ALL TEXT COLUMNS ---
print("\n" + "=" * 80)
print("STANDARDIZING TEXT (unidecode, lowercase, strip)")
print("=" * 80)

if clubs is not None:
    clubs['name'] = clubs['name'].apply(standardize_text)
    if 'pretty_name' in clubs.columns:
        clubs['pretty_name'] = clubs['pretty_name'].apply(standardize_text)
    print("   Standardized 'clubs' table: name, pretty_name")

# 'competitions' standardization omitted
    
if transfers is not None:
    transfers['from_club_name'] = transfers['from_club_name'].apply(standardize_text)
    transfers['to_club_name'] = transfers['to_club_name'].apply(standardize_text)
    transfers['player_name'] = transfers['player_name'].apply(standardize_text)
    print("   Standardized 'transfers' table: from_club_name, to_club_name, player_name")

if ranking is not None:
    # Clean column names (e.g., "club name " -> "club name")
    ranking.columns = ranking.columns.str.strip()
    if 'club name' in ranking.columns:
        ranking['club name'] = ranking['club name'].apply(standardize_text)
        print("   Standardized 'ranking' table: club name")
    else:
        print("   WARNING: 'club name' column not found in ranking.csv")
else:
    print("   INFO: ranking.csv not loaded, skipping standardization.")

if clubs_report is not None:
    # Create a new standardized column, keeping the original for the CSV
    clubs_report['standardized_name'] = clubs_report['club_name'].apply(standardize_text)
    print("   Standardized 'clubs_report' table: created 'standardized_name'")
else:
    print("   INFO: clubs_report.txt not loaded, skipping standardization.")

print("=" * 80)


# --- 3. BUILD MASTER KNOWLEDGE BASE ---
print("\n" + "=" * 80)
print("BUILDING MASTER KNOWLEDGE BASE")
print("=" * 80)

known_club_ids = set()
master_known_names = set()
known_names_from_clubs = set()
ranking_names = set()
report_names = set()

if clubs is not None:
    known_club_ids = set(clubs['club_id'].dropna().unique())
    known_names_from_clubs = set(clubs['name'].dropna().unique())
    master_known_names.update(known_names_from_clubs)
    print(f"   - Added {len(known_club_ids)} IDs and {len(known_names_from_clubs)} names from clubs.csv")
    print("     💡 BOOST: This is our 'main' file. It provides the core Club ID, domestic competition link, and official name.")
else:
    print("   - clubs.csv not loaded.")

if ranking is not None and 'club name' in ranking.columns:
    ranking_names = set(ranking['club name'].dropna().unique())
    master_known_names.update(ranking_names)
    print(f"   - Added {len(ranking_names)} names from ranking.csv")
    print("     💡 BOOST: This 'ranking' file adds clubs (by name) that might be missing from our main file, along with ranking data.")
else:
     print("   - ranking.csv not loaded or 'club name' column missing.")

if clubs_report is not None:
    report_names = set(clubs_report['standardized_name'].dropna().unique())
    master_known_names.update(report_names)
    print(f"   - Added {len(report_names)} names from SUMMARY.md")
    print("     💡 BOOST: This 'report' file discovers new clubs from unstructured text, providing names and country origins not found elsewhere.")
else:
    print("   - clubs_report (SUMMARY.MD) not loaded.")

print(f"\n   ➡️ Total Unique Known IDs: {len(known_club_ids)}")
print(f"   ➡️ Total Unique Known Names: {len(master_known_names)}")
print("=" * 80)


# --- 4. NEW: ANALYSIS: DATA BOOST COMPARISON ---
print("\n" + "=" * 80)
print("ANALYSIS: WHICH FILE ADDS MORE NEW CLUBS?")
print("=" * 80)

if (ranking is not None) and (clubs_report is not None) and (clubs is not None):
    # Find new clubs from ranking.csv (names in ranking but not in main clubs.csv)
    new_from_ranking = ranking_names - known_names_from_clubs
    count_ranking = len(new_from_ranking)
    
    # Find new clubs from SUMMARY.md (names in report but not in main clubs.csv)
    new_from_report = report_names - known_names_from_clubs
    count_report = len(new_from_report)
    
    print(f"   New clubs found only in ranking.csv: {count_ranking}")
    print(f"   New clubs found only in SUMMARY.md: {count_report}")
    
    print("\n" + "-" * 80)
    if count_report > count_ranking:
        print(f"   🏆 SUMMARY.md provides {count_report - count_ranking} more new clubs than ranking.csv.")
    elif count_ranking > count_report:
        print(f"   🏆 ranking.csv provides {count_ranking - count_report} more new clubs than SUMMARY.md.")
    else:
        print("   ⚖️ Both files provide the same number of new clubs.")
    print("-" * 80)
    
else:
    print("⚠️ Skipped: This comparison requires clubs.csv, ranking.csv, and SUMMARY.md to be loaded.")

print("\n" + "=" * 80)


# --- SECTION 4: CLUBS WITHOUT MATCHING DOMESTIC_COMPETITION_ID ---
# (OMITTED as requested)


# --- 5. ANALYSIS: CLUBS IN TRANSFERS TABLE (ID-to-ID Check) ---
print("\n" + "=" * 80)
print("CLUBS IN TRANSFERS TABLE (ID-to-ID Check)")
print("=" * 80)
if transfers is not None and clubs is not None:
    transfer_from_ids = set(transfers['from_club_id'].dropna().unique())
    transfer_to_ids = set(transfers['to_club_id'].dropna().unique())
    all_transfer_clubs = transfer_from_ids | transfer_to_ids
    print(f"\n📊 Transfers Table Statistics:")
    print(f"   Total transfers: {len(transfers)}")
    print(f"   Total unique clubs in transfers (by ID): {len(all_transfer_clubs)}")
    all_club_ids = set(clubs['club_id'].unique())
    clubs_in_both = all_transfer_clubs & all_club_ids
    clubs_in_transfers_not_clubs_table = all_transfer_clubs - all_club_ids
    clubs_in_clubs_table_not_transfers = all_club_ids - all_club_ids
    print(f"\n🔄 Overlap Analysis (ID-to-ID):")
    print(f"   Clubs in BOTH transfers and clubs table: {len(clubs_in_both)}")
    print(f"   Clubs in transfers but NOT in clubs table: {len(clubs_in_transfers_not_clubs_table)}")
    print(f"   Clubs in clubs table but NOT in transfers: {len(clubs_in_clubs_table_not_transfers)}")
else:
    print("⚠️ Skipped: transfers.csv or clubs.csv not loaded.")
print("\n" + "=" * 80)


# --- 6. ANALYSIS: CLUBS FROM RANKING FILE IN TRANSFERS (NOT IN CLUBS TABLE) ---
print("\n" + "=" * 80)
print("CLUBS FROM RANKING FILE IN TRANSFERS (NOT IN CLUBS TABLE)")
print("=" * 80)
if ranking is not None and clubs is not None and transfers is not None and 'club name' in ranking.columns:
    main_club_names = set(clubs['name'].dropna().unique())
    print(f"   Total standardized unique names in clubs table: {len(main_club_names)}")
    ranking_club_names = set(ranking['club name'].dropna().unique())
    print(f"   Total standardized unique names in ranking table: {len(ranking_club_names)}")
    new_clubs_to_check = ranking_club_names - main_club_names
    print(f"   Names in ranking but NOT in clubs table: {len(new_clubs_to_check)}")
    transfer_from_names = set(transfers['from_club_name'].dropna().unique())
    transfer_to_names = set(transfers['to_club_name'].dropna().unique())
    all_transfer_names = transfer_from_names | transfer_to_names
    print(f"   Total standardized unique names in transfers table: {len(all_transfer_names)}")
    found_in_transfers = new_clubs_to_check & all_transfer_names
    print(f"\n✅ Found {len(found_in_transfers)} clubs from the ranking file that are NOT in clubs.csv but DO appear in transfers.csv:")
    # ... (rest of the print statements from this section) ...
else:
    print("\n⚠️ Skipping ranking file check (required files not loaded or 'club name' column missing).")
print("\n" + "=" * 80)


# --- 7. CREATE CSV FROM CLUBS REPORT ---
print("\n" + "=" * 80)
print("CREATE CSV FROM CLUBS REPORT")
print("=" * 80)

if clubs_report is not None:
    # We use the original 'club_name' and 'country' columns for the export
    output_df = clubs_report[['club_name', 'country']].copy()
    
    # NEW: Format country column as requested (replace hyphen, apply title case)
    output_df['country'] = output_df['country'].str.replace('-', ' ').str.title()
    
    # NEW: Save to current directory
    output_path = 'parsed_clubs_report.csv'
    
    try:
        output_df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully saved parsed clubs to: {output_path}")
        print(f"   Total clubs saved: {len(output_df)}")
    except Exception as e:
        print(f"\n❌ Error saving parsed clubs CSV: {e}")
else:
    print("\n⚠️ Skipped: clubs_report.txt was not loaded or failed to parse.")
print("\n" + "=" * 80)


# --- 8. ANALYSIS: NEW CLUBS FROM REPORT FILE ---
print("\n" + "=" * 80)
print("NEW CLUBS FROM REPORT FILE (NOT IN CLUBS TABLE)")
print("=" * 80)
if clubs_report is not None and clubs is not None:
    # Use the standardized 'name' column from clubs
    main_club_names = set(clubs['name'].dropna().unique())
    
    # Use the new 'standardized_name' column from the report
    report_club_names = set(clubs_report['standardized_name'].dropna().unique())
    
    # Find names in the report that are not in the main clubs table
    new_clubs_from_report = report_club_names - main_club_names
    
    print(f"\n📊 Analysis:")
    print(f"   Total standardized unique names in clubs table: {len(main_club_names)}")
    print(f"   Total standardized unique names in report file: {len(report_club_names)}")
    
    print(f"\n✅ Found {len(new_clubs_from_report)} new club names in the report file that are NOT in clubs.csv:")
    
    if len(new_clubs_from_report) > 0:
        print("-" * 80)
        # OMITTED list of club names as requested
        print(f"   (List of {len(new_clubs_from_report)} club names omitted.)")
            
else:
    print("\n⚠️ Skipped: clubs_report.txt or clubs.csv not loaded.")
print("\n" + "=" * 80)


# --- 9. FINAL SUMMARY: TOTAL UNIQUE CLUB COVERAGE ---
print("\n" + "=" * 80)
print("⭐ FINAL SUMMARY: TOTAL UNIQUE CLUB COVERAGE")
print("=" * 80)

# Check that all required files are loaded
if transfers is not None and (len(known_club_ids) > 0 or len(master_known_names) > 0):

    # 1. Create the TARGET list of all unique clubs in the transfers table
    # A "club" in the transfers table is a unique (ID, Name) pair
    print("   1. Identifying all unique club entries in transfers.csv...")
    
    from_clubs = transfers[['from_club_id', 'from_club_name']].rename(
        columns={'from_club_id': 'id', 'from_club_name': 'name'})
    to_clubs = transfers[['to_club_id', 'to_club_name']].rename(
        columns={'to_club_id': 'id', 'to_club_name': 'name'})
    
    # This DF represents every unique (id, name) combination found in transfers
    all_transfer_club_pairs = pd.concat([from_clubs, to_clubs]).drop_duplicates().dropna(how='all')
    
    total_unique_transfer_pairs = len(all_transfer_club_pairs)
    print(f"     - Found {total_unique_transfer_pairs} unique (ID, Name) pairs in the transfers table.")

    # 2. Check coverage: See how many (ID, Name) pairs we can find
    print("\n   2. Calculating coverage...")
    
    def check_if_found(row):
        # A club is "found" if:
        # 1. Its ID exists in our known ID list
        # OR
        # 2. Its Name exists in our master known name list
        id_found = pd.notna(row['id']) and row['id'] in known_club_ids
        name_found = pd.notna(row['name']) and row['name'] in master_known_names
        return id_found or name_found

    # Apply the check to every row
    all_transfer_club_pairs['is_found'] = all_transfer_club_pairs.apply(check_if_found, axis=1)
    
    found_count = all_transfer_club_pairs['is_found'].sum()
    
    # 3. Report the final result
    if total_unique_transfer_pairs > 0:
        coverage_pct = (found_count / total_unique_transfer_pairs) * 100
        
        print("\n" + "-" * 80)
        print("   UNIQUE CLUB COVERAGE RESULT")
        print(f"   Of the {total_unique_transfer_pairs} unique club entries in the transfers table:")
        print(f"   - ✅ We found info for: {found_count}")
        print(f"   - ❌ We are missing info for: {total_unique_transfer_pairs - found_count}")
        print(f"\n   - ➡️ This is a {coverage_pct:.2f}% TOTAL coverage rate.")
        print("-" * 80)
    else:
        print("   No clubs found in the transfers table to analyze.")

else:
    print("⚠️ Skipped: Total coverage check requires 'transfers.csv' and at least one loaded info file.")
    print("   Please check for errors in the 'LOAD ALL DATA' and 'BUILD MASTER KNOWLEDGE BASE' sections.")

print("\n" + "=" * 80)


# --- 10. FINAL SUMMARY: TOTAL TRANSFER ENTRY COVERAGE ---
print("\n" + "=" * 80)
print("⭐ FINAL SUMMARY: TOTAL TRANSFER ENTRY COVERAGE (PER ROW)")
print("=" * 80)

if transfers is not None and (len(known_club_ids) > 0 or len(master_known_names) > 0):
    # Create a copy to avoid SettingWithCopyWarning
    transfers_checked = transfers.copy()
    
    # 1. Check if 'from' club is found
    transfers_checked['from_club_found'] = (
        transfers_checked['from_club_id'].isin(known_club_ids) |
        transfers_checked['from_club_name'].isin(master_known_names)
    )
    
    # 2. Check if 'to' club is found
    transfers_checked['to_club_found'] = (
        transfers_checked['to_club_id'].isin(known_club_ids) |
        transfers_checked['to_club_name'].isin(master_known_names)
    )

    # 3. Calculate metrics
    total_entries = len(transfers_checked)
    both_found = transfers_checked['from_club_found'] & transfers_checked['to_club_found']
    at_least_one_found = transfers_checked['from_club_found'] | transfers_checked['to_club_found']
    none_found = ~at_least_one_found
    
    count_both = both_found.sum()
    count_one = at_least_one_found.sum()
    count_none = none_found.sum()
    
    if total_entries > 0:
        pct_both = (count_both / total_entries) * 100
        pct_one = (count_one / total_entries) * 100
        pct_none = (count_none / total_entries) * 100

        print(f"   Analyzed {total_entries} total transfer entries (rows):")
        print("\n" + "-" * 80)
        print("   Fully Explained Transfers (Both clubs found):")
        print(f"   - ✅ Count: {count_both} entries ({pct_both:.2f}%)")
        print("\n   Partially Explained Transfers (At least one club found):")
        print(f"   - ✳️ Count: {count_one} entries ({pct_one:.2f}%)")
        print("\n   Unexplained Transfers (Neither club found):")
        print(f"   - ❌ Count: {count_none} entries ({pct_none:.2f}%)")
        print("-" * 80)
    else:
        print("   No transfer entries found to analyze.")
else:
    print("⚠️ Skipped: This analysis requires 'transfers.csv' and the Master Knowledge Base.")
    print("   Please check for errors in the 'LOAD ALL DATA' and 'BUILD MASTER KNOWLEDGE BASE' sections.")

print("\n" + "=" * 80)
print("--- SCRIPT FINISHED ---")