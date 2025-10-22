import pandas as pd
import plotly.express as px

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

base_path = 'datasets/transfermarkt/'
clubs = load_data(base_path + 'clubs.csv')
competitions = load_data(base_path + 'competitions.csv')
transfers = load_data(base_path + 'transfers.csv')

clubs_by_country = pd.merge(clubs, competitions[['competition_id', 'country_name']], left_on='domestic_competition_id', right_on='competition_id', how='left')
england_clubs = clubs_by_country[clubs_by_country['country_name'] == 'England']

premier_league_clubs = clubs[clubs['domestic_competition_id'] == 'GB1']

transfers['transfer_fee'] = pd.to_numeric(transfers['transfer_fee'], errors='coerce').fillna(0)

transfers_15_to_25 = transfers[transfers['transfer_season'].isin(['15/16', '16/17', '17/18', '18/19', '19/20', '20/21', '21/22', '22/23', '23/24', '24/25'])]

spending_by_club = transfers_15_to_25.groupby('to_club_id')['transfer_fee'].sum().reset_index()
spending_by_club = spending_by_club.rename(columns={'transfer_fee': 'total_spending', 'to_club_id': 'club_id'})

premier_league_spending = pd.merge(premier_league_clubs, spending_by_club, on='club_id')

premier_league_spending = premier_league_spending.sort_values(by='total_spending', ascending=False)

# Heatmap of spending by country
pl_transfers = transfers_15_to_25[transfers_15_to_25['to_club_id'].isin(premier_league_clubs['club_id'])]
from_club_countries = clubs_by_country[['club_id', 'country_name']].rename(columns={'country_name': 'from_country'})
pl_transfers_with_country = pd.merge(pl_transfers, from_club_countries, left_on='from_club_id', right_on='club_id', how='left')
pl_transfers_with_country = pd.merge(pl_transfers_with_country, premier_league_clubs[['club_id', 'name']], left_on='to_club_id', right_on='club_id', how='left')

spending_by_country = pl_transfers_with_country.groupby(['name', 'from_country'])['transfer_fee'].sum().reset_index()
spending_pivot = spending_by_country.pivot(index='name', columns='from_country', values='transfer_fee').fillna(0)

fig_heatmap = px.imshow(spending_pivot,
                    labels=dict(x="From Country", y="Premier League Club", color="Transfer Spending"),
                    x=spending_pivot.columns,
                    y=spending_pivot.index,
                    title="Premier League Spending by Country of Player Origin (15-25)"
                   )
fig_heatmap.show()
# fig = px.bar(premier_league_spending, x='name', y='total_spending', title='Premier League Club Transfer Spending 24/25', )
# fig.show()
