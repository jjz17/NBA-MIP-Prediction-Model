import numpy as np
import pandas as pd
import os

# url = 'https://raw.githubusercontent.com/jjz17/NBA-MIP-Prediction-Model/main/data/all_player_seasons_outcome.csv'
# season_data = pd.read_csv(url, index_col=0)
season_data = pd.read_csv(f'..{os.path.sep}data{os.path.sep}all_player_seasons_outcome.csv', index_col=0)
# season_data.head()

# url = 'https://raw.githubusercontent.com/jjz17/NBA-MIP-Prediction-Model/main/data/All_MIP.csv'
# mip_data = pd.read_csv(url, index_col=0)
mip_data = pd.read_csv(f'..{os.path.sep}data{os.path.sep}ALL_MIP.csv', index_col=0)
# mip_data.head()

# Replace NA Values with 0
season_data = season_data.fillna(0)
mip_data = mip_data.fillna(0)

# Create list of unique players
players_list = season_data['Player'].unique()

# Create list of numerical features
numerics = season_data.select_dtypes(include=np.number).columns.tolist()
# Age is not relevant
numerics.remove('Age')


# Function to compute change in performance
# Row 1 is the earlier season, Row 2 is the later season

def compute_change(row1, row2):
    diff = row2[numerics] - row1[numerics]
    diff['Season'] += row1['Season']
    diff['Outcome'] = row2['Outcome']
    diff['Player'] = row1['Player']
    return diff


changes_df = pd.DataFrame(columns=numerics)
changes_df['Player'] = []

# Compute changes in performance for each player and add it to the dataset

for player in players_list:

    player_data = season_data[season_data['Player'] == player]

    for i in range(len(player_data) - 1):
        row1 = player_data.iloc[i]
        row2 = player_data.iloc[i + 1]
        if row1['Season'] == row2['Season'] - 1:
            change = compute_change(row1, row2)
            changes_df = changes_df.append(change, ignore_index=True)


# Function to remove asterisks from strings

def replace_asterisk(player):
    return player.replace("*", "")


# Remove asterisks from player names
changes_df['Player'] = changes_df['Player'].map(replace_asterisk)


# Function to perform data insertion

def insert_outcomes(changes_df, mip_data):
    for i in range(len(mip_data)):
        player = mip_data.iloc[i]['Player']
        season = mip_data.iloc[i]['Season']
        shares = mip_data.iloc[i]['Share']

        changes_row = changes_df[changes_df['Player'] == player]
        changes_row = changes_row[changes_row['Season'] == season]

        if (len(changes_row) > 0):
            replace_row = changes_row.index[0]
            changes_df.loc[replace_row, ['Outcome']] = shares

    return changes_df


# Create dataset with continuous Outcome variable

continuous_changes_df = insert_outcomes(changes_df, mip_data)

continuous_changes_df.to_csv(f'..{os.path.sep}data{os.path.sep}wrangled_data.csv', index=False)

print('Done')