# # -*- coding: utf-8 -*-
# """
# Created on Tue Oct 21 10:03:53 2025

# @author: Robbi
# """

import requests
import pandas as pd
import time

# URLs to be read from
BASE_URL = "https://fantasy.premierleague.com/api/"
BOOTSTRAP_URL = BASE_URL + "bootstrap-static/"
GW_URL = BASE_URL + "event/{}/live/"

FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/?future=1"

# Files to be saved to
FILENAMES = ["etypes.csv",
             "team_data.csv",
             "fixture_data.csv",
             "fpl_player_data_with_weekly_points.csv"]

def main():

    # 1️⃣ Fetch all player info
    data = requests.get(BOOTSTRAP_URL).json()
    players = pd.DataFrame(data['elements'])
    
    etypes = pd.DataFrame(data["element_types"])
    print(etypes.head())
    # This dataframe might have columns like “id”, “singular_name”, “plural_name”, etc.
    etypes.to_csv(FILENAMES[0], index=False)
    print(f"✅ Saved as {FILENAMES[0]}")
    
    # Extract teams
    teams = pd.DataFrame(data["teams"])
    print(teams.head())
    teams.to_csv(FILENAMES[1], index=False)
    print(f"✅ Saved as {FILENAMES[1]}")
    
    data2 = requests.get(FIXTURES_URL).json()
    fixtures = pd.DataFrame(data2)
    fixtures.to_csv(FILENAMES[2], index=False)
    print(f"✅ Saved as {FILENAMES[2]}")
    
    # 2️⃣ Collect weekly points for all gameweeks
    weekly_points = pd.DataFrame(index=players['id'])
    for gw in range(1, 39):  # 38 GWs
        r = requests.get(GW_URL.format(gw))
        if r.status_code != 200:
            break
        gw_data = r.json()['elements']
        week_df = pd.DataFrame({
            p['id']: p['stats']['total_points'] for p in gw_data
        }, index=[f'GW{gw}']).T
        weekly_points = weekly_points.join(week_df, how='left')
        time.sleep(0.5)
    
    # 3️⃣ Merge base player info with weekly data
    combined = players.merge(weekly_points, left_on='id', right_index=True)
    
    # 4️⃣ Save to CSV
    combined.to_csv(FILENAMES[3], index=False)
    
    print(f"✅ Saved as {FILENAMES[3]}")


if __name__ == '__main__':
    main()
