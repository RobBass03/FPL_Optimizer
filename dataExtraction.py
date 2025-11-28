# # -*- coding: utf-8 -*-
# """
# Created on Tue Oct 21 10:03:53 2025

# @author: Robbi
# """

import requests
import pandas as pd
import time

# Base URLs
base_url = "https://fantasy.premierleague.com/api/"
bootstrap_url = base_url + "bootstrap-static/"
gw_url = base_url + "event/{}/live/"

# 1️⃣ Fetch all player info
data = requests.get(bootstrap_url).json()
players = pd.DataFrame(data['elements'])

etypes = pd.DataFrame(data["element_types"])
print(etypes.head())
# This dataframe might have columns like “id”, “singular_name”, “plural_name”, etc.
etypes.to_csv("etypes.csv", index=False)

# Extract teams
teams = pd.DataFrame(data["teams"])
print(teams.head())
teams.to_csv("team_data.csv", index=False)

fixtures_url = "https://fantasy.premierleague.com/api/fixtures/?future=1"
data2 = requests.get(fixtures_url).json()
fixtures = pd.DataFrame(data2)
fixtures.to_csv("fixture_data.csv", index=False)

# 2️⃣ Collect weekly points for all gameweeks
weekly_points = pd.DataFrame(index=players['id'])
for gw in range(1, 39):  # 38 GWs
    r = requests.get(gw_url.format(gw))
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
combined.to_csv("fpl_player_data_with_weekly_points.csv", index=False)

print("✅ Saved as fpl_player_data_with_weekly_points.csv")

# import requests
# import pandas as pd
# import numpy as np
# import time

# # --- Pandas display options ---
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 20)
# pd.set_option('display.width', 140)
# pd.set_option('display.max_colwidth', 15)

# # --- Helper: linear weights for historic points ---
# def linear_weights(n, min_w=0.5, max_w=1.0):
#     if n <= 1:
#         return np.array([max_w])
#     return np.linspace(min_w, max_w, n)

# # --- 1️⃣ Fetch full FPL dataset ---
# bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
# bootstrap = requests.get(bootstrap_url).json()

# players = pd.DataFrame(bootstrap['elements'])
# teams = pd.DataFrame(bootstrap['teams'])
# positions = pd.DataFrame(bootstrap['element_types'])

# # --- 2️⃣ Merge team and position info ---
# df = players.merge(
#     teams[['id','name','short_name']],
#     left_on='team', right_on='id', suffixes=('','_team')
# )
# df = df.merge(
#     positions[['id','singular_name','singular_name_short']],
#     left_on='element_type', right_on='id', suffixes=('','_position')
# )
# df = df.rename(columns={
#     'name': 'team_name',
#     'short_name': 'team_short',
#     'singular_name': 'position',
#     'singular_name_short': 'position_short'
# })
# df = df.drop(columns=['id_team','id_position'], errors='ignore')
# df['price_million'] = df['now_cost'] / 10.0

# # Keep ep_next and chance_of_playing_next_round
# df['ep_next'] = df['ep_next']
# df['chance_of_playing_next_round'] = df['chance_of_playing_next_round']

# # --- 3️⃣ Filter for high-scoring players ---
# df = df[df['total_points'] >= 32]

# # --- 4️⃣ Gaussian price penalty (position-specific) ---
# def compute_price_penalty(df):
#     df = df.copy()
#     # Position-specific median & sigma multiplier
#     position_params = {
#         'Forward':    {'median': 7.5, 'sigma_mult': 1.5},
#         'Midfielder': {'median': 6.5, 'sigma_mult': 1.5},
#         'Defender':   {'median': 5.5, 'sigma_mult': 1.5}
#     }
#     pos = df['position'].iloc[0]
#     params = position_params.get(pos, {'median': df['price_million'].mean(), 'sigma_mult': 1.5})

#     group_mean = df['price_million'].mean()
#     group_std  = df['price_million'].std()
#     print(f"{pos}: mean price = {group_mean:.2f}, std = {group_std:.2f}")

#     sigma = group_std * params['sigma_mult']
#     median_price = params['median']
#     df['penalty'] = np.exp(-((df['price_million'] - median_price)**2) / (2*sigma**2))
#     return df

# df = df.groupby('position', group_keys=False).apply(compute_price_penalty)

# # --- 5️⃣ Fetch historic weekly points ---
# history_rows = []
# for idx, row in df.iterrows():
#     pid = row['id']
#     summary_url = f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
#     try:
#         s = requests.get(summary_url).json()
#     except:
#         continue
#     hist = s.get('history', [])
#     if not hist:
#         continue
#     for h in hist:
#         pts = h.get('total_points', h.get('points',0))
#         history_rows.append({
#             'element': pid,
#             'web_name': row['web_name'],
#             'position': row['position'],
#             'price_penalty': row['penalty'],
#             'round': int(h['round']),
#             'gw_points': float(pts)
#         })
#     time.sleep(0.05)

# history_df = pd.DataFrame(history_rows)

# # --- 6️⃣ Apply linear weights to historic points ---
# def assign_weights(subdf, min_w=0.7, max_w=1.0):
#     subdf = subdf.sort_values('round').copy()
#     n = len(subdf)
#     w = linear_weights(n, min_w=min_w, max_w=max_w)
#     subdf['week_weight'] = w
#     # Combined: weekly points × price_penalty × week_weight
#     subdf['weighted_points'] = subdf['gw_points'] * subdf['price_penalty'] * subdf['week_weight']
#     return subdf

# history_weighted = history_df.groupby('element', group_keys=False).apply(assign_weights)

# # --- 7️⃣ Aggregate combined weighted points per player ---
# agg = history_weighted.groupby(['element','web_name','position']).agg(
#     weeks_count=('round','count'),
#     combined_weighted_points_sum=('weighted_points','sum'),
#     sum_week_weights=('week_weight','sum')
# ).reset_index()

# agg['combined_weighted_points_avg'] = agg['combined_weighted_points_sum'] / agg['sum_week_weights']

# # Merge back player metadata including ep_next and chance_of_playing_next_round
# agg = agg.merge(
#     df[['id','web_name','team_name','team_short','price_million',
#         'total_points','penalty','ep_next','chance_of_playing_next_round']],
#     left_on='element', right_on='id', how='left'
# )
# print('web_name' in agg.columns)
# print(agg.columns)
# # --- 8️⃣ Display top players per position ---
# for pos in ['Forward','Midfielder','Defender']:
#     print(f"\nTop {pos}s by combined weighted points:")
#     display_cols = ['web_name_y','team_short','price_million','total_points','penalty',
#                     'combined_weighted_points_avg','ep_next','chance_of_playing_next_round']
#     top = agg[agg['position']==pos].sort_values('combined_weighted_points_avg', ascending=False)
#     print(top[display_cols].head(10))

# # --- 9️⃣ Save CSV ---
# agg.to_csv("fpl_combined_weighted_points.csv", index=False)
# print("\n✅ Saved 'fpl_combined_weighted_points.csv'")


# # import requests
# # import pandas as pd
# # import numpy as np
# # import time

# # pd.set_option('display.max_columns', None)   # Show all columns
# # pd.set_option('display.max_rows', 20)       # Limit to 100 rows
# # pd.set_option('display.width', 100)            # Auto-fit wide tables
# # pd.set_option('display.max_colwidth', 15)  # Don't truncate long strings

# # # --- helper: create linear weights from min_w to max_w for n weeks ---
# # def linear_weights(n, min_w=0.5, max_w=1.0):
# #     if n <= 1:
# #         return np.array([max_w])  # single-week case
# #     return np.linspace(min_w, max_w, n)  # earliest -> min_w, latest -> max_w

# # # --- 1️⃣  Fetch the full dataset from FPL API ---
# # url = "https://fantasy.premierleague.com/api/bootstrap-static/"
# # data = requests.get(url).json()

# # players = pd.DataFrame(data['elements'])
# # teams = pd.DataFrame(data['teams'])
# # positions = pd.DataFrame(data['element_types'])

# # # --- 2️⃣  Merge team and position information ---

# # # Merge in team full and short names
# # df = players.merge(
# #     teams[['id', 'name', 'short_name']],
# #     left_on='team', right_on='id',
# #     suffixes=('', '_team')
# # )

# # # Merge in position names and abbreviations
# # df = df.merge(
# #     positions[['id', 'singular_name', 'singular_name_short']],
# #     left_on='element_type', right_on='id',
# #     suffixes=('', '_position')
# # )

# # # Rename columns for clarity
# # df = df.rename(columns={
# #     'name': 'team_name',
# #     'short_name': 'team_short',
# #     'singular_name': 'position',
# #     'singular_name_short': 'position_short'
# # })

# # # Drop duplicate numeric ID columns (cleanup)
# # df = df.drop(columns=['id_team', 'id_position'], errors='ignore')

# # # --- 3️⃣  Add readable price column ---
# # df['price_million'] = df['now_cost'] / 10.0  # convert to £ millions

# # # --- 4️⃣  Filter for players with 16+ total points ---
# # df = df[df['total_points'] >= 32]
# # # print(df['position'])
# # df_FWD = (df[df['position'] == 'Forward']).copy()
# # df_MID = (df[df['position'] == 'Midfielder']).copy()
# # df_DEF = (df[df['position'] == 'Defender']).copy()

# # def apply_price_penalty(df):
# #     df = df.copy()
#     # median_price = df['price_million'].mean()
#     # sigma = df['price_million'].std() *1.5
# #     df['penalty'] = np.exp(-((df['price_million'] - median_price) ** 2) / (2 * sigma**2))
# #     df['points_normalised'] = df['total_points'] * df['penalty']
# #     return df


# # df_FWD = apply_price_penalty(df[df['position'] == 'Forward'])
# # df_MID = apply_price_penalty(df[df['position'] == 'Midfielder'])
# # df_DEF = apply_price_penalty(df[df['position'] == 'Defender'])


# # # --- 5️⃣  Select key columns ---
# # columns_to_keep = [
# #     'id', 'first_name', 'second_name', 'web_name',
# #     'team_name', 'team_short',
# #     'position', 'position_short',
# #     'price_million', 'total_points',
# #     'minutes', 'goals_scored', 'assists',
# #     'clean_sheets', 'value_form', 'selected_by_percent', 'status'
# # ]
# # # df_out = df[columns_to_keep]
# # print('Forwards: ')
# # print(df_FWD.sort_values("points_normalised", ascending=False)[['points_normalised', 'first_name', 'second_name', 'price_million', 'team_short']].head(10))
# # print('Midfielders: ')
# # print(df_MID.sort_values("points_normalised", ascending=False)[['points_normalised', 'first_name', 'second_name', 'price_million', 'team_short']].head(10))
# # print('Defenders: ')
# # print(df_DEF.sort_values("points_normalised", ascending=False)[['points_normalised', 'first_name', 'second_name', 'price_million', 'team_short']].head(10))



