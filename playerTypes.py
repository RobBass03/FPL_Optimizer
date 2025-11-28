# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:39:33 2025

@author: Robbi
"""

import numpy as np
import pandas as pd

TEAM_IDS = np.loadtxt("team_data.csv", dtype=str, skiprows=1, usecols=5, delimiter=',')
TEAM_STRENGTH = np.loadtxt("team_data.csv", dtype=float, skiprows=1, usecols=(3,4), delimiter=',')
# Assuming col3 = strength_attack, col4 = strength_defence

FIXTURES = np.loadtxt("fixture_data.csv", dtype=int, skiprows=1, usecols=(9,11), delimiter=',')

DEFAULT_LENGTH = 5

MY_TEAM = ['Raya', 'Vicario', 
           'Chalobah', 'Virgil', 'Senesi', 'J.Timber', 'Gabriel',
           'Mbeumo', 'Gakpo', 'Semenyo', 'Szoboszlai', 'Casemiro',
           'João Pedro', 'Thiago', 'Mateta']

# MY_TEAM = ['Raya', 'Sánchez',                                            # Dylan
#            'Van de Ven', 'Guéhi', 'Senesi', 'Trippier', 'Gabriel',
#            'Kudus', 'Caicedo', 'Semenyo', 'Grealish', 'Reijnders',
#            'Haaland', 'Mateta', 'Woltemade']

# MY_TEAM = ['Raya', 'Vicario',                                       # WEEK 8 TEAM
#            'Chalobah', 'Virgil', 'Senesi', 'Romero', 'Ekdal',
#            'Mbeumo', 'M.Salah', 'Semenyo', 'Gravenberch', 'Iwobi',
#            'João Pedro', 'Thiago', 'Wood']

# MY_TEAM = ['Woodman', 'Vicario',                                       # SHIT TEAM
#            'Chalobah', 'Virgil', 'Senesi', 'Romero', 'Ekdal',
#            'Kabia', 'Broggio', 'Sambi', 'Gravenberch', 'Iwobi',
#            'João Pedro', 'Thiago', 'Wood']


MY_TEAM_IDS = [0] * 15

def team_name(team_id):
    return str(TEAM_IDS[team_id-1])

def process_status(status):
    if status == 'a':
        return 'Available'
    if status == 'u':
        return 'Unavailable'
    if status == 'i':
        return 'Injured'
    if status == 'd':
        return 'Doubtful'
    if status == 's':
        return 'Suspended'
    return f"other: {status}"

def set_points_per_GW(data):
    result = []
    for num in data:
        result.append(float(num))
    return result

def print_my_team(goalies, defs, middies, strikes):
    print(goalies[MY_TEAM_IDS[0]])
    print(goalies[MY_TEAM_IDS[1]])
    print(defs[MY_TEAM_IDS[2]])
    print(defs[MY_TEAM_IDS[3]])
    print(defs[MY_TEAM_IDS[4]])
    print(defs[MY_TEAM_IDS[5]])
    print(defs[MY_TEAM_IDS[6]])
    print(middies[MY_TEAM_IDS[7]])
    print(middies[MY_TEAM_IDS[8]])
    print(middies[MY_TEAM_IDS[9]])
    print(middies[MY_TEAM_IDS[10]])
    print(middies[MY_TEAM_IDS[11]])
    print(strikes[MY_TEAM_IDS[12]])
    print(strikes[MY_TEAM_IDS[13]])
    print(strikes[MY_TEAM_IDS[14]])

def find_next_fixture(team_id):
    for fixture in FIXTURES:
        if fixture[0] == team_id:
            return str(TEAM_IDS[fixture[1]-1])
        if fixture[1] == team_id:
            return str(TEAM_IDS[fixture[0]-1])

class Player:
    
    def __str__(self):
        string = f"{self.name} - {self.team} {self.playerType}:\n"
        string += f"    Total Points:               {self.total_points}\n"
        string += f"    Normalised Points:          {self.normalised_points:.3f}\n"
        string += f"    Current Cost:               £{self.now_cost}M\n"
        string += f"    Expected Points:            {self.ep_next}\n"
        string += f"    Status:                     {self.status}\n"
        if self.chance_of_playing_next_round != 'Undefined':   
            string += f"    Chance of Playing:          {self.chance_of_playing_next_round}\n"
        string += f"    Next Fixture:               {self.next_fixture}\n"
        string += f"    Next fixture difficulty:    {self.next_fixture_difficulty:.1f}\n"
        string += f"    Points Per GW Type Ranking: {self.points_per_game_rank_type}\n"
        string += f"    Points Last 3 Gws:          {int(self.points_per_GW[-1])}, {int(self.points_per_GW[-2])}, {int(self.points_per_GW[-3])}"
        string += "\n"
        return string
    
    __repr__ = __str__
    
    def _cust_init(self, data):
        """Initialize fields shared by all player types, enriched with team and fixture difficulty data."""
    
        # --- General attributes ---
        self.can_transact = bool(data[0])
        self.can_select = bool(data[1])
        self.ep_next = float(data[4])
        self.form = float(data[6])
        self.id = int(data[7])
        self.now_cost = float(data[8]) / 10
        self.points_per_game = float(data[9])
        self.status = process_status(data[11])
        self.team_id = int(data[12])
        self.team = team_name(self.team_id)
        self.total_points = int(data[14])
        self.name = str(data[15])
        self.minutes = float(data[16])
    
        # --- Shared attributes across positions ---
        self.yellow_cards = int(data[24])
        self.red_cards = int(data[25])
        self.bonus = int(data[27])
        self.starts = int(data[29])
    
        # --- Rankings and historical performance ---
        self.points_per_game_rank_type = int(data[41])
        self.points_per_GW = set_points_per_GW(data[45:])
        self.normalised_points = self._points_per_mil_pounds()
    
        # --- Chance of playing next round ---
        self.chance_of_playing_next_round = (
            float(data[2]) if float(data[2]) != -1.0 else 'Undefined'
        )
    
        # --- Fixture and Team Strength Integration ---
        # Load lightweight static data once per process (cached in globals)
        global TEAM_STRENGTHS, FIXTURE_MAP, TEAM_IDS, FIXTURES
    
        try:
            TEAM_STRENGTHS
        except NameError:
            # Load once: average attack/defence from home/away strengths
            df_team = pd.read_csv("team_data.csv")
            df_team["strength_attack_avg"] = (df_team["strength_attack_home"] + df_team["strength_attack_away"]) / 2
            df_team["strength_defence_avg"] = (df_team["strength_defence_home"] + df_team["strength_defence_away"]) / 2
            TEAM_STRENGTHS = df_team.set_index("id")[["strength_attack_avg", "strength_defence_avg"]].to_dict("index")
    
            # Build fixture lookup
            df_fix = pd.read_csv("fixture_data.csv")
            FIXTURE_MAP = {}
            for _, row in df_fix.iterrows():
                FIXTURE_MAP.setdefault(row["team_h"], []).append(
                    (row["team_a"], row["team_h_difficulty"])
                )
                FIXTURE_MAP.setdefault(row["team_a"], []).append(
                    (row["team_h"], row["team_a_difficulty"])
                )
    
        # Assign team-level strengths
        self.team_strength_attack = TEAM_STRENGTHS[self.team_id]["strength_attack_avg"]
        self.team_strength_defence = TEAM_STRENGTHS[self.team_id]["strength_defence_avg"]
    
        # --- Find next fixture difficulty ---
        opp_id, diff = None, None
        if self.team_id in FIXTURE_MAP:
            opp_id, diff = FIXTURE_MAP[self.team_id][0]  # first listed future fixture
    
        self.opponent_team_id = opp_id if opp_id else -1
        self.opponent_team_name = team_name(opp_id) if opp_id else "Unknown"
        self.next_fixture_difficulty = float(diff) if diff is not None else -1.0
    
        # --- Link player IDs if on your team ---
        if self.name in MY_TEAM:
            i = MY_TEAM.index(self.name)
            MY_TEAM_IDS[i] = self.id
    
        # --- Next fixture name (for display) ---
        self.next_fixture = self.opponent_team_name
    
        # --- Derived metric: adjusted difficulty factor (for scoring) ---
        # Lower difficulty → higher expected value
        if self.next_fixture_difficulty > 0:
            self.fixture_difficulty_factor = 1.1 - 0.02 * self.next_fixture_difficulty
        else:
            self.fixture_difficulty_factor = 1.0

    
    def _points_per_mil_pounds(self):
        return self.total_points / self.now_cost
    
    def set_starting(self, starting=True):
        self.starting = starting

class GK(Player):
    def __init__(self, data):
        self.playerType = 'GK'
        self._cust_init(data)
        self.clean_sheets = int(data[19])
        self.goals_conceded = int(data[20])
        self.own_goals = int(data[21])
        self.penalties_saved = int(data[22])
        self.saves = int(data[26])
        self.expected_goals_conceded = float(data[34])
        self.saves_per_90 = float(data[35])
        self.expected_goals_conceded_per_90 = float(data[38])
        self.goals_conceded_per_90 = float(data[39])
        self.points_per_game_rank = int(data[40])
        self.starts_per_90 = float(data[42])
        self.clean_sheets_per_90 = float(data[43])


class DEF(Player):
    def __init__(self, data):
        self.playerType = 'DEF'
        self._cust_init(data)
        self.goals_scored = int(data[17])
        self.assists = int(data[18])
        self.clean_sheets = int(data[19])
        self.goals_conceded = int(data[20])
        self.own_goals = int(data[21])
        self.defensive_contribution = int(data[28])
        self.expected_goals = float(data[30])
        self.expected_assists = float(data[31])
        self.expected_goal_involvements = float(data[32])
        self.expected_goals_conceded = float(data[33])
        self.expected_goals_per_90 = float(data[34])
        self.expected_assists_per_90 = float(data[36])
        self.expected_goal_involvements_per_90 = float(data[37])
        self.expected_goals_conceded_per_90 = float(data[38])
        self.clean_sheets_per_90 = float(data[43])
        self.defensive_contribution_per_90 = float(data[44])


class MID(Player):
    def __init__(self, data):
        self.playerType = 'MID'
        self._cust_init(data)
        self.goals_scored = int(data[17])
        self.assists = int(data[18])
        self.clean_sheets = int(data[19])
        self.expected_goals = float(data[30])
        self.expected_assists = float(data[31])
        self.expected_goal_involvements = float(data[32])
        self.expected_goals_per_90 = float(data[34])
        self.expected_assists_per_90 = float(data[36])
        self.expected_goal_involvements_per_90 = float(data[37])
        self.points_per_game_rank = int(data[40])
        self.starts_per_90 = float(data[42])
        self.clean_sheets_per_90 = float(data[43])


class FWD(Player):
    def __init__(self, data):
        self.playerType = 'FWD'
        self._cust_init(data)
        self.goals_scored = int(data[17])
        self.assists = int(data[18])
        self.expected_goals = float(data[30])
        self.expected_assists = float(data[31])
        self.expected_goal_involvements = float(data[32])
        self.expected_goals_per_90 = float(data[34])
        self.expected_assists_per_90 = float(data[36])
        self.expected_goal_involvements_per_90 = float(data[37])
        self.points_per_game_rank = int(data[40])
        self.starts_per_90 = float(data[42])

        self._cust_init(data)
    


class Players(dict):
    
    def __getitem__(self, key):
        """
        Allow lookup by ID (int) or player name (str).
        Returns the Player object itself.
        """
        # --- numeric lookup (default behaviour) ---
        if isinstance(key, int):
            return super().__getitem__(key)

        # --- string lookup by name (case-insensitive) ---
        if isinstance(key, str):
            for player in self.values():
                # Match if full name, partial name, or case-insensitive
                if key.lower() == player.name.lower():
                    return player
            raise KeyError(f"No player found with name matching '{key}'")

        # --- fallback ---
        raise KeyError(f"Unsupported key type: {type(key)}")
    
    def pop(self, key, default=None):
        """
        Remove and return a player by ID or name (case-insensitive).
        Works the same as dict.pop().
        """
        # --- numeric lookup ---
        if isinstance(key, int):
            return super().pop(key, default)

        # --- string lookup (name) ---
        if isinstance(key, str):
            for pid, player in list(self.items()):
                if key.lower() == player.name.lower():
                    return super().pop(pid)
            if default is not None:
                return default
            raise KeyError(f"No player found with name matching '{key}'")

        # --- fallback ---
        raise KeyError(f"Unsupported key type for pop: {type(key)}")
    
    def __str__(self):
        string = ""
        for player in self.values():
            string += player.__str__() + '\n'
        return string
    
    __repr__ = __str__
    
    def highest_points(self):
        max_points = -1
        for player, data in self.items():
            if data.total_points > max_points:
                top_lad = player
                max_points = data.total_points
        return self[top_lad]

    def cost_cutoff(self, cutoff):
        result = Players()
        for player, data in self.items():
            if data.now_cost <= cutoff:
                result[player] = data
        return result
    
    def ranked_normalised_points(self, length=DEFAULT_LENGTH, team=None):
        if len(self) < length:
            length = len(self)
        
        points = np.zeros((len(self), 2))
        i = 0
        for player, data in self.items():
            if team == None or team == data.team:
                points[i, 0] = data.normalised_points
                points[i, 1] = player
            i += 1
        
        points_sorted = points[np.argsort(points[:, 0])[::-1]]
        
        for i in range(length):
            print(self[int(points_sorted[i, 1])])
            
    def ranked_total_points(self, length=DEFAULT_LENGTH, team=None):
        if len(self) < length:
            length = len(self)
        
        points = np.zeros((len(self), 2))
        i = 0
        for player, data in self.items():
            if team == None or team == data.team:
                points[i, 0] = data.total_points
                points[i, 1] = player
            i += 1
        
        points_sorted = points[np.argsort(points[:, 0])[::-1]]
        
        for i in range(length):
            print(self[int(points_sorted[i, 1])])
            
    def ranked_ep_next(self, length=DEFAULT_LENGTH, team=None):
        if len(self) < length:
            length = len(self)
        
        points = np.zeros((len(self), 2))
        i = 0
        for player, data in self.items():
            if team == None or team == data.team:
                points[i, 0] = data.ep_next
                points[i, 1] = player
                i += 1
        
        points_sorted = points[np.argsort(points[:, 0])[::-1]]
        
        for j in range(length):
            print(self[int(points_sorted[j, 1])])
            
    def gks(self, printAll=False):
        for player in self.values():
            if player.playerType == 'GK' and (player.starting or printAll):
                print(player)
            
    def defs(self, printAll=False):
        for player in self.values():
            if player.playerType == 'DEF' and (player.starting or printAll):
                print(player)
    
    def mids(self, printAll=False):
        for player in self.values():
            if player.playerType == 'MID' and (player.starting or printAll):
                print(player)
    
    def fwds(self, printAll=False):
        for player in self.values():
            if player.playerType == 'FWD' and (player.starting or printAll):
                print(player)













        
    