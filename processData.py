types = [('can_transact', bool), 
         ('can_select',bool), 
         ('chance_of_playing_next_round',float), 
         ('element_type',int),
         ('ep_next',float),
         ('first_name',str),
         ('form',float),
         ('id',int),
         ('now_cost',float),
         ('points_per_game',float),
         ('second_name',str),
         ('status',str),
         ('team',int),
         ('team_code',int),
         ('total_points',int),
         ('web_name',str),
         ('minutes',float),
         ('goals_scored',int),
         ('assists',int),
         ('clean_sheets',int),
         ('goals_conceded',int),
         ('own_goals',int),
         ('penalties_saved',int),
         ('penalties_missed',int),
         ('yellow_cards',int),
         ('red_cards',int),
         ('saves',int),
         ('bonus',int),
         ('defensive_contribution',int),
         ('starts',int),
         ('expected_goals',float),
         ('expected_assists',float),
         ('expected_goal_involvements',float),
         ('expected_goals_conceded',float),
         ('expected_goals_per_90',float),
         ('saves_per_90',float),
         ('expected_assists_per_90',float),
         ('expected_goal_involvements_per_90',float),
         ('expected_goals_conceded_per_90',float),
         ('goals_conceded_per_90',float),
         ('points_per_game_rank',int),
         ('points_per_game_rank_type',int),
         ('starts_per_90',float),
         ('clean_sheets_per_90',float),
         ('defensive_contribution_per_90',float),
         ('def_strength', float),
         ('att_strength', float),
         ('next_fixture_difficulty', float)
         ]
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:42:04 2025

@author: Robbi
"""

GAME_WEEK = 12
BUDGET_EXCESS = 1.5
FREE_TRANS = 1
IGNORE_LIST = ['Semeno', 'Gabriel']

import numpy as np
from playerTypes import *
import matplotlib.pyplot as plt

FILENAME = "fpl_player_data_with_weekly_points.csv"

PLAIN_TYPES = []   

ALL_PLAYERS = 0
GKs = 0
DEFs = 0
MIDs = 0
FWDs = 0
my_team = 0
updated_team = 0

cols = [0,1,2,10,11,14,15,16,20,22,24,28,29,30,31,38,44,45,46,47,48,49,50,51,52,53,54,55,64,65,66,67,68,69,84,85,86,
        87,88,89,94,95,98,99,100]
for i in range(GAME_WEEK):
    cols.append(101+i)
    types.append((f"GW{i+1}", int))

for item in types:
    PLAIN_TYPES.append(item[1])

def load_data():
    data = np.loadtxt(FILENAME, delimiter=',', usecols=cols, dtype=str, skiprows=1)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if len(data[i,j]) == 0:
                if PLAIN_TYPES[j] == bool:
                    data[i,j] = True
                elif PLAIN_TYPES[j] == int:
                    data[i,j] = -1
                elif PLAIN_TYPES[j] == float:
                    data[i,j] = -1.0
                elif PLAIN_TYPES[j] == str:
                    data[i,j] = 'None'
    
    return data

def set_my_team(goalies, defs, middies, strikes):
    my_team = Players()
    for i in range(len(MY_TEAM)):
        if i < 2:
            my_team[MY_TEAM_IDS[i]] = goalies[MY_TEAM_IDS[i]]
        elif i < 7:
            my_team[MY_TEAM_IDS[i]] = defs[MY_TEAM_IDS[i]]
        elif i < 12:
            my_team[MY_TEAM_IDS[i]] = middies[MY_TEAM_IDS[i]]
        elif i < 15:
            my_team[MY_TEAM_IDS[i]] = strikes[MY_TEAM_IDS[i]]
    return my_team

def pick_best_lineup_auto(my_team):
    """
    Automatically pick the best FPL lineup, formation, and captain/vice-captain.

    Returns:
        dict containing:
            - formation
            - captain / vice_captain
            - total_points (with captain doubled)
            - raw_total_points (without doubling)
    """
    
    for player in my_team.values():
        player.set_starting(starting = False)

    # --- Split by position ---
    goalkeepers = [p for p in my_team.values() if p.playerType == 'GK']
    defenders   = [p for p in my_team.values() if p.playerType == 'DEF']
    midfielders = [p for p in my_team.values() if p.playerType == 'MID']
    forwards    = [p for p in my_team.values() if p.playerType == 'FWD']

    # --- Scoring function ---
    def projected_score(player):
        if player.status != 'Available':
            return -10.0
        return player.ep_next  # using ep_next as the key expected points metric

    # --- Sort each category ---
    goalkeepers.sort(key=projected_score, reverse=True)
    defenders.sort(key=projected_score, reverse=True)
    midfielders.sort(key=projected_score, reverse=True)
    forwards.sort(key=projected_score, reverse=True)

    valid_formations = [
        (3, 4, 3),
        (3, 5, 2),
        (4, 4, 2),
        (4, 3, 3),
        (4, 5, 1),
        (5, 3, 2),
        (5, 4, 1),
    ]

    best_lineup = None
    best_score = -1.0

    for formation in valid_formations:
        num_def, num_mid, num_fwd = formation
        if len(defenders) < num_def or len(midfielders) < num_mid or len(forwards) < num_fwd:
            continue

        starters = (
            [goalkeepers[0]] +
            defenders[:num_def] +
            midfielders[:num_mid] +
            forwards[:num_fwd]
        )
        total_projected = sum(projected_score(p) for p in starters)

        if total_projected > best_score:
            bench = (
                [goalkeepers[1]] +
                defenders[num_def:] +
                midfielders[num_mid:] +
                forwards[num_fwd:]
            )
            bench.sort(key=projected_score, reverse=True)

            best_score = total_projected
            best_lineup = {
                'GK': goalkeepers[0],
                'DEF': defenders[:num_def],
                'MID': midfielders[:num_mid],
                'FWD': forwards[:num_fwd],
                'BENCH': bench,
                'formation': formation,
                'raw_total_points': total_projected  # store before captain bonus
            }

    # --- Captain and vice-captain selection ---
    starters = [best_lineup['GK']] + best_lineup['DEF'] + best_lineup['MID'] + best_lineup['FWD']
    starters.sort(key=projected_score, reverse=True)
    for starter in starters:
        starter.set_starting()
    
    
    best_lineup['captain'] = starters[0]
    best_lineup['vice_captain'] = starters[1]

    # --- Add captain double points ---
    total_with_captain = best_lineup['raw_total_points'] + projected_score(best_lineup['captain'])
    best_lineup['total_points'] = total_with_captain

    return best_lineup


def print_optimal_team(my_team):
    lineup = pick_best_lineup_auto(my_team)
    formation = lineup['formation']
    total_points = lineup['total_points']

    print("\n=== Optimal Lineup ===")
    print(f"Formation: {formation[0]}-{formation[1]}-{formation[2]}")
    print(f"Predicted Total Points: {total_points:.2f}\n")

    # Rows
    gk_line = [lineup['GK']]
    def_line = lineup['DEF']
    mid_line = lineup['MID']
    fwd_line = lineup['FWD']

    # Helper to format names with captain/vice tags
    def fmt(player):
        tag = ""
        if player == lineup['captain']:
            tag = " (C)"
        elif player == lineup['vice_captain']:
            tag = " (VC)"
        return f"{player.name}{tag}"

    # Prepare lines of text for each row
    gk_text  = "   ".join(fmt(p) for p in gk_line)
    def_text = "   ".join(fmt(p) for p in def_line)
    mid_text = "   ".join(fmt(p) for p in mid_line)
    fwd_text = "   ".join(fmt(p) for p in fwd_line)

    # Find the maximum line width for alignment
    all_lines = [gk_text, def_text, mid_text, fwd_text]
    max_width = max(len(line) for line in all_lines)

    # Center each line according to the widest
    def center(line):
        return line.center(max_width)

    # Print formation
    print("  ", center(gk_text), '\n')
    print("  ", center(def_text), '\n')
    print("  ", center(mid_text), '\n')
    print("  ", center(fwd_text))

    # Bench at bottom
    bench_text = ", ".join(player.name for player in lineup['BENCH'])
    print("\n" + "=" * max_width)
    print(center(f"Bench: {bench_text}"))
    print("=" * max_width)


    

def suggest_optimal_transfer(my_team, all_players, budget_excess=0.5, force=None, ignore=None):
    """
    Suggest the single most optimal transfer from your current squad.
    Allows selling unavailable players, but only buys available ones.

    Parameters
    ----------
    my_team : Players
        Your 15-man squad (Players dictionary).
    all_players : Players
        Dictionary of all available players (from all positions).
    budget_excess : float
        Extra budget in millions available to spend (e.g. 0.5 = £0.5M extra).

    Returns
    -------
    dict
        {
            'out': Player to sell,
            'in': Player to buy,
            'gain': float (expected points gain),
            'cost_diff': float (cost increase),
            'new_ep_next': float (expected points of incoming),
            'in_status': str (status of incoming player)
        }
    """

    # --- Scoring function (same weighting as lineup logic) ---
    def projected_score(player):
        """Estimate performance potential, with minor penalty for unavailability."""
        base_score = 0.5 * player.ep_next + 0.2 * player.form + 0.1 * player.normalised_points + 0.2 * player.total_points

        # Heavily penalise unavailable players for accurate comparisons
        if player.status == 'Available':
            return base_score
        elif player.status == 'Doubtful':
            return base_score * 0.7
        elif player.status == 'Injured':
            return base_score * 0.4
        elif player.status == 'Suspended':
            return base_score * 0.3
        else:
            return base_score * 0.2

    best_gain = -1.0
    best_trade = None
    candidates = Players()
    
    if force != None:
        for candidate in force:
            candidates[candidate] = my_team[candidate]
    elif ignore != None:
        candidates = Players(my_team.copy())
        for candidate in ignore:
            temp = candidates.pop(candidate)
    else:
        candidates = Players(my_team.copy())
    
    
    team_count = {}
    for player in my_team.values():
        team = player.team_id 
        if team in team_count:
            team_count[team] += 1
        else:
            team_count[team] = 1
    
    
    # --- Loop through your current squad (anyone can be sold) ---
    for outgoing_player in candidates.values():
        position_type = outgoing_player.playerType
        current_score = projected_score(outgoing_player)
        current_cost = outgoing_player.now_cost
        
        # Search only *available* replacements of the same position
        for candidate in all_players.values():
            if candidate.playerType != position_type:
                continue
            if candidate.id in my_team.keys():
                continue
            if candidate.status != 'Available':
                continue  # only buy available players
            if candidate.now_cost > current_cost + budget_excess:
                continue  # outside your budget
            if candidate.team_id in team_count:
                if team_count[candidate.team_id] > 2:
                    continue

            # Calculate expected improvement
            gain = projected_score(candidate) - current_score
            if gain > best_gain:
                cost_diff = candidate.now_cost - current_cost
                best_trade = {
                    'out': outgoing_player,
                    'in': candidate,
                    'gain': candidate.ep_next - outgoing_player.ep_next,
                    'cost_diff': cost_diff,
                    'new_ep_next': candidate.ep_next,
                    'in_status': candidate.status
                }
                best_gain = gain

    return best_trade


def print_optimal_trade(my_team, all_players, budget_excess, force=None, ignore=None):
    trade = suggest_optimal_transfer(my_team, all_players, budget_excess, force, ignore)
    
    print("\n=== Optimal Transfer Suggestion ===")
    if trade:
        print(f"Sell: {trade['out'].name} (£{trade['out'].now_cost}M, {trade['out'].status})")
        print(f"Buy:  {trade['in'].name} (£{trade['in'].now_cost}M, {trade['in'].team})")
        print(f"Expected Points Gain: {trade['gain']:.2f}")
        print(f"Budget Change: {-1 * trade['cost_diff']:+.2f}M")
        print(f"New Player Projected Points: {trade['new_ep_next']:.2f}")
    else:
        print("No beneficial trades found within budget.")

import itertools
from tqdm import tqdm

def suggest_optimal_transfers(
    my_team, all_players,
    budget_excess=0.5,
    num_free_transfers=1,
    force=None, ignore=None,
    max_outgoing_per_type=5,
    improvement_threshold=0.3,
    show_progress=True,
    print_lineup=False
):
    """
    Search for the best combination of transfers (free + optional paid),
    prune low-value candidates, and print new optimal lineup.

    Prevents buying the same player twice.
    """

    def projected_score(p):
        # lower difficulty = easier fixture
        diff_factor = 1.2 - 0.1 * min(max(p.next_fixture_difficulty / 100, 0), 1)
        recent_points = 0.6*p.points_per_GW[-1] + 0.3*p.points_per_GW[-2] + 0.1*p.points_per_GW[-3]
        base = 0.5*p.ep_next + 0.3*recent_points + 0.2*p.form
        return base * diff_factor


    # ---- Step 1: prune dataset ----
    pruned_players = Players({pid: p for pid, p in all_players.items() if p.ep_next >= 4.0})

    # ---- Step 2: build initial candidate set ----
    if force:
        candidate_pool = [my_team[name] for name in force if name in my_team]
    elif ignore:
        candidate_pool = [p for p in my_team.values() if p.name not in ignore]
    else:
        candidate_pool = list(my_team.values())

    grouped = {}
    for p in candidate_pool:
        grouped.setdefault(p.playerType, []).append(p)
    trimmed_pool = []
    for g in grouped.values():
        g.sort(key=lambda x: projected_score(x))
        trimmed_pool.extend(g[:max_outgoing_per_type])

    # ---- Step 3: bookkeeping ----
    base_team_count = {}
    for p in my_team.values():
        base_team_count[p.team_id] = base_team_count.get(p.team_id, 0) + 1

    def valid_replacements(out, budget_left, team_count):
        out_score = projected_score(out)
        out_cost = out.now_cost
        for cand in pruned_players.values():
            if cand.playerType != out.playerType:
                continue
            if cand.id in my_team.keys():
                continue
            if cand.status != "Available":
                continue
            if cand.now_cost > out_cost + budget_left:
                continue
            # Allow same-team swaps when replacing a player from that team
            if team_count.get(cand.team_id, 0) >= 3 and cand.team_id != out.team_id:
                continue
            if projected_score(cand) < out_score + improvement_threshold:
                continue
            yield cand

    # ---- Step 4: main search ----
    best_combo = None
    best_gain = -1e9
    max_trades = num_free_transfers + 2

    for n_transfers in range(1, max_trades + 1):
        outs_combos = list(itertools.combinations(trimmed_pool, n_transfers))
        iterator = tqdm(outs_combos, desc=f"Searching {n_transfers}-transfer combos", disable=not show_progress)

        for outs in iterator:
            current_budget = budget_excess + sum(o.now_cost for o in outs)
            temp_team_count = base_team_count.copy()
            for o in outs:
                temp_team_count[o.team_id] -= 1

            in_options = [list(valid_replacements(o, budget_excess, temp_team_count)) for o in outs]
            if any(len(opt) == 0 for opt in in_options):
                continue

            for ins in itertools.product(*in_options):
                # 🧩 NEW: skip duplicates in incoming players
                in_ids = [p.id for p in ins]
                if len(in_ids) != len(set(in_ids)):
                    continue  # duplicate purchase detected, skip

                new_team_count = temp_team_count.copy()
                new_cost = sum(c.now_cost for c in ins)
                budget_used = new_cost - sum(o.now_cost for o in outs)
                if budget_used > budget_excess:
                    continue

                valid = True
                for c in ins:
                    new_team_count[c.team_id] = new_team_count.get(c.team_id, 0) + 1
                    if new_team_count[c.team_id] > 3:
                        valid = False
                        break
                if not valid:
                    continue

                gain = sum(c.ep_next - o.ep_next for o, c in zip(outs, ins))
                paid = max(0, n_transfers - num_free_transfers)
                net_gain = gain - 8 * paid
                if net_gain > (best_gain):
                    best_gain = net_gain
                    best_combo = {
                        "outs": outs,
                        "ins": ins,
                        "raw_gain": gain,
                        "net_gain": net_gain + 4*paid,
                        "n_paid": paid,
                        "budget_left": budget_excess - budget_used
                    }

    # ---- Step 5: print + apply ----
    if not best_combo:
        print("\nNo beneficial transfer combinations found.")
        return my_team

    print(f"\n=== Optimal Transfer Combination ({num_free_transfers} free) ===")
    print(f"Raw Team Gain: +{best_combo['raw_gain']:.2f} pts")
    print(f"Net Gain (after -4 penalties): +{best_combo['net_gain']:.2f} pts")
    print(f"Paid transfers: {best_combo['n_paid']}, Remaining budget: £{best_combo['budget_left']:.2f}M\n")

    new_team = Players(my_team.copy())
    for o, c in zip(best_combo["outs"], best_combo["ins"]):
        new_team.pop(o.id)
        new_team[c.id] = c
        print(f"  Sell: {o.name} (£{o.now_cost:.1f}M)  →  Buy: {c.name} (£{c.now_cost:.1f}M, {c.team}) "
              f"[Δ +{c.ep_next - o.ep_next:.2f} EP]")

    # ---- Step 6: re-evaluate best lineup ----
    from processData import pick_best_lineup_auto
    lineup = pick_best_lineup_auto(new_team)

    if print_lineup:    

        print("\n=== New Optimal Lineup (After Transfers) ===")
        print(f"Formation: {lineup['formation'][0]}-{lineup['formation'][1]}-{lineup['formation'][2]}")
        print(f"Predicted Total Points: {lineup['total_points']:.2f}\n")
    
        def fmt(p):
            tag = ""
            if p == lineup["captain"]:
                tag = " (C)"
            elif p == lineup["vice_captain"]:
                tag = " (VC)"
            return f"{p.name}{tag}"
    
        gk = "   ".join(fmt(p) for p in [lineup["GK"]])
        defs = "   ".join(fmt(p) for p in lineup["DEF"])
        mids = "   ".join(fmt(p) for p in lineup["MID"])
        fwds = "   ".join(fmt(p) for p in lineup["FWD"])
    
        all_lines = [gk, defs, mids, fwds]
        max_width = max(len(l) for l in all_lines)
        center = lambda line: line.center(max_width)
    
        print("  ", center(gk), "\n")
        print("  ", center(defs), "\n")
        print("  ", center(mids), "\n")
        print("  ", center(fwds))
        bench_text = ", ".join(p.name for p in lineup["BENCH"])
        print("\n" + "=" * max_width)
        print(center(f"Bench: {bench_text}"))
        print("=" * max_width)

    return new_team

def print_optimal_transfers(my_team, all_players, budget_excess=0.5, num_free_transfers=1, force=None, ignore=None):
    trades = suggest_optimal_transfers(my_team, all_players, budget_excess, num_free_transfers, force, ignore)
    print(f"\n=== Optimal Transfer Suggestions ({num_free_transfers} free) ===")
    if not trades:
        print("No beneficial transfers found.")
        return
    for i, t in enumerate(trades, 1):
        print(f"\nTrade {i}:")
        print(f"  Sell: {t['out'].name} (£{t['out'].now_cost}M, {t['out'].status})")
        print(f"  Buy:  {t['in'].name} (£{t['in'].now_cost}M, {t['in'].team})")
        print(f"  Raw Gain: {t['gain']:+.2f} pts | Net Gain (after penalties): {t['net_gain']:+.2f} pts")
        print(f"  Budget Change: {-1*t['cost_diff']:+.2f}M")
    print(f"\nRemaining budget: £{trades[-1]['new_budget']:.2f}M")

from PIL import Image, ImageDraw, ImageFont, ImageOps
import pandas as pd
import os

def render_optimal_team_image(my_team, filename="team.png",
                              width=1800, height=2400,
                              badge_folder="badges"):

    lineup = pick_best_lineup_auto(my_team)
    formation = lineup['formation']

    # Load team badge map
    df_team = pd.read_csv("team_data.csv")[["id","short_name"]]
    TEAM_SHORT = dict(zip(df_team["id"], df_team["short_name"]))

    # ---------------------------
    # Canvas
    # ---------------------------
    img = Image.new("RGB", (width, height), color=(10, 25, 50))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 80)
        name_font  = ImageFont.truetype("arial.ttf", 48)
        badge_font = ImageFont.truetype("arial.ttf", 40)
    except:
        title_font = name_font = badge_font = ImageFont.load_default()

    # ---------------------------
    # Stadium background
    # ---------------------------
    for y in range(height):
        v = int(30 + 20*(y/height))
        draw.line([(0,y),(width,y)], fill=(v, v+10, v+20))

    # ---------------------------
    # Title bar
    # ---------------------------
    draw.rectangle([(0,0),(width,180)], fill=(230,230,230))
    draw.text((width//2, 90),
              f"Optimal Lineup: {formation[0]}-{formation[1]}-{formation[2]}",
              anchor="mm", font=title_font, fill=(20,20,20))

    # ---------------------------
    # Pitch
    # ---------------------------
    pitch_top = 200
    pitch_bottom = height - 300
    pitch_left = 100
    pitch_right = width - 100

    stripe_h = 120
    g1 = (30,140,40)
    g2 = (25,130,35)

    cur = pitch_top
    toggle = True
    while cur < pitch_bottom:
        draw.rectangle([(pitch_left,cur),(pitch_right,cur+stripe_h)],
                       fill=g1 if toggle else g2)
        cur += stripe_h
        toggle = not toggle

    draw.rectangle([(pitch_left,pitch_top),(pitch_right,pitch_bottom)],
                   outline="white", width=8)

    cx = (pitch_left + pitch_right)//2
    draw.rectangle([(cx-180,pitch_top),
                    (cx+180,pitch_top+200)], outline="white", width=6)
    draw.rectangle([(cx-120,pitch_top),
                    (cx+120,pitch_top+120)], outline="white", width=6)
    draw.ellipse([(cx-250,(pitch_top+pitch_bottom)//2 - 250),
                  (cx+250,(pitch_top+pitch_bottom)//2 + 250)],
                 outline="white", width=6)

    # ---------------------------
    # Badge loader
    # ---------------------------
    def load_badge(team_id):
        short = TEAM_SHORT.get(team_id, None)
        if not short:
            return None
        path = os.path.join(badge_folder, f"{short}.png")
        if not os.path.exists(path):
            return None
        badge = Image.open(path).convert("RGBA")
        return badge

    # ---------------------------
    # Draw a player (New style)
    # ---------------------------
    def draw_player(x, y, player, is_captain=False, is_vice=False):
        """
        Draws:
        - black circular backing
        - team badge centered above name
        - nameplate
        - round C/VC badge below name
        """

        # ---- Team badge + black circular backing ----
        badge_img = load_badge(player.team_id)
        if badge_img:
            badge_img = ImageOps.contain(badge_img, (160,160))

            # Center point for badge + backing (tangent to nameplate)
            radius = 100
            badge_cx = x
            badge_cy = y - 60     # <--- new corrected center
            
            # Black circular backing
            draw.ellipse([(badge_cx - radius, badge_cy - radius),
                          (badge_cx + radius, badge_cy + radius)],
                         fill="black")
            
            # Paste badge centered inside
            bx = badge_cx - badge_img.width // 2
            by = badge_cy - badge_img.height // 2
            img.paste(badge_img, (bx, by), badge_img)


        else:
            # Fallback circle if badge missing
            draw.ellipse([(x-80, y-200), (x+80, y-40)],
                         fill=(0,0,0), outline="white", width=4)

        # ---- Nameplate ----
        draw.rectangle([(x-170, y+40), (x+170, y+100)],
                       fill=(0,0,0))
        draw.text((x, y+70), player.name, anchor="mm",
                  font=name_font, fill="white")

        # ---- Captain / VC circular badge (below name) ----
        if is_captain or is_vice:
            label = "C" if is_captain else "VC"
            radius = 45
            cx = x
            cy = y + 160

            # Circle
            draw.ellipse([(cx-radius, cy-radius),
                          (cx+radius, cy+radius)],
                         fill="black", outline="white", width=4)

            # Text
            draw.text((cx, cy), label, anchor="mm",
                      font=badge_font, fill="white")

    # ---------------------------
    # Layout rows (standard orientation)
    # ---------------------------
    rows = [
        [lineup["GK"]],
        lineup["DEF"],
        lineup["MID"],
        lineup["FWD"],
    ]

    row_y = [
        pitch_top + 260,
        pitch_top + 650,
        pitch_top + 1040,
        pitch_top + 1480,
    ]

    # Draw players
    for row, y in zip(rows, row_y):
        step = width // (len(row)+1)
        for i, p in enumerate(row, start=1):
            draw_player(
                step*i, y, p,
                is_captain=(p == lineup["captain"]),
                is_vice   =(p == lineup["vice_captain"])
            )

    # ---------------------------
    # Bench
    # ---------------------------
    bench_y = pitch_bottom + 70
    draw.rectangle([(0, bench_y-30), (width, bench_y+200)],
                   fill=(20,20,20))
    bench_text = "Bench: " + ", ".join(p.name for p in lineup["BENCH"])
    draw.text((width//2, bench_y+80), bench_text,
              anchor="mm", font=name_font, fill="white")

    img.save(filename)

    plt.figure(figsize=(8, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    return filename



def print_data(data):
    global GKs
    global DEFs
    global MIDs
    global FWDs
    global my_team
    global updated_team
    GKs  = Players()
    DEFs = Players()
    MIDs = Players()
    FWDs = Players()
    for row in data:
        if row[3] == '1':
            player = GK(row)
            GKs[player.id] = player
        elif row[3] == '2':
            player = DEF(row)
            DEFs[player.id] = player
        elif row[3] == '3':
            player = MID(row)
            MIDs[player.id] = player
        elif row[3] == '4':
            player = FWD(row)
            FWDs[player.id] = player

    my_team = set_my_team(GKs, DEFs, MIDs, FWDs)
    # cheapFWDS = FWDs.cost_cutoff(15)
    # print(cheapFWDS.highest_points())
    # FWDs.cost_cutoff(9.5).ranked_total_points(length=8)
    # print_my_team(GKs, DEFs, MIDs, FWDs)
    # print("\n\n\n")
    # my_team.ranked_ep_next()
    
    print_optimal_team(my_team)
    
    # Combine all positions
    all_players = Players()
    all_players.update(GKs)
    all_players.update(DEFs)
    all_players.update(MIDs)
    all_players.update(FWDs)
    
    global ALL_PLAYERS
    ALL_PLAYERS = all_players
    
    updated_team = suggest_optimal_transfers(
    my_team, ALL_PLAYERS,
    budget_excess=BUDGET_EXCESS,
    num_free_transfers=FREE_TRANS,
    ignore=IGNORE_LIST,
    print_lineup=True
    )
    
    render_optimal_team_image(my_team, "my_lineup.png")
    render_optimal_team_image(updated_team, "updated_lineup.png")
    # print(updated_team)
    # print_optimal_transfers(my_team, all_players, BUDGET_EXCESS, num_free_transfers=FREE_TRANS) #ignore = ['Virgil', 'Wood']
    
    
    


def main():
    DATA = load_data()
    print_data(DATA)




if __name__ == '__main__':
    main()