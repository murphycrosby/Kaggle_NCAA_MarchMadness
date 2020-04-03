import argparse
import os
from datetime import datetime
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from collections import OrderedDict


data_dir2 = './google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage2'
teams_url = 'https://www.espn.com/mens-college-basketball/teams'
team_schedule_url = 'https://www.espn.com/mens-college-basketball/team/schedule/_/id'
box_score_url = 'https://www.espn.com/mens-college-basketball/boxscore'


def get_espn_teams():
    espn_teams = pd.DataFrame()
    driver = webdriver.Chrome('/Users/murphycrosby/opt/chromedriver/chromedriver')
    driver.get(teams_url)
    content = driver.page_source
    soup = BeautifulSoup(content, 'html5lib')

    teams_div = soup.find('div', attrs={'class': 'Wrapper'})
    for row in teams_div.findAll('section', attrs={'class': 'TeamLinks'}):
        df = {
            'Team': row.findNext("h2").text,
            'Url': row.findNext("a", attrs={"class": "AnchorLink"}).attrs["href"]
        }
        espn_teams = espn_teams.append(df, ignore_index=True)
    espn_teams.to_csv(os.path.join(data_dir2, 'MESPNTeams.csv'), index=False, header=True)


def combine_teams_espn():
    teams_df = pd.read_csv(os.path.join(data_dir2, 'MTeams.csv'))
    espn_df = pd.read_csv(os.path.join(data_dir2, 'MESPNTeams.csv'))
    espn_id = []
    for index, row in teams_df.iterrows():
        team = espn_df[espn_df.Team == row.TeamName]
        if len(team) == 0:
            team = espn_df[espn_df.Team.str.contains(row.TeamName)]
        if len(team) == 0:
            longest = max(row.TeamName.split(' '), key=len)
            team = espn_df[espn_df.Team.str.contains(longest)]
        if (len(team) == 0 or len(team) > 1) \
                and row.LastD1Season == 2020:
            print(f'Match - {row.TeamName}:\n')
            print(f'\t-1: None')
            for idx, row_t in team.iterrows():
                print(f'\t{idx}: {row_t.Team}')
            value = input('Which One: ')
            if value == '-1':
                espn_id.append(0)
                continue
            else:
                team = team.loc[int(value)]
                team = espn_df[espn_df.Team == team.Team]
        if len(team) == 0 and row.LastD1Season < 2020:
            espn_id.append(0)
            continue
        if len(team) == 0:
            espn_id.append(0)
            continue
        url = team['Url'].values[0].split('/')
        espn_id.append(int(url[len(url) - 2]))
    teams_df['ESPNID'] = espn_id
    teams_df.to_csv(os.path.join(data_dir2, 'MTeamsESPN.csv'), index=False, header=True)


def get_specific_team_schedules(espn_id):
    games = pd.DataFrame()
    driver = webdriver.Chrome('/Users/murphycrosby/opt/chromedriver/chromedriver')
    driver.get(f'{team_schedule_url}/{int(espn_id)}')
    content = driver.page_source
    soup = BeautifulSoup(content, 'html5lib')
    schedule_table = soup.find('div', attrs={'class': 'Table__Scroller'})
    for a_row in schedule_table.findAll('a', attrs={'class': 'AnchorLink'}):
        if 'espn.com/mens-college-basketball/game?gameId=' in a_row.attrs['href']:
            game_id = a_row.attrs['href'].split('=')[1]
            df = {
                'GameID': game_id,
                'GameURL': a_row.attrs['href']
            }
            games = games.append(df, ignore_index=True)
    games = games.drop_duplicates()
    games.to_csv(os.path.join(data_dir2, f'MESPNGames-{espn_id}.csv'), index=False, header=True)


def get_team_schedules():
    games = pd.DataFrame()
    driver = webdriver.Chrome('/Users/murphycrosby/opt/chromedriver/chromedriver')
    teams_df = pd.read_csv(os.path.join(data_dir2, 'MTeamsESPN.csv'))
    tourney_df = pd.read_csv(os.path.join(data_dir2, 'MNCAATourneySeeds.csv'))
    for index, row in tourney_df.iterrows():
        team = teams_df[teams_df.TeamID == row.TeamID]
        if len(team) == 0:
            raise ValueError('Could not find team')
        driver.get(f'{team_schedule_url}/{int(team.ESPNID)}')
        content = driver.page_source
        soup = BeautifulSoup(content, 'html5lib')
        schedule_table = soup.find('div', attrs={'class': 'Table__Scroller'})
        for a_row in schedule_table.findAll('a', attrs={'class': 'AnchorLink'}):
            if 'espn.com/mens-college-basketball/game?gameId=' in a_row.attrs['href']:
                game_id = a_row.attrs['href'].split('=')[1]
                df = {
                    'GameID': game_id,
                    'GameURL': a_row.attrs['href']
                }
                games = games.append(df, ignore_index=True)
    games = games.drop_duplicates()
    games.to_csv(os.path.join(data_dir2, 'MESPNGames.csv'), index=False, header=True)


def get_box_scores(ignore_completed=False,
                   season_results_file='MRegularSeasonDetailedResults.csv',
                   espn_games_file='MESPNGames.csv'):
    convert_dict = {
        'Season': int, 'DayNum': int, 'WTeamID': int, 'WScore': int, 'LTeamID': int, 'LScore': int, 'WLoc': str,
        'NumOT': int,
        'WFGM': int, 'WFGA': int, 'WFGM3': int, 'WFGA3': int, 'WFTM': int, 'WFTA': int,
        'WOR': int, 'WDR': int, 'WAst': int, 'WTO': int, 'WStl': int, 'WBlk': int, 'WPF': int,
        'LFGM': int, 'LFGA': int, 'LFGM3': int, 'LFGA3': int, 'LFTM': int, 'LFTA': int,
        'LOR': int, 'LDR': int, 'LAst': int, 'LTO': int, 'LStl': int, 'LBlk': int, 'LPF': int
    }
    start_date = datetime.strptime('11/4/2019', '%m/%d/%Y')
    completed_games = pd.read_csv(os.path.join(data_dir2, 'MCompletedGames.csv'))
    season_results = pd.read_csv(os.path.join(data_dir2, season_results_file))
    games_df = pd.read_csv(os.path.join(data_dir2, espn_games_file))
    teams_df = pd.read_csv(os.path.join(data_dir2, 'MTeamsESPN.csv'))
    driver = webdriver.Chrome('/Users/murphycrosby/opt/chromedriver/chromedriver')
    count = 0
    for index, row in games_df.iterrows():
        found = completed_games[completed_games.GameID == row.GameID]
        if len(found) > 0 and ignore_completed is False:
            continue
        driver.get(f'{box_score_url}?gameId={int(row.GameID)}')
        content = driver.page_source
        soup = BeautifulSoup(content, 'html5lib')
        title = soup.find('head').find('title')
        game_date_str = title.text.split(' - ')[2]
        game_date = datetime.strptime(game_date_str, '%B %d, %Y')
        day_num = int((game_date - start_date).days)

        game_status = soup.find('span', attrs={'class': 'game-time status-detail'})
        ots = game_status.text.split('/')
        if len(ots) > 1:
            ots_str = ots[1].replace('OT', '')
            if ots_str == '':
                ots = 1
            else:
                ots = int(ots_str)
        else:
            ots = 0

        table = soup.find('section', attrs={'id': 'pane-main'})
        away_team_div = table.find('div', attrs={'class': 'team away'})
        away_score = away_team_div.find('div', attrs={'class': 'score'})
        away_team_name = away_team_div.find('a', attrs={'class': 'team-name'})
        if away_team_name is None:
            completed_games = completed_games.append({
                'GameID': row.GameID,
                'Status': 'NoAwayTeam'
            }, ignore_index=True)
            continue
        away_url = away_team_name.attrs['href'].split('/')
        away_team_id = away_url[len(away_url) - 2]

        home_team_div = table.find('div', attrs={'class': 'team home'})
        home_score = home_team_div.find('div', attrs={'class': 'score'})
        home_team_name = home_team_div.find('a', attrs={'class': 'team-name'})
        if home_team_name is None:
            completed_games = completed_games.append({
                'GameID': row.GameID,
                'Status': 'NoHomeTeam'
            }, ignore_index=True)
            continue
        home_url = home_team_name.attrs['href'].split('/')
        home_team_id = home_url[len(home_url) - 2]
        # 0 is away, 1 is home
        if int(home_score.text) > int(away_score.text):
            w_idx = 1
            l_idx = 0
            w_loc = 'H'
            w_team = teams_df[teams_df.ESPNID == int(home_team_id)]
            w_team_id = w_team.TeamID.values[0]
            l_team = teams_df[teams_df.ESPNID == int(away_team_id)]
            l_team_id = l_team.TeamID.values[0]
        else:
            w_idx = 0
            l_idx = 1
            w_loc = 'A'
            l_team = teams_df[teams_df.ESPNID == int(home_team_id)]
            l_team_id = l_team.TeamID.values[0]
            w_team = teams_df[teams_df.ESPNID == int(away_team_id)]
            w_team_id = w_team.TeamID.values[0]

        box_score = table.find('div', attrs={'id': 'gamepackage-box-score'})
        box_scores = box_score.find_all('div', attrs={'class': 'sub-module'})

        w_team_totals = box_scores[w_idx].find('tr', attrs={'class': 'highlight'})
        w_team_fg = w_team_totals.find('td', attrs={'class': 'fg'})
        w_team_3pt = w_team_totals.find('td', attrs={'class': '3pt'})
        w_team_ft = w_team_totals.find('td', attrs={'class': 'ft'})
        w_team_oreb = w_team_totals.find('td', attrs={'class': 'oreb'})
        w_team_dreb = w_team_totals.find('td', attrs={'class': 'dreb'})
        w_team_ast = w_team_totals.find('td', attrs={'class': 'ast'})
        w_team_stl = w_team_totals.find('td', attrs={'class': 'stl'})
        w_team_blk = w_team_totals.find('td', attrs={'class': 'blk'})
        w_team_to = w_team_totals.find('td', attrs={'class': 'to'})
        w_team_pf = w_team_totals.find('td', attrs={'class': 'pf'})
        w_team_pts = w_team_totals.find('td', attrs={'class': 'pts'})
        # 1 is home
        l_team_totals = box_scores[l_idx].find('tr', attrs={'class': 'highlight'})
        l_team_fg = l_team_totals.find('td', attrs={'class': 'fg'})
        l_team_3pt = l_team_totals.find('td', attrs={'class': '3pt'})
        l_team_ft = l_team_totals.find('td', attrs={'class': 'ft'})
        l_team_oreb = l_team_totals.find('td', attrs={'class': 'oreb'})
        l_team_dreb = l_team_totals.find('td', attrs={'class': 'dreb'})
        l_team_ast = l_team_totals.find('td', attrs={'class': 'ast'})
        l_team_stl = l_team_totals.find('td', attrs={'class': 'stl'})
        l_team_blk = l_team_totals.find('td', attrs={'class': 'blk'})
        l_team_to = l_team_totals.find('td', attrs={'class': 'to'})
        l_team_pf = l_team_totals.find('td', attrs={'class': 'pf'})
        l_team_pts = l_team_totals.find('td', attrs={'class': 'pts'})

        if w_team_pts.text == '--' or l_team_pts.text == '--':
            completed_games = completed_games.append({
                'GameID': row.GameID,
                'Status': 'DashesSkipped'
            }, ignore_index=True)
            continue

        df = {
            'Season': 2020,
            'DayNum': day_num,
            'WTeamID': w_team_id,
            'WScore': int(w_team_pts.text),
            'LTeamID': l_team_id,
            'LScore': int(l_team_pts.text),
            'WLoc': w_loc,
            'NumOT': ots,
            'WFGM': w_team_fg.text.split('-')[0],
            'WFGA': w_team_fg.text.split('-')[1],
            'WFGM3': w_team_3pt.text.split('-')[0],
            'WFGA3': w_team_3pt.text.split('-')[1],
            'WFTM': w_team_ft.text.split('-')[0],
            'WFTA': w_team_ft.text.split('-')[1],
            'WOR': w_team_oreb.text,
            'WDR': w_team_dreb.text,
            'WAst': w_team_ast.text,
            'WTO': w_team_to.text,
            'WStl': w_team_stl.text,
            'WBlk': w_team_blk.text,
            'WPF': w_team_pf.text,
            'LFGM': l_team_fg.text.split('-')[0],
            'LFGA': l_team_fg.text.split('-')[1],
            'LFGM3': l_team_3pt.text.split('-')[0],
            'LFGA3': l_team_3pt.text.split('-')[1],
            'LFTM': l_team_ft.text.split('-')[0],
            'LFTA': l_team_ft.text.split('-')[1],
            'LOR': l_team_oreb.text,
            'LDR': l_team_dreb.text,
            'LAst': l_team_ast.text,
            'LTO': l_team_to.text,
            'LStl': l_team_stl.text,
            'LBlk': l_team_blk.text,
            'LPF': l_team_pf.text
        }
        season_results = season_results.append(df, ignore_index=True, sort=False)
        completed_games = completed_games.append({
            'GameID': row.GameID,
            'Status': 'Downloaded'
        }, ignore_index=True)
        count += 1

        if count % 1 == 0:
            season_results = season_results.drop_duplicates()
            season_results = season_results.astype(convert_dict)
            season_results[[
                'Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT',
                'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
                'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR',
                'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
            ]].to_csv(os.path.join(data_dir2, season_results_file), index=False, header=True)

            completed_games = completed_games.drop_duplicates()
            completed_games = completed_games.astype({
                'GameID': int,
                'Status': str
            })
            completed_games.to_csv(os.path.join(data_dir2, 'MCompletedGames.csv'), index=False, header=True)

    season_results = season_results.drop_duplicates()
    season_results = season_results.astype(convert_dict)
    season_results[[
        'Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT',
        'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
        'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR',
        'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
    ]].to_csv(os.path.join(data_dir2, season_results_file), index=False, header=True)

    completed_games = completed_games.drop_duplicates()
    completed_games = completed_games.astype({
        'GameID': int,
        'Status': str
    })
    completed_games.to_csv(os.path.join(data_dir2, 'MCompletedGames.csv'), index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloads BoxScores from ESPN.')
    parser.add_argument('--download', help='Process Historical Data', action='store_true')
    args = parser.parse_args()

    # get_espn_teams()
    # combine_teams_espn()
    # get_team_schedules()
    # get_specific_team_schedules(201)
    get_box_scores()
    # go_download()
