import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
import time
import itertools as it
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
data_dir = './google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1'
data_dir2 = './google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage2'
model_dir = './model'
training_dir = './training_data'
multithread = False


def get_scaler():
    regular = pd.read_csv(os.path.join(data_dir, 'MRegularSeasonDetailedResults.csv'))
    regular2 = pd.read_csv(os.path.join(data_dir2, 'MRegularSeasonDetailedResults.csv'))
    tourney = pd.read_csv(os.path.join(data_dir, 'MNCAATourneyDetailedResults.csv'))
    full = pd.concat([regular, regular2, tourney])
    full = full.drop_duplicates()
    min_stats = [
        min(full['WFGM'].min(), full['LFGM'].min()),
        min(full['WFGA'].min(), full['LFGA'].min()),
        min(full['WFGM3'].min(), full['LFGM3'].min()),
        min(full['WFGA3'].min(), full['LFGA3'].min()),
        min(full['WFTM'].min(), full['LFTM'].min()),
        min(full['WFTA'].min(), full['LFTA'].min()),
        min(full['WOR'].min(), full['LOR'].min()),
        min(full['WDR'].min(), full['LDR'].min()),
        min(full['WAst'].min(), full['LAst'].min()),
        min(full['WTO'].min(), full['LTO'].min()),
        min(full['WStl'].min(), full['LStl'].min()),
        min(full['WBlk'].min(), full['LBlk'].min()),
        min(full['WPF'].min(), full['LPF'].min())
    ]
    max_stats = [
        max(full['WFGM'].max(), full['LFGM'].max()),
        max(full['WFGA'].max(), full['LFGA'].max()),
        max(full['WFGM3'].max(), full['LFGM3'].max()),
        max(full['WFGA3'].max(), full['LFGA3'].max()),
        max(full['WFTM'].max(), full['LFTM'].max()),
        max(full['WFTA'].max(), full['LFTA'].max()),
        max(full['WOR'].max(), full['LOR'].max()),
        max(full['WDR'].max(), full['LDR'].max()),
        max(full['WAst'].max(), full['LAst'].max()),
        max(full['WTO'].max(), full['LTO'].max()),
        max(full['WStl'].max(), full['LStl'].max()),
        max(full['WBlk'].max(), full['LBlk'].max()),
        max(full['WPF'].max(), full['LPF'].max())
    ]
    print(f'Min: {min_stats}')
    print(f'Max: {max_stats}')
    scaler = MinMaxScaler()
    scaler.fit([min_stats, max_stats])
    return scaler


def load_historical_data(num_of_games, season_year, onehot_season, onehot_daynum, onehot_teams):
    regular = pd.read_csv(os.path.join(data_dir, 'MRegularSeasonDetailedResults.csv'))
    regular2 = pd.read_csv(os.path.join(data_dir2, 'MRegularSeasonDetailedResults.csv'))
    tourney = pd.read_csv(os.path.join(data_dir2, 'MNCAATourneyDetailedResults.csv'))
    full = pd.concat([regular, regular2, tourney])
    full = full.drop_duplicates()

    scaler = get_scaler()

    train = full[(full.Season == season_year)]
    if len(train) == 0:
        return

    print(f'\t    Full: {str(full.shape)}')
    print(f'\tTraining: {str(train.shape)}')

    if multithread:
        threads = 4
        df_split = np.array_split(train, threads)
        args_ = []
        thread_count = 0
        for x in df_split:
            args_.append([thread_count, num_of_games, x, train, onehot_season, onehot_daynum, onehot_teams, scaler])
            thread_count += 1

        print(f'\t-- Starting Pooling --')
        with Pool(processes=threads) as pool:
            df = pool.starmap(parse_stats_data, args_)
        print(f'\t-- Complete Pooling --')
        x_train = []
        y_train = []
        print(f'\t-- Starting Concatenation --')
        for f in df:
            if len(x_train) == 0:
                x_train = f[0]
                y_train = f[1]
            else:
                if len(f[0]) > 0 and len(f[1]) > 0:
                    x_train = np.concatenate((x_train, f[0]), axis=0)
                    y_train = np.concatenate((y_train, f[1]), axis=0)
        print(f'\t-- Complete Concatenation --')
    else:
        x_train, y_train = parse_stats_data(0, num_of_games,
                                            train,
                                            train,  # really only matters when multiprocessing
                                            onehot_season,
                                            onehot_daynum,
                                            onehot_teams,
                                            scaler)

    print(x_train.shape)
    print(y_train.shape)

    df_x = pd.DataFrame(x_train)
    df_y = pd.DataFrame(y_train)

    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
    path = os.path.join(training_dir, f'{num_of_games}')
    if not os.path.exists(path):
        os.mkdir(path)

    # x_train, y_train = parse_stats_data(train, full, onehot_season, onehot_daynum, onehot_teams)
    df_x.to_csv(f'{path}/x_train-{season_year}.csv', index=False)
    df_y.to_csv(f'{path}/y_train-{season_year}.csv', index=False)


def parse_stats_data(thread_count, num_of_games, df, full, onehot_season, onehot_daynum, onehot_teams, stats_scaler):
    # print('parse_data')

    x = []
    y = []
    start_time = time.time()
    for index, row in df.iterrows():
        if index % 100 == 0:
            gig = time.time() - start_time
            print(f'\tThread: {thread_count} Row: {index} Time: {gig} secs')
            start_time = time.time()
        # print(onehot_encoder.transform(np.array(row.WTeamID).reshape(-1, 1)))
        # print(onehot_encoder.transform(np.array(row.LTeamID).reshape(-1, 1)))

        # print('===============')
        # print(' Season: ', row.Season)
        # print(' DayNum: ', row.DayNum)
        # print('WTeamID: ', row.WTeamID)
        # print('LTeamID: ', row.LTeamID)

        w_hist = full[(full.Season == row.Season) & (full.DayNum < row.DayNum) &
                      ((full.WTeamID == row.WTeamID) | (full.LTeamID == row.WTeamID))] \
            .sort_values(by=['DayNum'], ascending=False).head(num_of_games)

        l_hist = full[(full.Season == row.Season) & (full.DayNum < row.DayNum) &
                      ((full.WTeamID == row.LTeamID) | (full.LTeamID == row.LTeamID))] \
            .sort_values(by=['DayNum'], ascending=False).head(num_of_games)

        if len(w_hist) < num_of_games or len(l_hist) < num_of_games:
            continue

        # print(len(w_hist))
        for i in range(num_of_games):
            team1stats = []
            team2stats = []
            team3stats = []
            team4stats = []

            if row.WTeamID < row.LTeamID:
                season1vec = onehot_season.transform(np.array(w_hist.iloc()[i].Season).reshape(-1, 1))
                daynum1vec = onehot_daynum.transform(np.array(w_hist.iloc()[i].DayNum).reshape(-1, 1))

                season2vec = onehot_season.transform(np.array(l_hist.iloc()[i].Season).reshape(-1, 1))
                daynum2vec = onehot_daynum.transform(np.array(l_hist.iloc()[i].DayNum).reshape(-1, 1))

                if row.WTeamID == w_hist.iloc()[i].WTeamID:
                    team1vec = onehot_teams.transform(np.array(w_hist.iloc()[i].WTeamID).reshape(-1, 1))
                    team1stats.append(w_hist.iloc()[i].WFGM)
                    team1stats.append(w_hist.iloc()[i].WFGA)
                    team1stats.append(w_hist.iloc()[i].WFGM3)
                    team1stats.append(w_hist.iloc()[i].WFGA3)
                    team1stats.append(w_hist.iloc()[i].WFTM)
                    team1stats.append(w_hist.iloc()[i].WFTA)
                    team1stats.append(w_hist.iloc()[i].WOR)
                    team1stats.append(w_hist.iloc()[i].WDR)
                    team1stats.append(w_hist.iloc()[i].WAst)
                    team1stats.append(w_hist.iloc()[i].WTO)
                    team1stats.append(w_hist.iloc()[i].WStl)
                    team1stats.append(w_hist.iloc()[i].WBlk)
                    team1stats.append(w_hist.iloc()[i].WPF)

                    team3vec = onehot_teams.transform(np.array(w_hist.iloc()[i].LTeamID).reshape(-1, 1))
                    team3stats.append(w_hist.iloc()[i].LFGM)
                    team3stats.append(w_hist.iloc()[i].LFGA)
                    team3stats.append(w_hist.iloc()[i].LFGM3)
                    team3stats.append(w_hist.iloc()[i].LFGA3)
                    team3stats.append(w_hist.iloc()[i].LFTM)
                    team3stats.append(w_hist.iloc()[i].LFTA)
                    team3stats.append(w_hist.iloc()[i].LOR)
                    team3stats.append(w_hist.iloc()[i].LDR)
                    team3stats.append(w_hist.iloc()[i].LAst)
                    team3stats.append(w_hist.iloc()[i].LTO)
                    team3stats.append(w_hist.iloc()[i].LStl)
                    team3stats.append(w_hist.iloc()[i].LBlk)
                    team3stats.append(w_hist.iloc()[i].LPF)
                else:
                    team1vec = onehot_teams.transform(np.array(w_hist.iloc()[i].LTeamID).reshape(-1, 1))
                    team1stats.append(w_hist.iloc()[i].LFGM)
                    team1stats.append(w_hist.iloc()[i].LFGA)
                    team1stats.append(w_hist.iloc()[i].LFGM3)
                    team1stats.append(w_hist.iloc()[i].LFGA3)
                    team1stats.append(w_hist.iloc()[i].LFTM)
                    team1stats.append(w_hist.iloc()[i].LFTA)
                    team1stats.append(w_hist.iloc()[i].LOR)
                    team1stats.append(w_hist.iloc()[i].LDR)
                    team1stats.append(w_hist.iloc()[i].LAst)
                    team1stats.append(w_hist.iloc()[i].LTO)
                    team1stats.append(w_hist.iloc()[i].LStl)
                    team1stats.append(w_hist.iloc()[i].LBlk)
                    team1stats.append(w_hist.iloc()[i].LPF)

                    team3vec = onehot_teams.transform(np.array(w_hist.iloc()[i].WTeamID).reshape(-1, 1))
                    team3stats.append(w_hist.iloc()[i].WFGM)
                    team3stats.append(w_hist.iloc()[i].WFGA)
                    team3stats.append(w_hist.iloc()[i].WFGM3)
                    team3stats.append(w_hist.iloc()[i].WFGA3)
                    team3stats.append(w_hist.iloc()[i].WFTM)
                    team3stats.append(w_hist.iloc()[i].WFTA)
                    team3stats.append(w_hist.iloc()[i].WOR)
                    team3stats.append(w_hist.iloc()[i].WDR)
                    team3stats.append(w_hist.iloc()[i].WAst)
                    team3stats.append(w_hist.iloc()[i].WTO)
                    team3stats.append(w_hist.iloc()[i].WStl)
                    team3stats.append(w_hist.iloc()[i].WBlk)
                    team3stats.append(w_hist.iloc()[i].WPF)

                if row.LTeamID == l_hist.iloc()[i].WTeamID:
                    team2vec = onehot_teams.transform(np.array(l_hist.iloc()[i].WTeamID).reshape(-1, 1))
                    team2stats.append(l_hist.iloc()[i].WFGM)
                    team2stats.append(l_hist.iloc()[i].WFGA)
                    team2stats.append(l_hist.iloc()[i].WFGM3)
                    team2stats.append(l_hist.iloc()[i].WFGA3)
                    team2stats.append(l_hist.iloc()[i].WFTM)
                    team2stats.append(l_hist.iloc()[i].WFTA)
                    team2stats.append(l_hist.iloc()[i].WOR)
                    team2stats.append(l_hist.iloc()[i].WDR)
                    team2stats.append(l_hist.iloc()[i].WAst)
                    team2stats.append(l_hist.iloc()[i].WTO)
                    team2stats.append(l_hist.iloc()[i].WStl)
                    team2stats.append(l_hist.iloc()[i].WBlk)
                    team2stats.append(l_hist.iloc()[i].WPF)

                    team4vec = onehot_teams.transform(np.array(l_hist.iloc()[i].LTeamID).reshape(-1, 1))
                    team4stats.append(l_hist.iloc()[i].LFGM)
                    team4stats.append(l_hist.iloc()[i].LFGA)
                    team4stats.append(l_hist.iloc()[i].LFGM3)
                    team4stats.append(l_hist.iloc()[i].LFGA3)
                    team4stats.append(l_hist.iloc()[i].LFTM)
                    team4stats.append(l_hist.iloc()[i].LFTA)
                    team4stats.append(l_hist.iloc()[i].LOR)
                    team4stats.append(l_hist.iloc()[i].LDR)
                    team4stats.append(l_hist.iloc()[i].LAst)
                    team4stats.append(l_hist.iloc()[i].LTO)
                    team4stats.append(l_hist.iloc()[i].LStl)
                    team4stats.append(l_hist.iloc()[i].LBlk)
                    team4stats.append(l_hist.iloc()[i].LPF)

                else:
                    team2vec = onehot_teams.transform(np.array(l_hist.iloc()[i].LTeamID).reshape(-1, 1))
                    team2stats.append(l_hist.iloc()[i].LFGM)
                    team2stats.append(l_hist.iloc()[i].LFGA)
                    team2stats.append(l_hist.iloc()[i].LFGM3)
                    team2stats.append(l_hist.iloc()[i].LFGA3)
                    team2stats.append(l_hist.iloc()[i].LFTM)
                    team2stats.append(l_hist.iloc()[i].LFTA)
                    team2stats.append(l_hist.iloc()[i].LOR)
                    team2stats.append(l_hist.iloc()[i].LDR)
                    team2stats.append(l_hist.iloc()[i].LAst)
                    team2stats.append(l_hist.iloc()[i].LTO)
                    team2stats.append(l_hist.iloc()[i].LStl)
                    team2stats.append(l_hist.iloc()[i].LBlk)
                    team2stats.append(l_hist.iloc()[i].LPF)

                    team4vec = onehot_teams.transform(np.array(l_hist.iloc()[i].WTeamID).reshape(-1, 1))
                    team4stats.append(l_hist.iloc()[i].WFGM)
                    team4stats.append(l_hist.iloc()[i].WFGA)
                    team4stats.append(l_hist.iloc()[i].WFGM3)
                    team4stats.append(l_hist.iloc()[i].WFGA3)
                    team4stats.append(l_hist.iloc()[i].WFTM)
                    team4stats.append(l_hist.iloc()[i].WFTA)
                    team4stats.append(l_hist.iloc()[i].WOR)
                    team4stats.append(l_hist.iloc()[i].WDR)
                    team4stats.append(l_hist.iloc()[i].WAst)
                    team4stats.append(l_hist.iloc()[i].WTO)
                    team4stats.append(l_hist.iloc()[i].WStl)
                    team4stats.append(l_hist.iloc()[i].WBlk)
                    team4stats.append(l_hist.iloc()[i].WPF)

            else:
                season1vec = onehot_season.transform(np.array(l_hist.iloc()[i].Season).reshape(-1, 1))
                daynum1vec = onehot_daynum.transform(np.array(l_hist.iloc()[i].DayNum).reshape(-1, 1))

                season2vec = onehot_season.transform(np.array(w_hist.iloc()[i].Season).reshape(-1, 1))
                daynum2vec = onehot_daynum.transform(np.array(w_hist.iloc()[i].DayNum).reshape(-1, 1))

                if row.LTeamID == l_hist.iloc()[i].WTeamID:
                    team1vec = onehot_teams.transform(np.array(l_hist.iloc()[i].WTeamID).reshape(-1, 1))
                    team1stats.append(l_hist.iloc()[i].WFGM)
                    team1stats.append(l_hist.iloc()[i].WFGA)
                    team1stats.append(l_hist.iloc()[i].WFGM3)
                    team1stats.append(l_hist.iloc()[i].WFGA3)
                    team1stats.append(l_hist.iloc()[i].WFTM)
                    team1stats.append(l_hist.iloc()[i].WFTA)
                    team1stats.append(l_hist.iloc()[i].WOR)
                    team1stats.append(l_hist.iloc()[i].WDR)
                    team1stats.append(l_hist.iloc()[i].WAst)
                    team1stats.append(l_hist.iloc()[i].WTO)
                    team1stats.append(l_hist.iloc()[i].WStl)
                    team1stats.append(l_hist.iloc()[i].WBlk)
                    team1stats.append(l_hist.iloc()[i].WPF)

                    team3vec = onehot_teams.transform(np.array(l_hist.iloc()[i].LTeamID).reshape(-1, 1))
                    team3stats.append(l_hist.iloc()[i].LFGM)
                    team3stats.append(l_hist.iloc()[i].LFGA)
                    team3stats.append(l_hist.iloc()[i].LFGM3)
                    team3stats.append(l_hist.iloc()[i].LFGA3)
                    team3stats.append(l_hist.iloc()[i].LFTM)
                    team3stats.append(l_hist.iloc()[i].LFTA)
                    team3stats.append(l_hist.iloc()[i].LOR)
                    team3stats.append(l_hist.iloc()[i].LDR)
                    team3stats.append(l_hist.iloc()[i].LAst)
                    team3stats.append(l_hist.iloc()[i].LTO)
                    team3stats.append(l_hist.iloc()[i].LStl)
                    team3stats.append(l_hist.iloc()[i].LBlk)
                    team3stats.append(l_hist.iloc()[i].LPF)
                else:
                    team1vec = onehot_teams.transform(np.array(l_hist.iloc()[i].LTeamID).reshape(-1, 1))
                    team1stats.append(l_hist.iloc()[i].LFGM)
                    team1stats.append(l_hist.iloc()[i].LFGA)
                    team1stats.append(l_hist.iloc()[i].LFGM3)
                    team1stats.append(l_hist.iloc()[i].LFGA3)
                    team1stats.append(l_hist.iloc()[i].LFTM)
                    team1stats.append(l_hist.iloc()[i].LFTA)
                    team1stats.append(l_hist.iloc()[i].LOR)
                    team1stats.append(l_hist.iloc()[i].LDR)
                    team1stats.append(l_hist.iloc()[i].LAst)
                    team1stats.append(l_hist.iloc()[i].LTO)
                    team1stats.append(l_hist.iloc()[i].LStl)
                    team1stats.append(l_hist.iloc()[i].LBlk)
                    team1stats.append(l_hist.iloc()[i].LPF)

                    team3vec = onehot_teams.transform(np.array(l_hist.iloc()[i].WTeamID).reshape(-1, 1))
                    team3stats.append(l_hist.iloc()[i].WFGM)
                    team3stats.append(l_hist.iloc()[i].WFGA)
                    team3stats.append(l_hist.iloc()[i].WFGM3)
                    team3stats.append(l_hist.iloc()[i].WFGA3)
                    team3stats.append(l_hist.iloc()[i].WFTM)
                    team3stats.append(l_hist.iloc()[i].WFTA)
                    team3stats.append(l_hist.iloc()[i].WOR)
                    team3stats.append(l_hist.iloc()[i].WDR)
                    team3stats.append(l_hist.iloc()[i].WAst)
                    team3stats.append(l_hist.iloc()[i].WTO)
                    team3stats.append(l_hist.iloc()[i].WStl)
                    team3stats.append(l_hist.iloc()[i].WBlk)
                    team3stats.append(l_hist.iloc()[i].WPF)

                if row.WTeamID == w_hist.iloc()[i].WTeamID:
                    team2vec = onehot_teams.transform(np.array(w_hist.iloc()[i].WTeamID).reshape(-1, 1))
                    team2stats.append(w_hist.iloc()[i].WFGM)
                    team2stats.append(w_hist.iloc()[i].WFGA)
                    team2stats.append(w_hist.iloc()[i].WFGM3)
                    team2stats.append(w_hist.iloc()[i].WFGA3)
                    team2stats.append(w_hist.iloc()[i].WFTM)
                    team2stats.append(w_hist.iloc()[i].WFTA)
                    team2stats.append(w_hist.iloc()[i].WOR)
                    team2stats.append(w_hist.iloc()[i].WDR)
                    team2stats.append(w_hist.iloc()[i].WAst)
                    team2stats.append(w_hist.iloc()[i].WTO)
                    team2stats.append(w_hist.iloc()[i].WStl)
                    team2stats.append(w_hist.iloc()[i].WBlk)
                    team2stats.append(w_hist.iloc()[i].WPF)

                    team4vec = onehot_teams.transform(np.array(w_hist.iloc()[i].LTeamID).reshape(-1, 1))
                    team4stats.append(w_hist.iloc()[i].LFGM)
                    team4stats.append(w_hist.iloc()[i].LFGA)
                    team4stats.append(w_hist.iloc()[i].LFGM3)
                    team4stats.append(w_hist.iloc()[i].LFGA3)
                    team4stats.append(w_hist.iloc()[i].LFTM)
                    team4stats.append(w_hist.iloc()[i].LFTA)
                    team4stats.append(w_hist.iloc()[i].LOR)
                    team4stats.append(w_hist.iloc()[i].LDR)
                    team4stats.append(w_hist.iloc()[i].LAst)
                    team4stats.append(w_hist.iloc()[i].LTO)
                    team4stats.append(w_hist.iloc()[i].LStl)
                    team4stats.append(w_hist.iloc()[i].LBlk)
                    team4stats.append(w_hist.iloc()[i].LPF)
                else:
                    team2vec = onehot_teams.transform(np.array(w_hist.iloc()[i].LTeamID).reshape(-1, 1))
                    team2stats.append(w_hist.iloc()[i].LFGM)
                    team2stats.append(w_hist.iloc()[i].LFGA)
                    team2stats.append(w_hist.iloc()[i].LFGM3)
                    team2stats.append(w_hist.iloc()[i].LFGA3)
                    team2stats.append(w_hist.iloc()[i].LFTM)
                    team2stats.append(w_hist.iloc()[i].LFTA)
                    team2stats.append(w_hist.iloc()[i].LOR)
                    team2stats.append(w_hist.iloc()[i].LDR)
                    team2stats.append(w_hist.iloc()[i].LAst)
                    team2stats.append(w_hist.iloc()[i].LTO)
                    team2stats.append(w_hist.iloc()[i].LStl)
                    team2stats.append(w_hist.iloc()[i].LBlk)
                    team2stats.append(w_hist.iloc()[i].LPF)

                    team4vec = onehot_teams.transform(np.array(w_hist.iloc()[i].WTeamID).reshape(-1, 1))
                    team4stats.append(w_hist.iloc()[i].WFGM)
                    team4stats.append(w_hist.iloc()[i].WFGA)
                    team4stats.append(w_hist.iloc()[i].WFGM3)
                    team4stats.append(w_hist.iloc()[i].WFGA3)
                    team4stats.append(w_hist.iloc()[i].WFTM)
                    team4stats.append(w_hist.iloc()[i].WFTA)
                    team4stats.append(w_hist.iloc()[i].WOR)
                    team4stats.append(w_hist.iloc()[i].WDR)
                    team4stats.append(w_hist.iloc()[i].WAst)
                    team4stats.append(w_hist.iloc()[i].WTO)
                    team4stats.append(w_hist.iloc()[i].WStl)
                    team4stats.append(w_hist.iloc()[i].WBlk)
                    team4stats.append(w_hist.iloc()[i].WPF)

            team1stats = stats_scaler.transform([team1stats]).reshape(13, )
            team2stats = stats_scaler.transform([team2stats]).reshape(13, )
            team3stats = stats_scaler.transform([team3stats]).reshape(13, )
            team4stats = stats_scaler.transform([team4stats]).reshape(13, )

            x_1 = np.concatenate((
                season1vec, daynum1vec,
                team1vec, np.array(team1stats),
                team3vec, np.array(team3stats),
                daynum2vec,
                team2vec, np.array(team2stats),
                team4vec, np.array(team4stats)
            ), axis=None)
            x_1 = x_1.reshape(1, len(x_1))

            if len(x) == 0:
                x = x_1
            else:
                x = np.concatenate((x, x_1), axis=0)

        # print(x.shape)

        if row.WTeamID < row.LTeamID:
            y_1 = [1, 0]
        # y_1 = [row.WFGM, row.WFGA, row.WFGM3, row.WFGA3, row.WFTM, row.WFTA, row.WOR, row.WDR, row.WAst, row.WTO,
        #        row.WStl, row.WBlk, row.WPF, row.LFGM, row.LFGA, row.LFGM3, row.LFGA3, row.LFTM, row.LFTA, row.LOR,
        #        row.LDR, row.LAst, row.LTO, row.LStl, row.LBlk, row.LPF]
        else:
            y_1 = [0, 1]
        # y_1 = [row.LFGM, row.LFGA, row.LFGM3, row.LFGA3, row.LFTM, row.LFTA, row.LOR, row.LDR, row.LAst, row.LTO,
        #        row.LStl, row.LBlk, row.LPF, row.WFGM, row.WFGA, row.WFGM3, row.WFGA3, row.WFTM, row.WFTA, row.WOR,
        #        row.WDR, row.WAst, row.WTO, row.WStl, row.WBlk, row.WPF]

        y_1 = np.array(y_1)
        # y_1 = y_1.reshape(1, 26)
        y_1 = y_1.reshape(1, 2)

        if len(y) == 0:
            y = y_1
        else:
            y = np.concatenate((y, y_1), axis=0)

    return x, y


def predict_stage(num_of_games):
    model_name = os.path.join(model_dir, 'ncaa_model.h5')
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        return

    season = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
              2014, 2015, 2016, 2017, 2018, 2019, 2020]
    season_df = pd.DataFrame(season, columns=['Season'])
    onehot_season = OneHotEncoder(sparse=False, categories='auto')
    onehot_season.fit(season_df.Season.values.reshape(-1, 1))

    days = []
    for i in range(155):
        days.append(i)

    daynum_df = pd.DataFrame(days, columns=['DayNum'])
    onehot_daynum = OneHotEncoder(sparse=False, categories='auto')
    onehot_daynum.fit(daynum_df.DayNum.values.reshape(-1, 1))

    teams_df = pd.read_csv(os.path.join(data_dir, 'MTeams.csv'))
    onehot_teams = OneHotEncoder(sparse=False, categories='auto')
    onehot_teams.fit(teams_df.TeamID.values.reshape(-1, 1))

    regular = pd.read_csv(os.path.join(data_dir, 'MRegularSeasonDetailedResults.csv'))
    regular2 = pd.read_csv(os.path.join(data_dir2, 'MRegularSeasonDetailedResults.csv'))
    tourney = pd.read_csv(os.path.join(data_dir2, 'MNCAATourneyDetailedResults.csv'))
    full = pd.concat([regular, regular2, tourney])
    full = full.drop_duplicates()

    seeds = pd.read_csv(os.path.join(data_dir2, 'MNCAATourneySeeds.csv'))
    scaler = get_scaler()

    preds = []
    preds2 = []

    for season in range(2020, 2021):
        seeds_p = seeds[(seeds.Season == season)].sort_values(by=['TeamID'])

        teams = seeds_p.TeamID.unique()

        combos = it.combinations(teams, 2)
        matchups = np.array(list(combos))

        for m in matchups:
            if m[0] < m[1]:
                team1 = teams_df[(teams_df.TeamID == m[0])]
                team2 = teams_df[(teams_df.TeamID == m[1])]
            else:
                team1 = teams_df[(teams_df.TeamID == m[1])]
                team2 = teams_df[(teams_df.TeamID == m[0])]

            arr = np.array([[season, 134, m[0], 0, m[1], 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0]])
            df = pd.DataFrame(arr, columns=tourney.columns)

            x, y = parse_stats_data(0, num_of_games, df, full, onehot_season, onehot_daynum, onehot_teams, scaler)
            x = x.reshape((1, num_of_games, 1848))

            prediction = model.predict(x)

            print(f'{season} - {team1.TeamName.values[0]}-{team1.TeamID.values[0]} ({prediction[0][0]:.2f}) vs. '
                  f'{team2.TeamName.values[0]}-{team2.TeamID.values[0]} ({prediction[0][1]:.2f})')

            # 2012_1112_1181,0.47
            s = f'{season}_{team1.TeamID.values[0]}_{team2.TeamID.values[0]}'
            s2 = f'{season}_{team1.TeamName.values[0]}_{team2.TeamName.values[0]}'

            frame = pd.DataFrame(np.array([[s, round(prediction[0][0], 2)]]), columns=['ID', 'Pred'])
            frame2 = pd.DataFrame(np.array([[s2, round(prediction[0][0], 2), round(prediction[0][1], 2)]]),
                                  columns=['info', 'prediction1', 'prediction2'])

            if len(preds) == 0:
                preds = frame
                preds2 = frame2
            else:
                preds = pd.concat([preds, frame])
                preds2 = pd.concat([preds2, frame2])

            print('--------------')

    if not os.path.exists('./submission_files'):
        os.mkdir('./submission_files')

    print(f'Saving Submission: {preds.shape}')
    preds.to_csv('./submission_files/submission_stage.csv', index=False)

    print(f'Saving Full Submission: {preds2.shape}')
    preds2.to_csv('./submission_files/submission_stage_full.csv', index=False)


def train_model(epochs, num_of_games, start_season, end_season):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = os.path.join(model_dir, 'ncaa_model.h5')
    if os.path.exists(model_name):
        print(f'Loading Model')
        model = load_model(model_name)
    else:
        print(f'Creating new Model')
        model = Sequential()
        # model.add(LSTM(32, input_shape=(3, 1860), activation='sigmoid', return_sequences=True))
        # model.add(LSTM(128, activation='sigmoid', return_sequences=True))
        # model.add(LSTM(8, activation='softmax', return_sequences=False))
        # model.add(Dense(2, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        #
        model.add(LSTM(64, input_shape=(num_of_games, 1848), activation='softmax', return_sequences=True))
        model.add(LSTM(32, activation='tanh', return_sequences=True))
        model.add(LSTM(128, activation='elu', return_sequences=False))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

        print(model.summary())
        model.save(model_name)

    x = []
    y = []
    index = 0
    for season in range(start_season, end_season + 1):
        file_x = os.path.join(training_dir, f'{num_of_games}/x_train-{season}.csv')
        file_y = os.path.join(training_dir, f'{num_of_games}/y_train-{season}.csv')
        if not os.path.exists(file_x) or \
                not os.path.exists(file_y):
            continue
        a = pd.read_csv(file_x)
        b = pd.read_csv(file_y)
        if len(a) == 0 or len(b) == 0:
            continue
        if index == 0:
            index = 1
            x = a
            y = b
        else:
            x = pd.concat([x, a], sort=True)
            y = pd.concat([y, b], sort=True)
        print(f'{season}: {x.shape}')

    x = x.values.reshape(y.shape[0], num_of_games, x.shape[1])
    print(f'X Shape: {str(x.shape)}')
    print(f'Y Shape: {str(y.shape)}')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    es = EarlyStopping(monitor='acc', mode='auto', verbose=1, patience=5)
    mc = ModelCheckpoint(model_name, monitor='acc', verbose=1, save_best_only=True, mode='auto', period=5)
    model.fit(x_train, y_train, epochs=epochs, batch_size=20, verbose=1, callbacks=[es, mc])

    score = model.evaluate(x_test, y_test)
    print('=========================')
    print(f'Accuracy: {score[1]:.3f}')
    print('=========================')

    model.save(model_name)

    return


def process_historical(num_of_games):
    season = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
              2014, 2015, 2016, 2017, 2018, 2019, 2020]
    season_df = pd.DataFrame(season, columns=['Season'])
    onehot_season = OneHotEncoder(sparse=False, categories='auto')
    onehot_season.fit(season_df.Season.values.reshape(-1, 1))

    days = []
    for i in range(155):
        days.append(i)

    daynum_df = pd.DataFrame(days, columns=['DayNum'])
    onehot_daynum = OneHotEncoder(sparse=False, categories='auto')
    onehot_daynum.fit(daynum_df.DayNum.values.reshape(-1, 1))

    teams = pd.read_csv(os.path.join(data_dir, 'MTeams.csv'))
    onehot_teams = OneHotEncoder(sparse=False, categories='auto')
    onehot_teams.fit(teams.TeamID.values.reshape(-1, 1))

    for season in range(2020, 2021):
        print(f'Processing Season {season}')
        load_historical_data(num_of_games, season, onehot_season, onehot_daynum, onehot_teams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts Data and Predicts Mens College Basketball games.')
    parser.add_argument('--num', default=5)
    parser.add_argument('--history', help='Process Historical Data', action='store_true')
    parser.add_argument('--train', help='Train Model', action='store_true')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--start_season', default=2003, type=int)
    parser.add_argument('--end_season', default=2020, type=int)
    parser.add_argument('--predict', help='Predict future games', action='store_true')
    args = parser.parse_args()

    number_of_games = args.num

    if args.history is True:
        process_historical(number_of_games)
    if args.train is True:
        train_model(args.epochs, number_of_games, args.start_season, args.end_season)
    if args.predict is True:
        predict_stage(number_of_games)
