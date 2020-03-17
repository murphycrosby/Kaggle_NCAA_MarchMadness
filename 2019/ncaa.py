import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
from pathlib import Path
import itertools as it
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
data_dir = './mens-machine-learning-competition-2019/Stage2DataFiles'
multithread = False


def load_historical_data(num_of_games, season_year, onehot_season, onehot_daynum, onehot_teams):
	regular = pd.read_csv(os.path.join(data_dir, 'RegularSeasonDetailedResults.csv'))
	tourney = pd.read_csv(os.path.join(data_dir, 'NCAATourneyDetailedResults.csv'))

	print(tourney.columns)
	# tourney_b = tourney[(tourney.Season < 2014)]
	# data = tourney[(tourney.Season >= 2014)]

	full = pd.concat([regular, tourney])
	# train = pd.concat([regular, tourney_b])

	train = full[(full.Season == season_year)]
	if len(train) == 0:
		return

	print(train.shape)

	print('    Full: %s' % str(full.shape))
	print('Training: %s' % str(train.shape))

	if multithread:
		df_split = np.array_split(train, 100)
		args = []
		for x in df_split:
			args.append([num_of_games, x, full, onehot_season, onehot_daynum, onehot_teams])

		with Pool(processes=8) as pool:
			df = pool.starmap(parse_stats_data, args)

		x_train = []
		y_train = []
		for f in df:
			if len(x_train) == 0:
				x_train = f[0]
				y_train = f[1]
			else:
				if len(f[0]) > 0 and len(f[1]) > 0:
					x_train = np.concatenate((x_train, f[0]), axis=0)
					y_train = np.concatenate((y_train, f[1]), axis=0)
	else:
		x_train, y_train = parse_stats_data(num_of_games, train, full, onehot_season, onehot_daynum, onehot_teams)

	print(x_train.shape)
	print(y_train.shape)

	df_x = pd.DataFrame(x_train)
	df_y = pd.DataFrame(y_train)

	print(df_x.shape)
	print(df_y.shape)

	path = './training_data/%i/' % num_of_games
	if not os.path.exists(path):
		os.mkdir(path)

	# x_train, y_train = parse_stats_data(train, full, onehot_season, onehot_daynum, onehot_teams)
	df_x.to_csv(('%s/x_train-%i.csv' % (path, season_year)), index=False)
	df_y.to_csv(('%s/y_train-%i.csv' % (path, season_year)), index=False)


def parse_stats_data(num_of_games, df, full, onehot_season, onehot_daynum, onehot_teams):
	# print('parse_data')

	x = []
	y = []

	for index, row in df.iterrows():
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
			# print('- Winner History (%s) -' % str(row.WTeamID))
			# print(' DayNum: ', w_hist.iloc()[i].DayNum)
			# print('WTeamID: ', w_hist.iloc()[i].WTeamID)
			# print('LTeamID: ', w_hist.iloc()[i].LTeamID)

			# print('- Loser History (%s) -' % str(row.LTeamID))
			# print(' DayNum: ', l_hist.iloc()[i].DayNum)
			# print('WTeamID: ', l_hist.iloc()[i].WTeamID)
			# print('LTeamID: ', l_hist.iloc()[i].LTeamID)

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

			x_1 = np.concatenate((
				season1vec, daynum1vec,
				team1vec, np.array(team1stats),
				team3vec, np.array(team3stats),
				season2vec, daynum2vec,
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

	if len(x) == 0 or len(y) == 0:
		return x, y

	# print(x.shape)
	# print(y.shape)

	return x, y


def predict_stage(num_of_games):
	model_name = './models/ncaa_model.h5'
	model_file = Path(model_name)
	if model_file.exists():
		model = load_model(model_name)
	else:
		return

	season = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
	          2014, 2015, 2016, 2017, 2018, 2019]
	season_df = pd.DataFrame(season, columns=['Season'])
	onehot_season = OneHotEncoder(sparse=False, categories='auto')
	onehot_season.fit(season_df.Season.values.reshape(-1, 1))

	days = []
	for i in range(155):
		days.append(i)

	daynum_df = pd.DataFrame(days, columns=['DayNum'])
	onehot_daynum = OneHotEncoder(sparse=False, categories='auto')
	onehot_daynum.fit(daynum_df.DayNum.values.reshape(-1, 1))

	teams_df = pd.read_csv(os.path.join(data_dir, 'Teams.csv'))
	onehot_teams = OneHotEncoder(sparse=False, categories='auto')
	onehot_teams.fit(teams_df.TeamID.values.reshape(-1, 1))

	regular = pd.read_csv(os.path.join(data_dir, 'RegularSeasonDetailedResults.csv'))
	tourney = pd.read_csv(os.path.join(data_dir, 'NCAATourneyDetailedResults.csv'))
	full = pd.concat([regular, tourney])

	seeds = pd.read_csv(os.path.join(data_dir, 'NCAATourneySeeds.csv'))

	preds = []
	preds2 = []

	for season in range(2019, 2020):
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

			x, y = parse_stats_data(num_of_games, df, full, onehot_season, onehot_daynum, onehot_teams)
			x = x.reshape(1, num_of_games, 1860)

			prediction = model.predict(x)

			print('%i - %s-%i (%.2f) vs. %s-%i (%.2f)' %
			      (season, team1.TeamName.values[0], team1.TeamID.values[0], prediction[0][0],
			       team2.TeamName.values[0], team2.TeamID.values[0], prediction[0][1]))

			# 2012_1112_1181,0.47
			s = '%i_%i_%i' % (season, team1.TeamID.values[0], team2.TeamID.values[0])
			s2 = '%i_%s_%s' % (season, team1.TeamName.values[0], team2.TeamName.values[0])

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

		print(preds.shape)

	if not os.path.exists('./submission_files'):
		os.mkdir('./submission_files')

	print(preds.shape)
	preds.to_csv('./submission_files/submission_stage.csv', index=False)

	print(preds2.shape)
	preds2.to_csv('./submission_files/submission_stage_full.csv', index=False)


def train_model(num_of_games):
	model_name = './models/ncaa_model.h5'
	model_file = Path(model_name)
	if model_file.exists():
		model = load_model(model_name)
	else:
		model = Sequential()
		# model.add(LSTM(32, input_shape=(3, 1860), activation='sigmoid', return_sequences=True))
		# model.add(LSTM(128, activation='sigmoid', return_sequences=True))
		# model.add(LSTM(8, activation='softmax', return_sequences=False))
		# model.add(Dense(2, activation='softmax'))
		# model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
		#
		model.add(LSTM(64, input_shape=(num_of_games, 1860), activation='softmax', return_sequences=True))
		model.add(LSTM(32, activation='tanh', return_sequences=True))
		model.add(LSTM(128, activation='elu', return_sequences=False))
		model.add(Dense(2, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

		print(model.summary())
		model.save(model_name)

	x = []
	y = []
	index = 0
	for season in range(2003, 2020):
		a = pd.read_csv('./training_data/%i/x_train-%i.csv' % (num_of_games, season))
		b = pd.read_csv('./training_data/%i/y_train-%i.csv' % (num_of_games, season))
		if len(a) == 0 or len(b) == 0:
			continue
		if index == 0:
			index = 1
			x = a
			y = b
		else:
			x = pd.concat([x, a])
			y = pd.concat([y, b])
		print('%i: %s' % (season, x.shape))

	x = x.values.reshape(y.shape[0], num_of_games, 1860)
	print('X Shape: %s' % str(x.shape))
	print('Y Shape: %s' % str(y.shape))

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

	es = EarlyStopping(monitor='acc', mode='auto', verbose=1, patience=5)
	mc = ModelCheckpoint(model_name, monitor='acc', verbose=1, save_best_only=True, mode='auto', period=5)
	model.fit(x_train, y_train, epochs=50, batch_size=20, verbose=1, callbacks=[es, mc])

	score = model.evaluate(x_test, y_test)
	print('=========================')
	print('Accuracy: %.3f' % score[1])
	print('=========================')

	model.save(model_name)

	return


def process_historical(num_of_games):
	season = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
	          2014, 2015, 2016, 2017, 2018, 2019]
	season_df = pd.DataFrame(season, columns=['Season'])
	onehot_season = OneHotEncoder(sparse=False, categories='auto')
	onehot_season.fit(season_df.Season.values.reshape(-1, 1))

	days = []
	for i in range(155):
		days.append(i)

	daynum_df = pd.DataFrame(days, columns=['DayNum'])
	onehot_daynum = OneHotEncoder(sparse=False, categories='auto')
	onehot_daynum.fit(daynum_df.DayNum.values.reshape(-1, 1))

	teams = pd.read_csv(os.path.join(data_dir, 'Teams.csv'))
	onehot_teams = OneHotEncoder(sparse=False, categories='auto')
	onehot_teams.fit(teams.TeamID.values.reshape(-1, 1))

	for season in range(2003, 2020):
		print('Processing Season %i' % season)
		load_historical_data(num_of_games, season, onehot_season, onehot_daynum, onehot_teams)


def main():
	num_of_games = 5

	process_historical(num_of_games)
	train_model(num_of_games)
	predict_stage(num_of_games)


# {'nb_neurons': [32, 128, 8], 'activation': ['sigmoid', 'sigmoid', 'softmax'],
# 'nb_layers': 3, 'optimizer': 'adagrad'}


if __name__ == '__main__':
	main()
