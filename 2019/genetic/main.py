import sys
import os
import logging
import json
import pandas as pd
from tqdm import tqdm
from optimizer import Optimizer


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)


def train_networks(networks, x, y):
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(2, x, y)
        pbar.update(1)
    pbar.close()


def get_average_accuracy(networks):
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def generate(filename, generations, starting_gen, population, nn_param_choices, x, y):
    optimizer = Optimizer(nn_param_choices, population=population)
    logging.info(filename)
    if filename == "" or not os.path.isfile(filename):
        logging.info("***Randomly creating networks***")
        networks = optimizer.create_population(population)
        filename = "gen-0-pre.json"
        with open(filename, 'w') as outfile:
            json.dump([ob.__dict__ for ob in networks], outfile)
    else:
        logging.info("***Loading Evolution File (%s)***" % filename)
        with open(filename) as json_file:
            data = json.load(json_file) 
        networks = optimizer.load_population(data)

        if "-post" in filename:
            networks = optimizer.evolve(networks)
                
    logging.info(' ')

    for i in range(starting_gen, generations):
        logging.info("***Doing generation %d of %d***" % (i + 1, generations))
        logging.info(' ')
        
        filename = "gen-" + str(i + 1) + "-pre.json"
        logging.info("Writing file: %s" % filename)
        with open(filename, 'w') as outfile:
            json.dump([ob.__dict__ for ob in networks], outfile)

        for n in networks:
            n.print_network()

        logging.info(' ')

        train_networks(networks, x, y)

        average_accuracy = get_average_accuracy(networks)

        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        for n in networks:
            n.print_network()
        
        logging.info(' ')
        logging.info('-'*80)
        logging.info(' ')

        filename = "gen-" + str(i + 1) + "-post.json"
        with open(filename, 'w') as outfile:
            json.dump([ob.__dict__ for ob in networks], outfile)

        if i != generations - 1:
            networks = optimizer.evolve(networks)

    networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)

    print_networks(networks[:5])


def print_networks(networks):
    logging.info('-'*80)
    for network in networks:
        network.print_network()


def main():
    filename = ""
    x_filename = "x_train-2004.csv"
    y_filename = "y_train-2004.csv"
    generations = 10
    starting_gen = 0
    population = 20
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    if len(sys.argv) > 2:
        try:
            generations = int(sys.argv[2])
        except ValueError:
            logging.info("Generation is not an integer")
            return
    
    if len(sys.argv) > 3:
        try:
            starting_gen = int(sys.argv[3]) - 1
        except ValueError:
            logging.info("Starting Generation is not an integer")
            return

    if len(sys.argv) > 4:
        try:
            population = int(sys.argv[4])
        except ValueError:
            logging.info("Population is not an integer")
            return

    if not os.path.isfile(x_filename):
        logging.info("File (%s) does not exist." % x_filename)
        return
    if not os.path.isfile(y_filename):
        logging.info("File (%s) does not exist." % y_filename)
        return

    x = pd.read_csv(x_filename)
    y = pd.read_csv(y_filename)
    x = x.values.reshape(y.shape[0], 3, 1860)

    print(x.shape)
    print(y.shape)

    nn_param_choices = {
        'nb_layers': [1, 2, 3, 4],
        'nb_neurons': [4, 8, 16, 32, 64, 128, 256, 512],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'softmax'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Evolving %d generations with population %d***" % (generations, population))

    generate(filename, generations, starting_gen, population, nn_param_choices, x, y)


if __name__ == '__main__':
    main()
