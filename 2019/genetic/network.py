import random
import logging
from train import train_and_score


class Network:
    def __init__(self, nn_param_choices=None):
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}

    def create_random(self):
        self.network['nb_layers'] = random.choice(self.nn_param_choices['nb_layers'])
        self.network['optimizer'] = random.choice(self.nn_param_choices['optimizer'])
        self.network['nb_neurons'] = []
        self.network['activation'] = []
        for i in range(self.network['nb_layers']):
            self.network['nb_neurons'].append(random.choice(self.nn_param_choices['nb_neurons']))
            self.network['activation'].append(random.choice(self.nn_param_choices['activation']))

    def create_set(self, network):
        self.network['nb_layers'] = network['nb_layers']
        self.network['optimizer'] = network['optimizer']
        self.network['nb_neurons'] = []
        self.network['nb_neurons'].extend(network['nb_neurons'])
        self.network['activation'] = []
        self.network['activation'].extend(network['activation'])

    def train(self, nb_classes, x, y):
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network, nb_classes, x, y)

    def print_network(self):
        logging.info("%s - Accuracy: %.2f" % (self.network, self.accuracy))
