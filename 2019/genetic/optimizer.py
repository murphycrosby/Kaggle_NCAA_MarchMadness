from functools import reduce
from operator import add
import random
from network import Network


class Optimizer:
    def __init__(self, nn_param_choices, population=20, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices
        self.population = population

    def load_population(self, data):
        pop = []
        for n in data:
            new_network = Network(n['nn_param_choices'])
            new_network.network['nb_neurons'] = []
            new_network.network['activation'] = []

            new_network.network['nb_neurons'].extend(n['network']['nb_neurons'])
            new_network.network['nb_layers'] = n['network']['nb_layers']
            new_network.network['activation'].extend(n['network']['activation'])
            new_network.network['optimizer'] = n['network']['optimizer']
            new_network.accuracy = n['accuracy']

            pop.append(new_network)

        for c in pop:
            if not self.check_network(c.network):
                raise ValueError('Network out of balance')

        return pop

    def create_population(self, count):
        pop = []
        for _ in range(0, count):
            network = Network(self.nn_param_choices)
            network.create_random()

            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        return network.accuracy

    def grade(self, pop):
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    @staticmethod
    def check_network(network):
        layer_count = network['nb_layers']

        nb_neuron_count = len(network['nb_neurons'])
        activation_count = len(network['activation'])

        if (layer_count != nb_neuron_count) or (layer_count != activation_count):
            return False

        return True

    def breed(self, mother, father):
        children = []
        for _ in range(2):

            child = {}

            for param in self.nn_param_choices:
                n = random.choice([mother.network, father.network])
                if (param == 'nb_neurons') or (param == 'activation'):
                    child[param] = []
                    child[param].extend(n[param])
                else:
                    child[param] = random.choice(
                        [mother.network[param], father.network[param]]
                    )

            if self.mutate_chance > random.random():
                child = self.mutate(child)

            if child['nb_layers'] > len(child['nb_neurons']):
                for i in range(child['nb_layers'] - len(child['nb_neurons'])):
                    child['nb_neurons'].append(random.choice(self.nn_param_choices['nb_neurons']))
            elif child['nb_layers'] < len(child['nb_neurons']):
                for i in range(len(child['nb_neurons']) - child['nb_layers']):
                    del(child['nb_neurons'][random.randint(0, len(child['nb_neurons'])-1)])

            if child['nb_layers'] > len(child['activation']):
                for i in range(child['nb_layers'] - len(child['activation'])):
                    child['activation'].append(random.choice(self.nn_param_choices['activation']))
            elif child['nb_layers'] < len(child['activation']):
                for i in range(len(child['activation']) - child['nb_layers']):
                    del(child['activation'][random.randint(0, len(child['activation'])-1)])

            network = Network(self.nn_param_choices)
            network.create_set(child)

            children.append(network)

        for n in children:
            if not self.check_network(n.network):
                raise ValueError('Network out of balance')

        return children

    def mutate(self, network):
        mutation = random.choice(list(self.nn_param_choices.keys()))

        if (mutation == 'nb_neurons') or (mutation == 'activation'):
            network[mutation] = []
            for i in range(network['nb_layers']):
                network[mutation].append(random.choice(self.nn_param_choices[mutation]))
        else:
            network[mutation] = random.choice(self.nn_param_choices[mutation])

        return network

    def evolve(self, pop):
        graded = [(self.fitness(network), network) for network in pop]
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        retain_length = int(self.population*self.retain)

        parents = graded[:retain_length]

        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        parents_length = len(parents)
        desired_length = self.population - parents_length
        children = []
        count = 0
        
        while len(children) < desired_length and count < (self.population*2):
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            if male != female:
                male = parents[male]
                female = parents[female]

                babies = self.breed(male, female)

                for baby in babies:
                    if len(children) < desired_length:
                        add_baby = True
                        for p in parents:
                            if p.network == baby.network:
                                add_baby = False
                        for c in children:
                            if c.network == baby.network:
                                add_baby = False
                        if add_baby:
                            children.append(baby)
            
            # Just don't go on forever if we can't get a unique combo
            count = count + 1

        parents.extend(children)

        for p in parents:
            if not self.check_network(p.network):
                raise ValueError('Network out of balance')

        return parents
