from configparser import ConfigParser
from network import *
import copy
from data import *

def read_config_network(file):
    network = neuralNetwork()
    config = ConfigParser()
    config.read(file)
    print(config.sections())
    sections = config.sections()
    for i in range(len(sections)):
        if config.get(sections[i], 'type') == 'global':
            network.add_regulizer(regulizer=config.get(sections[i], 'regulizer'), scaling_factor=config.get(sections[i], 'scaling_factor'))
            network.add_loss_function(loss_function=config.get(sections[i], 'loss'))
        elif config.get(sections[i], 'type') == 'dense':
            network.add_dense_layer(input_size=config.getint(sections[i], 'nodes'), output_size=config.getint(sections[i+1], 'nodes'))
            network.add_activation_layer(activation_function=config.get(sections[i], 'activation'))
        elif config.get(sections[i], 'type') == 'output':
            network.add_softmax_layer(input_size=config.getint(sections[i], 'nodes'))
    return network

def read_config_data(file):
    generator = DataGenerator()
    config = ConfigParser()
    config.read(file)
    settings = config._sections

    globals = settings["globals"]
    globals["dim"], globals["n_symbols"], globals["centering"] = int(globals["dim"]), int(globals["n_symbols"]), int(globals["centering"])
    globals["noise"], globals["train"], globals["valid"], globals["test"] = \
        float(globals["noise"]), float(globals["train"]), float(globals["valid"]), float(globals["test"])

    circle, cross, rectangle, bars = settings["circle"], settings["cross"], settings["rectangle"], settings["horisontal_bars"],
    circle = {k: int(circle[k]) for k in circle}
    cross = {k: int(cross[k]) for k in cross}
    rectangle = {k: int(rectangle[k]) for k in rectangle}
    bars = {k: int(bars[k]) for k in bars}

    data = generator.selfMadeData(globals, circle, cross, rectangle, bars)

    return data


def main():
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = read_config_data("data1.ini")
    print("-----MODEL WITH REGULIZATION------")
    network = read_config_network("network1.ini")
    network.train_model(x_train, y_train, x_valid, y_valid, epochs=15, verbose=True)
    print("---END---")
    network.make_prediction(x_test, y_test)
    print("\n")

if __name__ == '__main__':
    main()

