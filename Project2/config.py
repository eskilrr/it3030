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
            #network.add_loss_function(loss_function=config.get(sections[i], 'loss'))
        elif config.get(sections[i], 'type') == 'dense':
            network.add_dense_layer(input_size=None, output_size=config.getint(sections[i], 'output_size'))
        elif config.get(sections[i], 'type') == 'output':
            network.add_softmax_layer(input_size=config.getint(sections[i], 'nodes'))
        elif config.get(sections[i], 'type') == 'conv':
            network.add_conv_layer(filter_size=config.getint(sections[i],'filter_size'), prev_n_channels=config.getint(sections[i], 'prev_n_channels'),
                                   dim=config.getint(sections[i],'dim'), n_channels=config.getint(sections[i],'n_channels'),
                                   stride=config.getint(sections[i],'stride'), l_rate=config.getfloat(sections[i],'l_rate'), learning=True)
    return network

def read_config_data(file, flatten):
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

    data = generator.selfMadeData(globals, circle, cross, rectangle, bars, flatten = False)

    return data

def read_config_data_1d(file):
    generator = DataGenerator()
    config = ConfigParser()
    config.read(file)
    settings = config._sections

    globals = settings["globals"]
    globals["dim"], globals["n_symbols"] = int(globals["dim"]), int(globals["n_symbols"])
    globals["train"], globals["valid"], globals["test"] = float(globals["train"]), float(globals["valid"]), float(globals["test"])

    data = generator.selfMadeData_1d(globals)

    return data

def main():
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = read_config_data("C:\\Users\eskil\PycharmProjects\it3030\Project2\data_2d.ini", flatten=False)
    network = read_config_network("C:\\Users\eskil\PycharmProjects\it3030\Project2\letwork1_cnn.ini")
    network.train_model(x_train, y_train, x_valid, y_valid, epochs=7, verbose=True)
    print("---END---")
    network.make_prediction(x_test, y_test, dim = 2)
    print("\n")

def main2(): #Use A LOT of epochs if trying to learn the 1d data.. Also, use the method if not using the network from the config file.
    #generator = DataGenerator()
    #pic, answer = generator.horisontal_bars(20, 0.01, 3, 6, centering=False)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = read_config_data("C:\\Users\eskil\PycharmProjects\it3030\Project2\data_2d.ini", flatten = False)
    #(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = read_config_data_1d("C:\\Users\eskil\PycharmProjects\it3030\Project2\data_1d.ini")
    print("morn")
    network = neuralNetwork()
    network.add_conv_layer(filter_size=3, prev_n_channels=1, dim=2, n_channels=4, stride=3, learning = True, l_rate = 0.008)
    network.add_activation_layer('relu')
    network.add_conv_layer(filter_size=3, prev_n_channels=4, dim=2, n_channels=8, stride=1, learning=True, l_rate=0.008)
    network.add_activation_layer('relu')
    network.add_dense_layer(input_size=None, output_size=4)
    network.add_softmax_layer(input_size=4)

    network.train_model(x_train, y_train, x_valid, y_valid, epochs=7, verbose=True, dim=2)
    network.make_prediction(x_test, y_test, dim=2)

if __name__ == '__main__':
    main()

