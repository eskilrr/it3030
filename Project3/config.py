from configparser import ConfigParser
from autoencoder import *


#Read the config file into a dictionary and return the dict.
def read_config_file(file):
    config = ConfigParser()
    config.read(file)
    params = config.sections()[0]
    parameters = {
        "dataset" : config.get(params, 'dataset'),
        "picture_size" : int(config.get(params, 'picture_size')),
        "lr_ae" : float(config.get(params, 'lr_ae')),
        "lr_cf" : float(config.get(params, "lr_cf")),
        "loss_function_ae" : config.get(params, "loss_function_ae"),
        "optimizer" : config.get(params, "optimizer"),
        "latent_size" : int(config.get(params, "latent_size")),
        "epochs_ae" : int(config.get(params, "epochs_ae")),
        "epochs_cf" : int(config.get(params, "epochs_cf")),

        "d1" : float(config.get(params, "d1")),
        "d2" : float(config.get(params, "d2")),
        "d2_train" : float(config.get(params, "d2_train")),
        "d2_valid" : float(config.get(params, "d2_valid")),
        "d2_test" : float(config.get(params, "d2_test")),

        "freeze_weight" : int(config.get(params, "freeze_weight")),
        "n_reconstructions" : int(config.get(params, "n_reconstructions")),
        "plot_tsne" : int(config.get(params, "plot_tsne")),
        "epochs_shown": int(config.get(params, "epochs_shown"))
    }

    return parameters



def main():
    params = read_config_file("config.ini")
    data_loader_train_ae, data_loader_valid_ae, data_loader_train_cf, data_loader_valid_cf, data_loader_test_cf, tsne_dataset = get_dataset(params=params, size=10000)

    # Train autoencoder and test classifier
    print("-----train Autoencoder and classifier------")
    autoencoder = Autoencoder(params.get('latent_size'), picture_size=params.get("picture_size"))
    plot_tsne(model=autoencoder.encoder, tsne_loader=tsne_dataset)
    outputs = autoencoder.train_autoencoder(data_loader_train_ae, data_loader_valid_ae, params)
    plot_tsne(model=autoencoder.encoder, tsne_loader=tsne_dataset)
    show_images(params.get("epochs_ae"), outputs, epochs_shown=params.get("epochs_shown"))

    classifier_semi = Classifier(autoencoder.encoder)
    train_accuracies_semi, valid_accuracies_semi = classifier_semi.train_classifier(data_loader_train_cf, data_loader_valid_cf, params)
    plot_tsne(model=classifier_semi.pretrained, tsne_loader=tsne_dataset)
    print("\n-----test Autoencoeer and classifier------")
    check_accuracy(loader=data_loader_test_cf, model=classifier_semi)

    #Train only classifier and test it
    print("\n----train only classifier-----")
    autoencoder = Autoencoder(latent_size=params.get('latent_size'), picture_size=params.get("picture_size"))
    classifier_sup = Classifier(autoencoder.encoder)
    train_accuracies_super, valid_accuracies_super = classifier_sup.train_classifier(data_loader_train_cf, data_loader_valid_cf, params=params)
    plot_accuracies(train_accuracies_semi, valid_accuracies_semi, train_accuracies_super, valid_accuracies_super, params.get("epochs_cf"))
    print("\n-----test classifier------")
    check_accuracy(loader=data_loader_test_cf, model=classifier_sup)

    #Check final accuracy
    print("\n-----FINAL TEST SEMI-----")
    check_accuracy(loader=data_loader_train_ae, model=classifier_semi)
    print("\n-----FINAL TEST SUPER----")
    check_accuracy(loader=data_loader_train_ae, model=classifier_sup)

if __name__ == '__main__':
    main()