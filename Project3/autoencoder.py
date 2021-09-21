import torch
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


# Have to make a separate view due to using sequential. Same as reshape in numpy (but using tensors instead).
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# Class for the encoder and head consisting of one extra dense layer + softmax.
class Classifier(nn.Module):
    def __init__(self, encoder):
        super(Classifier, self).__init__()
        self.pretrained = encoder
        self.new_layers = nn.Sequential(
            nn.Linear(encoder.latent_size, 10),
            nn.Softmax()
        )

    def forward(self, x):
        out1 = self.pretrained(x)
        out2 = self.new_layers(out1)
        return out2

    def train_classifier(self, train_loader, valid_loader, params):
        # Method for training the encoder even further with labels.
        accuracies_train = []  # Accumulate the accuracies for plotting
        accuracies_valid = []
        num_correct = 0
        num_samples = 0

        # init network
        # Loss and optimizer
        criterion = nn.MSELoss() if params.get("loss_function_cf") == "mse" else nn.CrossEntropyLoss() # Loss function
        # Optimizer for the backprop
        optimizer = optim.Adam(self.parameters(), lr=params.get("lr_cf")) if params.get("optimizer") == "adam" else optim.SGD(self.parameters(), lr=params.get("lr_cf"))

        # Training the network
        for epoch in range(params.get("epochs_cf")):
            for (img, labels) in train_loader:  # Extract the image and the corresponding labels.
                # Forward
                predicted = self.forward(img)
                loss = criterion(predicted, labels)

                # Extract the accuracies
                _, predictions = predicted.max(1)
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)
                train_acc = float(num_correct/num_samples)
                accuracies_train.append(train_acc)

                # Backward
                optimizer.zero_grad()
                loss.backward() # Backprop through all layers to update weights given the gradient of the loss wrt las output

                # Gradient descent or adam step
                optimizer.step()

            #Validate model after each epoch
            for (img, labels) in valid_loader:
                # get to correct shape
                # Forward
                predicted = self.forward(img)
                _, predictions = predicted.max(1)
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)

            accuracies_valid.append(num_correct/num_samples)
            # print Loss and valid. accuracies
            print(f'Epoch: {epoch + 1}, loss: {loss.item():.4f}')
            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
        return accuracies_train, accuracies_valid


# Encoder. Encode into latent vectors.
class Encoder(nn.Module):
    def __init__(self, latent_size, picture_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.picture_size = picture_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 25, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(25, 50, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(50,latent_size,int(picture_size/4)), # End up with one 64*1*1 (aka one pixel, 64 channels)
                                                           # Easiest solution if latent vectors should be given in advance
                                                           # and also have to adapt to pictures of different dimentions
                                                           # easier to hypertune parameters. Also, dense in middle gave
                                                           # horrible results :/
            nn.Flatten() # Flatten for the TSNE plot.

        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


# Decode the model using sequential layers. Opposite procerure as encoder.
class Decoder(nn.Module):
    def __init__(self, picture_size, latent_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            View((-1, latent_size, 1, 1)), # Reshape the latent vector (basically just adds one extra dimension).
            nn.ConvTranspose2d(latent_size, 50, int(picture_size/4)),
            nn.ReLU(),
            nn.ConvTranspose2d(50, 25, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(25, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


# Just consists of the encoder and decoder specified above.
class Autoencoder(nn.Module):
    def __init__(self, latent_size, picture_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_size=latent_size, picture_size=picture_size)
        self.decoder = Decoder(latent_size=latent_size, picture_size=picture_size)

    def forward(self,x):
        encoded = self.encoder.forward(x)
        decoded = self.decoder.forward(encoded)
        return decoded

    # Training the autoencoder using the input images as "labels". Mostly procedure as in the classifier.
    def train_autoencoder(self, train_loader, valid_loader, params):
        outputs = []
        train_losses = []  # accumulate losses for plotting.
        valid_losses = []
        # init network
        # Loss and optimizer
        criterion = nn.MSELoss() if params.get("loss_function_ae") == "mse" else nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=params.get("lr_ae")) if params.get("optimizer") == "adam" else optim.SGD(self.parameters(), lr=params.get("lr_ae"))

        # Train Network
        for epoch in range(params.get("epochs_ae")):
            loss_sum = 0
            for (img, _) in train_loader:  # Dont use the added labels. Img is the "labels".
                # get to correct shape
                # Forward
                reconstructed = self.forward(img)
                loss = criterion(reconstructed, img)
                train_losses.append(loss)

                # Backprop
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            for (img, _) in valid_loader:
                reconstructed = self.forward(img)
                loss_sum += criterion(reconstructed, img)

            print(loss_sum/len(valid_loader))
            valid_losses.append(loss_sum/len(valid_loader))
            print(f'Epoch: {epoch + 1}, loss: {loss.item():.4f}')
            outputs.append((epoch, img, reconstructed))
        # plot train and valid loss, given accumulated losses.
        plot_losses(train_losses, valid_losses, params.get("epochs_ae"))
        return outputs


"""-----BLOW ARE FUNCTIONS USED FOR PLOTTING AND PREPERATION OF THE DATASET------"""

def show_images(total_epochs, outputs, epochs_shown):
    for epoch in range(0, total_epochs, epochs_shown):
        plt.figure(figsize=(6, 2)) #init figure
        plt.gray()

        pictures = outputs[epoch][1].detach().numpy() # Must convert tensor object to numpy array to be able to plot the images
        reconstructed = outputs[epoch][2].detach().numpy()

        #Shown initial images
        for i, item in enumerate(pictures):
            if i >= 6: break
            plt.subplot(2, 6, i + 1)
            plt.imshow(item[0])

        #Show the recontructed images below for comparison
        for i, item in enumerate(reconstructed):
            if i >= 6: break
            plt.subplot(2, 6, 6 + i + 1)  # row_length + i + 1
            # item: 1, 28, 28
            plt.imshow(item[0])

        plt.show()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Total correct predictions was {num_correct} / {num_samples}, with an accuracy: {float(num_correct) / float(num_samples)*100:.2f}')

def get_dataset(params, size):
    if (params.get("dataset") == 'mnist'):
        dataset = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
    elif (params.get("dataset") == 'fashion-mnist'):
        dataset = datasets.FashionMNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
    elif (params.get("dataset") == 'cifar-10'):
        transform = transforms.Compose([tv.transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])
        dataset = datasets.CIFAR10(root='datasets/', train=True, transform=transform, download=True)

    indices = torch.randperm(len(dataset))[:size]
    subset = Subset(dataset, indices)

    #get train_dataset_ae, train_dataset_cf, valid_dataset_cf, test_dataset_cf
    train_dataset_ae, dataset_cf = random_split(subset, [int(np.ceil(size*params.get("d1"))), int(np.floor(size*params.get("d2")))])
    train_dataset_ae, valid_dataset_ae = random_split(train_dataset_ae, [int(np.ceil(len(train_dataset_ae) * 0.9)), int(np.floor(len(train_dataset_ae) * 0.1))])

    temp, test_dataset_cf = random_split(dataset_cf, [int(np.ceil(len(dataset_cf) * (params.get("d2_train") + params.get("d2_valid")))), int(np.floor(len(dataset_cf)*params.get("d2_test")))])
    train_fraction = params.get("d2_train")/(params.get("d2_train") + params.get("d2_valid"))
    valid_fraction = 1-train_fraction
    train_dataset_cf, valid_dataset_cf = random_split(temp, [int(np.ceil(len(temp)*train_fraction)), int(np.floor(len(temp)*valid_fraction))])
    indices_tsne = torch.randperm(len(test_dataset_cf))[:200]
    tsne_dataset = Subset(test_dataset_cf, indices_tsne)

    data_loader_train_ae = DataLoader(dataset=train_dataset_ae, batch_size=64, shuffle=True)
    data_loader_valid_ae = DataLoader(dataset=valid_dataset_ae, batch_size=64, shuffle=True)
    data_loader_train_cf = DataLoader(dataset=train_dataset_cf, batch_size=64, shuffle=True)
    data_loader_valid_cf = DataLoader(dataset=valid_dataset_cf, batch_size=64, shuffle=True)
    data_loader_test_cf = DataLoader(dataset=test_dataset_cf, batch_size=64, shuffle=True)
    data_loader_tsne = DataLoader(dataset=tsne_dataset, batch_size=1, shuffle=True)
    return data_loader_train_ae, data_loader_valid_ae, \
           data_loader_train_cf, data_loader_valid_cf, \
           data_loader_test_cf, data_loader_tsne

def plot_losses(training_losses, validation_losses, epochs):
    x_axis1 = np.linspace(0, epochs, len(training_losses))
    plt.plot(x_axis1, training_losses, label="training")
    x_axis2 = np.linspace(0, epochs, len(validation_losses))
    plt.plot(x_axis2, validation_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def plot_accuracies(train_acc_semi, valid_acc_semi, train_acc_sup, valid_acc_sup, epochs):
    x_axis1 = np.linspace(0, epochs, len(train_acc_semi))
    plt.plot(x_axis1, train_acc_semi, label="training-acc-semi")

    x_axis2 = np.linspace(0, epochs, len(valid_acc_semi))
    plt.plot(x_axis2, valid_acc_semi, label="validation-acc-semi")

    x_axis3 = np.linspace(0, epochs, len(train_acc_sup))
    plt.plot(x_axis3, train_acc_sup, label="train-acc-sup")

    x_axis4 = np.linspace(0, epochs, len(valid_acc_sup))
    plt.plot(x_axis4, valid_acc_sup, label="validation-acc-sup")

    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='best')
    plt.show()

def plot_tsne(model, tsne_loader, ):
    latent_vectors = []
    labels = []
    for (img, label) in tsne_loader:
        latent = model.forward(img)
        latent = latent.detach().numpy()
        latent_vectors.append(latent)
        label = label.detach().numpy()
        labels.append(label)
    latent_vectors = np.array(latent_vectors)
    latent_vectors = latent_vectors.reshape((200, -1))
    tsne = TSNE()
    tsne_results = tsne.fit_transform(latent_vectors)
    df_subset = pd.DataFrame({'X': tsne_results[:, 0], 'Y': tsne_results[:, 1], 'Targets': labels})
    df_subset = df_subset.astype({'Targets': int})
    colors = sns.color_palette("bright", 10)
    sns.scatterplot(x= "X", y= "Y", hue='Targets', legend = 'full', palette = colors, data=df_subset)
    plt.show()


