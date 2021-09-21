from PIL import Image
import numpy as np
import random
import cv2
from config import *



class DataGenerator():

    def circle_greyscale(self, n, prob, min_radius, max_radius, pic_number, centering = False):
        data = np.zeros((n, n), dtype=np.uint8)

        radius = random.randint(min_radius, max_radius)
        eps = int(np.floor(n/5))
        #Move circle to random areas each time
        max_skew = np.floor(n/2 - radius)
        skew_x, skew_y = 0, 0
        if not centering:
            skew_x = random.randint(-max_skew, max_skew)
            skew_y = random.randint(-max_skew, max_skew)
        center_circle = [n/2 + skew_x, n/2 + skew_y]

        for col in range(n):
            for row in range(n):
                if (np.abs((center_circle[0] - row) ** 2 + (center_circle[1] - col) ** 2 - (radius ** 2)) < eps): #Sirkelformelen
                    data[col, row] = 255 if random.random() > prob else 0 #lower and upper boundary to insert a random black/white pixel (noise)
                else:
                    data[col, row] = 0 if random.random() > prob else 255
        #img = Image.fromarray(data, mode='L')  # Lage et bilde
        #img.save("mornja.png")
        return data, 0

    def cross(self, n, prob, min_size, max_size, centering = False):
        data = np.zeros((n, n), dtype=np.uint8)
        size = random.randint(min_size,max_size)
        center_pic = np.floor(n/2) - 1 #Senter av bilde

        #Kode for å skyve bilde i random x og y posisjoner. Må passe på den den ikke skyves ut av bilde (derav min/max skew)
        skew_X, skew_y = 0, 0
        if not centering:
            max_skew = np.floor((n - size)/2)
            skew_X = random.randint(-max_skew, max_skew)
            skew_y = random.randint(-max_skew, max_skew)
        center_x = int(center_pic + skew_X)
        center_y = int(center_pic + skew_y)

        #Kode for å tegne krysset med senter i center_x,y.
        for i in range(0, int(np.floor(size/2))):
            data[center_x + i, center_y] = 255 if random.random() > prob else 0
            data[center_x - i, center_y] = 255 if random.random() > prob else 0
            data[center_x, center_y + i] = 255 if random.random() > prob else 0
            data[center_x, center_y - i] = 255 if random.random() > prob else 0

        #Kode for å legge til støy
        for col in range(n):
            for row in range(n):
                rdn = random.random()
                if rdn < prob:
                    data[col][row] = 255
        #img = Image.fromarray(data, mode='L')  # Lage et bilde
        #img.save("mornja.png")
        return data,1

    #Code for the rectangle.
    def rectangle(self, n, prob, min_side_size, max_side_size, centering = False):
        data = np.zeros((n, n), dtype=np.uint8)
        x1 = random.randint(0, n-1) if not centering else int(np.floor(n/4))
        if x1 + min_side_size < n-1: #Check which way to draw the line (left or right). dont want it outside picture.
            x2 = random.randint(x1 + min_side_size, min(x1 + max_side_size, n-1))
        else:
            x2 = random.randint(max(x1 - max_side_size, 0), x1 - min_side_size)

        #same as above but in vertical direction
        y1 = random.randint(0, n-1) if not centering else int(np.floor(n/4))
        if y1 + min_side_size < n-1:
            y2 = random.randint(y1 + min_side_size, min(y1 + max_side_size, n-1))
        else:
            y2 = random.randint(max(y1 - max_side_size, 0), y1 - min_side_size)

        #use built-in function to draw the rectangle given the two points specified above.
        cv2.rectangle(data, pt1=(x1, y1), pt2=(x2, y2), color=255, thickness=1)

        #add the noise
        for col in range(n):
            for row in range(n):
                rdn = random.random()
                if rdn < prob:
                    data[col][row] = 255

        #img = Image.fromarray(data, mode='L')  # Create a PIL image
        #img.save("mornja.png")
        return data, 2

    def horisontal_bars(self, n, prob, min_num_bars, max_num_bars, centering = False):
        data = np.zeros((n, n), dtype=np.uint8)
        num_bars = random.randint(min_num_bars, max_num_bars)
        bars = []
        bar = random.randint(0, n)
        bars.append(bar)
        cv2.line(data, pt1=(0, bar), pt2=(n-1, bar), color=255, thickness=1)
        for i in range(num_bars):
            while True:
                bar = random.randint(0, n) #The row in which the bar will be drawn
                if (bar not in bars) and (bar+1 not in bars) and (bar-1 not in bars): #dont draw the bar at occupied location or straight above/below
                    bars.append(bar)
                    cv2.line(data, pt1=(0, bar), pt2=(n - 1, bar), color=(255, 0, 0), thickness=1)
                    break

        #Add the noise
        for col in range(n):
            for row in range(n):
                rdn = random.random()
                if rdn < prob:
                    data[col][row] = 255
        #img = Image.fromarray(data)  # Create a PIL image
        #img.save("mornja.png")
        return data, 3

    def selfMadeData(self, globals, circle, cross, rectangle, bars, flatten):
        data = [] #list of tuples with (pic, target)
        #input is a dictionary. Extract all information.
        dim, noise, centering, n_symbols = globals["dim"], globals["noise"], globals["centering"], globals["n_symbols"]
        train_size, valid_size, test_size = globals["train"], globals["valid"], globals["test"]
        for i in range(n_symbols):
            number = random.randint(0, 3) #choose from random symbols
            if (number == 0):
                x, y = self.circle_greyscale(dim, noise, circle["min_radius"], circle["max_radius"], centering)
            elif (number == 1):
                x, y = self.cross(dim, noise, cross["min_size"], cross["max_size"], centering)
            elif (number == 2):
                x, y = self.rectangle(dim, noise, rectangle["min_side_size"], rectangle["max_side_size"], centering)
            else:
                x, y = self.horisontal_bars(dim, noise, bars["min_num_bars"], bars["max_num_bars"], centering)
            img = Image.fromarray(x, mode='L')  # Create a PIL image
            path = "C:\\Users\eskil\PycharmProjects\it3030\data\mornja" + str(i) + ".png"
            img.save(path)
            data.append([x, y])

        #Extract a train-, valid- and test set from the data given the userspecified distribution
        start_valid, start_test = int(np.ceil(n_symbols*train_size)), int(np.ceil(n_symbols*(1-test_size)))
        train = data[:start_valid]
        validation = data[start_valid+1:start_test]
        test = data[start_test+1:]

        #Clean the inputs. (one hot encode targets, normalize inputs, flatten image etc)
        x_train, y_train = self.get_train_test_data(train, dim, flatten)
        x_valid, y_valid = self.get_train_test_data(validation, dim, flatten)
        x_test, y_test = self.get_train_test_data(test, dim, flatten)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    # Helper method for getting the data in the right dims.
    def get_train_test_data(self, data, dim, flatten):
        x = np.array([x[0] for x in data]) #extract the picture
        if (flatten == True):
            x = x.reshape(x.shape[0], dim * dim, 1) #flatten it
        x = x.astype('float32') #change to float
        x /= 255 #normalize
        y = np.array([x[1] for x in data]) #extract target
        y = np.eye(4)[y] #one-hot encode target
        y = y.reshape(y.shape[0], 4, 1) #make column vector

        return x, y


    def binary_1d_vectors(self, n, num_sequences):
        data = np.zeros(n, dtype=np.uint8)
        sequences_start = []
        sequences_end = []
        sequence_start = random.randint(0, n-1)
        sequences_start.append(sequence_start)
        for i in range(num_sequences - 1):
            while True:
                sequence_start = random.randint(0, n-1) #The row in which the bar will be drawn
                if (sequence_start not in sequences_start) \
                        and (sequence_start+1 not in sequences_start) \
                        and (sequence_start-1 not in sequences_start): #dont draw the bar at occupied location or straight above/below
                    sequences_start.append(sequence_start)
                    break
        sequences_start = np.array(sequences_start)
        sequences_start = np.sort(sequences_start)

        for i in range(len(sequences_start) - 1):
            end = random.randint(sequences_start[i], sequences_start[i+1]-2)
            sequences_end.append(end)
        end = random.randint(sequences_start[-1], n-1)
        sequences_end.append(end)
        for i in range(num_sequences):
            data[sequences_start[i]:sequences_end[i]+1] = 255
        return data

    def selfMadeData_1d(self, globals):
        data = []
        n_symbols, pixels = 100, 20
        dim, n_symbols = globals["dim"], globals["n_symbols"]
        train_size, valid_size, test_size = globals["train"], globals["valid"], globals["test"]
        visualized = np.zeros((20, pixels), dtype=np.uint8)
        for i in range(n_symbols):
            number = random.randint(2,5)
            if number == 2:
                x, y = self.binary_1d_vectors(n=pixels, num_sequences=1), 0
            elif number == 3:
                x, y = self.binary_1d_vectors(n=pixels, num_sequences=3), 1
            elif number == 4:
                x, y = self.binary_1d_vectors(n=pixels, num_sequences=5), 2
            else:
                x, y = self.binary_1d_vectors(n=pixels, num_sequences=7), 3

            if i < 20:
                visualized[i,:] = x
            data.append([x, y])
        img = Image.fromarray(visualized, mode='L')  # Create a PIL image
        path = "C:\\Users\eskil\PycharmProjects\it3030\data_1d\mornja.png"
        img.save(path)

        #Extract a train-, valid- and test set from the data given the userspecified distribution
        start_valid, start_test = int(np.ceil(n_symbols*train_size)), int(np.ceil(n_symbols*(1-test_size)))
        train = data[:start_valid]
        validation = data[start_valid+1:start_test]
        test = data[start_test+1:]

        x_train, y_train = self.get_train_test_data_1d(train)
        x_valid, y_valid = self.get_train_test_data_1d(validation)
        x_test, y_test = self.get_train_test_data_1d(test)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    def get_train_test_data_1d(self, data):
        x = np.array([x[0] for x in data])
        x = x.astype('float32') #change to float
        x /= 255 #normalize
        y = np.array([x[1] for x in data]) #extract target
        y = np.eye(4)[y] #one-hot encode target
        y = y.reshape(y.shape[0], 4, 1) #make column vector

        return x, y


def main():
    generator = DataGenerator()
    morn = generator.binary_1d_vectors(20,4)


if __name__ == '__main__':
    main()




