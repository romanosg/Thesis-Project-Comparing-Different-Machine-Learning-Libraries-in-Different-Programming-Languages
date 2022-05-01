# This is a sample Python script.

import time

from sklearn import datasets, model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from torch import nn, no_grad, max, from_numpy, sigmoid, multiprocessing, unsqueeze, equal
from torch.optim import Adam

from tensorflow import keras
from keras.datasets import mnist


# The following methods are Sklearn oriented

# function that trains a given model, evaluates it through f1 score, and calculates the training and testing times.
def skl_master_model(X_train, X_test, y_train, y_test, model):

    # training phase
    print("Initialize Model Training. Wait a bit...")
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_duration = end - start
    print("Model Completed!")

    # testing phase
    print("Initialize Testing Phase")
    start = time.time()
    y_predict = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_predict)
    end = time.time()
    test_duration = end - start

    # printing results
    print("Test Phase completed!")
    print("Training Time (in seconds):" + str(train_duration) +
          ". Testing Time (in seconds): " + str(test_duration) + ".")
    print("Accuracy score: " + str(acc_score * 100) + "%")


# method that creates a Neural Network MLP Model of 784 inputs, 2 hidden layers of 50 neurons each and the output layer
# to determine 10 different classes.
def skl_MLP(X_train, X_test, y_train, y_test, random_state=42, epoch=1000):
    print("Initializing MLP model")
    model = MLPClassifier(
        solver="adam", hidden_layer_sizes=(50, 2), random_state=random_state,
        max_iter=epoch, learning_rate_init=0.001
    )
    skl_master_model(X_train, X_test, y_train, y_test, model)
    print("End of MLP model")
    print()


# method that creates a knn model with k as input (default 3), which is trained, tested and evaluated
def skl_knn(X_train, X_test, y_train, y_test, k=3):
    print("Initializing Knn model, k =", k)
    model = KNeighborsClassifier(k)
    skl_master_model(X_train, X_test, y_train, y_test, model)
    print("End of Knn model")
    print()


# method that creates an svm model with C and kernel as input (default 1 and rbf respectively) which is trained,
# tested and evaluated. Policy is one versus one
def skl_svm(X_train, X_test, y_train, y_test, random_state=42, C=1, kernel='rbf'):
    print("Initializing SVM model, C =", C, "Kernel =", kernel)
    model = SVC(C=C, kernel=kernel, random_state=random_state)
    skl_master_model(X_train, X_test, y_train, y_test, model)
    print("End of SVM model")
    print()


# method that creates a random forest model with max depth as input (default 3), which is trained, tested and evaluated
def skl_random_forest(X_train, X_test, y_train, y_test, random_state=42, depth=3):
    print("Initializing Random Forest Model, depth =", depth)
    model = RandomForestClassifier(max_depth=depth, random_state=random_state)
    skl_master_model(X_train, X_test, y_train, y_test, model)
    print("End of Random Forest model")
    print()


# method that creates a decision tree model with max depth as input (default 3), which is trained, tested and evaluated
def skl_decision_tree(X_train, X_test, y_train, y_test, random_state=42, depth=3):
    print("Initializing Decision Tree model, depth =", depth)
    model = DecisionTreeClassifier(random_state=random_state, max_depth=depth)
    skl_master_model(X_train, X_test, y_train, y_test, model)
    print("End of Decision Tree model")
    print()

# method that creates a gradient boost model with max depth as input (default 3), which is trained, tested and evaluated
def skl_gradient_boost(X_train, X_test, y_train, y_test, random_state=42, depth=3):
    print("Initializing Gradient Boosting model")
    model = GradientBoostingClassifier(random_state=random_state, max_depth=3)
    skl_master_model(X_train, X_test, y_train, y_test, model)
    print("End of Gradient Boosting model")
    print()

# method that creates a ada boost model, which is trained, tested and evaluated
def skl_ada_boost(X_train, X_test, y_train, y_test, random_state=42):
    print("Initializing Ada Boosting model")
    model = AdaBoostClassifier(random_state=random_state)
    skl_master_model(X_train, X_test, y_train, y_test, model)
    print("End of Ada Boosting model")
    print()


# method that creates a gaussian naive bayes model, which is trained tested and evaluated
def skl_naive_bayes(X_train, X_test, y_train, y_test):
    print("Initializing Gaussian Naive Bayes model")
    model = GaussianNB()
    skl_master_model(X_train, X_test, y_train, y_test, model)
    print("End of Gaussian Naive Bayes model")
    print()


# The following methods are Tensorflow Oriented
# method that creates a CNN keras model to train mnist data (28x28) and evaluate
def tf_keras_CNN(X_train, X_test, y_train, y_test, epochs=1000):
    # creating the CNN Neural Network Model of 3 filter layers of 5x5 followed by to Fully Connected layers of
    # 50 neurons each.
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(1, kernel_size=(5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(keras.layers.Conv2D(1, kernel_size=(5, 5), activation="relu"))
    model.add(keras.layers.Conv2D(1, kernel_size=(5, 5), activation="relu"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(50, activation="sigmoid"))
    model.add(keras.layers.Dense(50, activation="sigmoid"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print("Training Keras CNN")
    tf_keras_model_train_and_test(X_train, X_test, y_train, y_test, model, epochs)


# method that creates an MLP keras model to train mnist data (28x28) and evaluate
def tf_keras_MLP(X_train, X_test, y_train, y_test, epochs=1000):
    # creating the MLP Neural Network Model (28*28 input layer, 2 layers of 50 neurons, 10 output neurons)
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(50, input_dim=28*28, activation="sigmoid"))
    model.add(keras.layers.Dense(50, activation="sigmoid"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print("Training Keras MLP")
    tf_keras_model_train_and_test(X_train, X_test, y_train, y_test, model, epochs)


# method that creates a neural network in TensorFlow
def tf_keras_model_train_and_test(X_train, X_test, y_train, y_test, model, epochs=1000):
    # training phase
    print("Initialize Model Training. Wait a bit...")
    start = time.time()
    model.fit(X_train, y_train, epochs=epochs, verbose=0)  # includes both forward and backwards phase
    end = time.time()
    train_duration = end - start
    print("Model Completed!")

    # testing phase
    print("Initialize Testing Phase")
    start = time.time()
    acc = model.evaluate(X_test, y_test)
    end = time.time()
    test_duration = end - start

    # printing results
    print("Test Phase completed!")
    print("Training Time (in seconds): " + str(train_duration))
    print("Testing Time (in seconds): " + str(test_duration))
    print("[Test Loss (in %), Test acc (in %)] =", acc)


# The following methods are PyTorch Oriented

def pt_MLP(X_train, X_test, y_train, y_test, epochs=1000):
    # creating the MLP Neural Network Model (784 input layer, 2 layers of 50 neurons, 10 output neurons)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 50),
        nn.Sigmoid(),
        nn.Linear(50, 50),
        nn.Sigmoid(),
        nn.Linear(50, 10),
    )
    X_train = from_numpy(X_train).float()
    y_train = from_numpy(y_train).float()
    X_test = from_numpy(X_test).float()
    y_test = from_numpy(y_test).float()
    print("Training PyTorch MLP")
    pt_model_train_and_test(X_train, X_test, y_train, y_test, model, epochs)


def pt_CNN(X_train, X_test, y_train, y_test, epochs=1000):
    # creating the CNN Neural Network Model (3 layers of 5x5 filters into 2 FC layers of 50 neurons)
    model = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=(5, 5)),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=(5, 5)),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_size=(5, 5)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*16, 50),
        nn.Sigmoid(),
        nn.Linear(50, 50),
        nn.Sigmoid(),
        nn.Linear(50, 10)
    )
    # Important: Torch can only support Tensor Data. Transforming data into such
    X_train = from_numpy(X_train).float()
    X_test = from_numpy(X_test).float()
    # Important 2: Conv2D layer support data of the form: [Index, Channel_id, height_id, width_id]. Transforming
    # data of the current form: [Index, height_id, width_id] into such
    X_train = unsqueeze(X_train, 1)
    X_test = unsqueeze(X_test, 1)
    y_train = from_numpy(y_train).float()
    y_test = from_numpy(y_test).float()
    print("Training PyTorch CNN")
    pt_model_train_and_test(X_train, X_test, y_train, y_test, model, epochs)


def pt_model_train_and_test(X_train, X_test, y_train, y_test, model, epochs=1000):
    # Important: Must transform data from numpy to tensor data
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=0.001)
    # training phase
    start = time.time()
    print("Initialize Model Training. Wait a bit...")
    model.train()
    for i in range(epochs):

        # forward phase
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # backward phase
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # printing training per epoch
        # print("Epoch: {}/{},  loss: {:.4f}".format(i+1, epochs, loss.item()))

    end = time.time()
    train_duration = end - start

    print("Model Completed!")

    # testing phase
    print("Initialize Testing Phase")
    start = time.time()
    with no_grad():
        n_correct = 0
        n_samples = len(y_test)
        outputs = model(X_test)
        for index in range(n_samples):
            _, prediction = max(outputs[index].data, 0)  # getting prediction id
            # transforming prediction to categorical tensor
            prediction = keras.utils.to_categorical(prediction, 10)
            prediction = from_numpy(prediction)

            if equal(prediction, y_test[index]):  # using torch.equal to compare 2 categorial tensors
                n_correct += 1
    end = time.time()
    acc = n_correct/n_samples
    test_duration = end - start

    # printing results
    print("Test Phase completed!")
    print("Training Time (in seconds): " + str(train_duration))
    print("Testing Time (in seconds): " + str(test_duration))
    print("Test acc =", str(acc*100) + "%")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # defining a specific random_state
    random_state = 42
    # load data
    # loading mnist
    print("loading mnist digits through tensorflow")
    digits = mnist.load_data()
    print("loading successful")
    print("loading iris dataset")

    print("loading successful")
    #  split and transform data
    # mnist data
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = digits

    # SKL mnist data/label preprocessing
    X_train_mnist_skl = X_train_mnist.reshape(X_train_mnist.shape[0], -1)[:1000] / 255.0
    X_test_mnist_skl = X_test_mnist.reshape(X_test_mnist.shape[0], -1)[:1000] / 255.0
    y_train_mnist_skl = y_train_mnist[:1000]
    y_test_mnist_skl = y_test_mnist[:1000]

    # transforming mnist datallabels for pytorch/tensorflow
    X_train = X_train_mnist.reshape((X_train_mnist.shape[0], 28 * 28)).astype('float32')
    X_test = X_test_mnist.reshape((X_test_mnist.shape[0], 28 * 28)).astype('float32')
    X_train_mnist = X_train_mnist[:1000]
    y_train_mnist = y_train_mnist[:1000]
    X_test_mnist = X_test_mnist[:1000]
    y_test_mnist = y_test_mnist[:1000]
    X_train_mnist = X_train_mnist / 255.0
    X_test_mnist = X_test_mnist / 255.0
    y_train_mnist = keras.utils.to_categorical(y_train_mnist, 10)
    y_test_mnist = keras.utils.to_categorical(y_test_mnist, 10)
    # model training, testing and evaluation
    # iris data
    df_iris = datasets.load_iris()
    X_train_iris_foo = df_iris.data
    y_train_iris_foo = df_iris.target
    X_train_iris = []
    y_train_iris = []
    # enlargen the dataset instances by copying itself: 150 -> 30000
    for i in range(0, 200):
        X_train_iris.extend(X_train_iris_foo)
        y_train_iris.extend(y_train_iris_foo)
    X_test_iris = X_train_iris
    y_test_iris = y_train_iris

    # Scikit learn category
    # mnist classifiers
    print("Iris length is", len(y_train_iris))
    print()
    # print("Evaluating Scikit-learn library:")
    print("Mnist Dataset Classifiers:")
    # time: 10.334857702255249 acc: 51.7%
    skl_MLP(X_train_mnist_skl, X_test_mnist_skl, y_train_mnist_skl, y_test_mnist_skl)  # MLP Neural Network model
    # iris classifiers
    print("Iris Dataset Classifiers:")
    # time: 0.0019969940185546875 acc: 100%
    skl_knn(X_train_iris, X_test_iris, y_train_iris, y_test_iris)
    # time: 0.003101825714111328 acc: 97.33333333333334%
    skl_decision_tree(X_train_iris, X_test_iris, y_train_iris, y_test_iris)
    # time: 0.48766326904296875 acc: 97.33333333333334%
    skl_random_forest(X_train_iris, X_test_iris, y_train_iris, y_test_iris)
    # time: 4.079087972640991 acc: 100.0%
    skl_gradient_boost(X_train_iris, X_test_iris, y_train_iris, y_test_iris)
    # time: 0.4667530059814453 acc: 96.0%
    skl_ada_boost(X_train_iris, X_test_iris, y_train_iris, y_test_iris)
    # time: 1.306537389755249 acc: 98.66666666666667%
    skl_svm(X_train_iris, X_test_iris, y_train_iris, y_test_iris)
    # time: 0.016459941864013672 acc: 96.0%
    skl_naive_bayes(X_train_iris, X_test_iris, y_train_iris, y_test_iris)
    print("End of evaluation of Scikit-learn library\n")

    # TensorFlow category
    # mnist classifiers
    print("Evaluating TensorFlow framework (with help of Keras):")
    # time: 40.56000351905823 loss: 1.2039401531219482 acc: 0.843999981880188
    tf_keras_MLP(X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist)  # MLP Neural Network model
    # time: 261.3809230327606 loss: 0.9589889645576477 acc: 0.8709999918937683
    tf_keras_CNN(X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist)  # CNN Neural Network model
    print("End of evaluation of TensorFlow framework\n")

    # PyTorch category
    # mnist classifiers
    print("Evaluating PyTorch framework:")
    # time: 3.7038354873657227 acc: 83.1%
    pt_MLP(X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist)  # MLP Neural Network model
    # time: 122.4336507320404 acc: 84.39999999999999%
    pt_CNN(X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist)  # CNN Neural Network model
    print("End of evaluation of PyTorch framework\n")
