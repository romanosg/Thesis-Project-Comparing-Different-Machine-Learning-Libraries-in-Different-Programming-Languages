Julia_Flux_DL

using Pkg # Package Installer

# the following packages were used and downloaded through Pkg, for isntance: Pkg.add("Flux")
# Mnist Classification Libraries
using Flux
using Flux: Data.DataLoader # Mnist data manager
# IrisClassification Libraries
using ScikitLearn: fit!, predict, @sk_import # ScikitLearn model handler
using DecisionTree # DecisionTree and AdaBoost Classifier for scikitlearnj
using GradientBoosting # Julia Library used for GradientBoosting
using LIBSVM # SVM Classifier for ScikitLearn
using NearestNeighbors # Knn classifier
# Datasets
using MLDatasets # Mnist Dataset
using RDatasets # Iris Dataset

function flux_train_and_test(train_set, test_set, model, epoch=1000)
    # neural network related data
    loss(x,y) = Flux.logitcrossentropy(model(x), y) # loss function
    parameters = Flux.params(model); # weight parameters
    optimizer = ADAM(0.001) # optimizer

    # train the neural network model

    print("Begin train\n")
    @time begin
        for active_epoch in 1:epoch
            print("\nepoch:", active_epoch)
            Flux.train!(loss, parameters, train_set, optimizer)
        end
        print("\nTraining time:")
    end
    print("End Train\n")
    # useful variables for testing
    print("Start Testing\n")
    # calculating accuracy
    test_samples_len = 1000
    correct = 0
    @time begin
        for (x,y) in test_set
            prediction = Flux.onecold(model(x))
            actual = Flux.onecold(y)
            if prediction == actual
                correct = correct + 1
            end
        end
        correct/test_samples_len
        accuracy = (1.0*correct)/(1.0*test_samples_len)
    end
    print("End Testing\n")
    print("Accuracy is: ", accuracy*100)
    print("%")
end

# function that creates an MLP model with the help of Flux, which is trained, tested and evaluated
function flux_MLP(train_set, test_set, epoch=1000)
    # neural network model 784x50x50x10
    model = Flux.Chain(
    Flux.flatten,
    Flux.Dense((28*28), 50, Flux.σ),
    Flux.Dense(50, 50, Flux.σ),
    Flux.Dense(50, 10)
    )
    flux_train_and_test(train_set, test_set, model, epoch)
end

# function that creates an CNN model with the help of Flux, which is trained, tested and evaluated
function flux_CNN(train_set, test_set, epoch=1000)
    # CNN model 3 layers of a 5x5 filter each, into 2 Fully Connected layers of 50 neurons each
    model = Flux.Chain(
    Flux.Conv((5,5), 1=>1, relu), # 28x28 => 24x24
    Flux.Conv((5,5), 1=>1, relu), # 24x24 => 20x20
    Flux.Conv((5,5), 1=>1, relu), # 20x20 => 16x16
    Flux.flatten,
    Flux.Dense((16*16), 50, Flux.σ),
    Flux.Dense(50, 50, Flux.σ),
    Flux.Dense(50, 10)
    )
    flux_train_and_test(train_set, test_set, model, epoch)
end

# MNIST Classification using Flux
# Load the data with MLDatasets
X_train_mnist, y_train_mnist = MLDatasets.MNIST.traindata(Float32);
X_test_mnist, y_test_mnist = MLDatasets.MNIST.testdata(Float32);

# data related info
train_samples = 1000; # amount of test samples
test_samples = 1000; # amount of test samples

# get subset data, reshaping and rescaling data
X_train_mnist = X_train_mnist[:,:,1:train_samples] / 255.0;
y_train_mnist = y_train_mnist[1:train_samples];
X_test_mnist = X_test_mnist[:,:,1:test_samples] / 255.0;
y_test_mnist = y_test_mnist[1:test_samples];
# Add the channel layer
X_train_mnist = Flux.unsqueeze(X_train_mnist, 3);
X_test_mnist = Flux.unsqueeze(X_test_mnist, 3);
# Encode labels
y_train_mnist = Flux.onehotbatch(y_train_mnist, 0:9);
y_test_mnist = Flux.onehotbatch(y_test_mnist, 0:9);

# form sets
train_set_mnist = DataLoader((X_train_mnist, y_train_mnist));
test_set_mnist = DataLoader((X_test_mnist, y_test_mnist));
flux_MLP(train_set_mnist, test_set_mnist, 1000); # time: 678.639904, # acc: 79.6%
flux_CNN(train_set_mnist, test_set_mnist, 1000); # time: 1017.11, # acc: 9.9% (does not train layers)

# Iris Classification using other libraries
# Load Dataframe using RDatasets
iris_df = dataset("datasets", "iris");
X_train_iris = Matrix(iris_df[!,1:3]); # getting the data
y_train_iris = Vector{String}(iris_df.Species) # getting the labels

# encoding labels from string to integer ids
y_train_iris_nums_temp = y_train_iris;
y_train_iris_nums_temp = replace(y_train_iris_nums_temp, "setosa"=>1)
y_train_iris_nums_temp = replace(y_train_iris_nums_temp, "versicolor"=>2)
y_train_iris_nums_temp = replace(y_train_iris_nums_temp, "virginica"=>3)

# enlargening the dataset by copying itself 200 times.
y_train_iris_labels = rand(30000) # creating
X_train_iris_data = rand(30000, 3)
for i in 1:200
    batch_id = (i-1)*150
    for j in 1:150
        y_train_iris_labels[batch_id+j] = y_train_iris_nums_temp[j]
        X_train_iris_data[batch_id+j,1:3] = X_train_iris[j,1:3]
    end
end

# Decision Tree Classifier of Scikit Learn max depth = 5
# documentation and examples of decision Tree, RandomForest, AdaBoost: https://github.com/bensadeghi/DecisionTree.jl#scikitlearnjl
println("Decision Tree Model training. Max Depth = 5")
@time begin
    fit!(DecisionTreeClassifier(max_depth=5), X_train_iris_data, y_train_iris_labels); # time: 0.006676
end

# Random Forest Classifier of Scikit Learn with max depth = 3
println("Random Forest Model training. Max Depth = 3")
@time begin
    fit!(RandomForestClassifier(max_depth=3), X_train_iris_data, y_train_iris_labels); # time: 0.039413 
end

# Gradient Boosting Classifier with max depth = 3, learning rate of 0.1 and number of trees (n_estimators) = 100. The 2 latter
# are the default assignments of scikit-learn library in python for gradient boosting
println("Gradient Boosting Model training")
@time begin
    GradientBoosting.fit(y_train_iris_labels, X_train_iris_data, 0.1, 3, 100) # time: 1.271912
end

# AdaBoost Classifier of Scikit Learn
println("AdaBoost Model training")
@time begin
    fit!(AdaBoostStumpClassifier(), X_train_iris_data, y_train_iris_labels); # time: 0.089260
end

# SVM with the help of LibSVM that adapts the model in ScikitLearn template. 
# SVM documentation and examples: https://github.com/JuliaML/LIBSVM.jl
println("SVM model (one versus one) training")
@time begin
    fit!(SVC(;kernel=Kernel.RadialBasis), X_train_iris_data, y_train_iris_labels); # time: 3.691794
end
print("\nFinish");