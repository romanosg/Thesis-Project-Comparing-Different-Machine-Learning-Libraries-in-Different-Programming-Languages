# install packages with method install_packages("package_name_in_string")

# call libraries
library(caret) # iris classification library
library(tensorflow) # mnist classification framework. Using Κeras instead
library(keras) # mnist classification framework
library(torch) # mnist classification framework
library(dplyr) # R toolkit
library(magrittr) # R toolkit
library(tictoc) # library that counts time

# DATA LOADING/PREPROCESSING PHASE: USING KERAS
# number of train and test set samples
train_samples <- 1000
test_samples <- 1000
# reading the dataset using keras. 28x28 images
mnist <- dataset_mnist()
# get 1000 data to 
X_train_data <- mnist$train$x[1:train_samples,,]
X_test_data <- mnist$test$x[1:test_samples,,]
y_train_data <- mnist$train$y[1:train_samples]
y_test_data <- mnist$test$y[1:test_samples]
# reshape data in the form [amount, height, width, channel]. It is not accepted
# in Torch CNN model
X_train <- array_reshape(X_train_data, c(1000, 28, 28, 1))
X_test <- array_reshape(X_test_data, c(1000, 28, 28, 1))
# rescale data to [0,1]
X_train <- X_train / 255
X_test <- X_test / 255

# transforming y labels from a numeric value into categorical
y_train <- to_categorical(y_train_data, num_classes = 10)
y_test <- to_categorical(y_test_data, num_classes = 10)

# KERAS MLP NEURAL NETWORK 
# building model using keras, 2 hidden layers of 50 neurons each
print("Building and training Keras MLP Model")
keras_mlp_model <- keras_model_sequential() %>%
  layer_flatten() %>%
  layer_dense(units = 50, activation = "sigmoid", input_shape = c(28*28)) %>%
  layer_dense(units = 50, activation = "sigmoid") %>%
  layer_dense(units = 10, activation = "softmax")

# creating compile parameters and training the model
keras_mlp_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

# training phase
tic()
print("Initiating Keras MLP model Training")
history <- keras_mlp_model %>% 
  fit(X_train, y_train, epochs = 1000, verbose=0) # train time: 38.82
toc()
print("Keras MLP model finished training")

# OPTIONAL: INITIATE TESTING PHASE

# KERAS CNN
# building model using keras. No padding, default stride = 1
# # followed by 2 Fully connected layers of 50 neurons each
print("Building and training Keras CNN Model")
keras_cnn_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 1, kernel_size = c(5,5), activation = 'relu', 
                input_shape = c(28,28,1)) %>% # 28x28 => 24x24
  
  layer_conv_2d(filters = 1, kernel_size = c(5,5), activation = 'relu') %>% # 24x24 => 20x20
  layer_conv_2d(filters = 1, kernel_size = c(5,5), activation = 'relu') %>% # 20x20 => 16x16
  layer_flatten() %>%
  layer_dense(units = 50, activation = "sigmoid") %>%
  layer_dense(units = 50, activation = "sigmoid") %>%
  layer_dense(units = 10, activation = "softmax")

# creating compile parameters and training the model
keras_cnn_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)
# training phase
# training phase
tic()
print("Initiating Keras CNN model Training")
history <- keras_cnn_model %>% 
  fit(X_train, y_train, epochs = 1000, verbose=0) # train time: 3.665918 x 60
toc()
print("Keras CNN model finished training")

# TORCH MLP
# Transforming data from atomic vectors into tensors (required to run torch)
# reshaping data in the form of [amount, channels, height, width]. Does not
# work for Keras CNN model
X_train_torch <- array_reshape(X_train_data, c(1000, 1, 28, 28))
X_test_torch <- array_reshape(X_test_data, c(1000, 1, 28, 28))
X_train_tensor = torch_tensor(X_train_torch, dtype = torch_float())
y_train_tensor = torch_tensor(y_train, dtype = torch_float())
X_test_tensor = torch_tensor(X_test_torch, dtype = torch_float())
y_test_tensor = torch_tensor(y_test, dtype = torch_float())
# building model using Torch of 2 hidden layers of 50 neurons each
print("Building and training Torch MLP Model")
torch_mlp_model <- nn_sequential(
  nn_flatten(),
  nn_linear(28*28, 50),
  nn_sigmoid(),
  nn_linear(50, 50),
  nn_sigmoid(),
  nn_linear(50,10),
)
# creating loss and optimizer parameters nd training the model
criterion = nn_cross_entropy_loss()  
optimizer = optim_adam(torch_mlp_model$parameters, lr = 0.001)
# training phase
print("Initiating Torch MLP model Training")
tic()
for(i in 1:1000){
  y_pred <- torch_mlp_model(X_train_tensor)
  loss <- criterion(y_pred, y_train_tensor)
  optimizer$zero_grad()
  loss$backward()
  optimizer$step()
  # train time: 25.3987109661102
}
toc()
print("Torch MLP model finished training")

# OPTIONAL: INITIATE TESTING PHASE

# TORCH CNN
# Transforming data from atomic vectors into tensors (required to run torch)
X_train_torch <- array_reshape(X_train_data, c(1000, 1, 28, 28))
X_test_torch <- array_reshape(X_test_data, c(1000, 1, 28, 28))
X_train_tensor = torch_tensor(X_train_torch, dtype = torch_float())
y_train_tensor = torch_tensor(y_train, dtype = torch_float())
X_test_tensor = torch_tensor(X_test_torch, dtype = torch_float())
y_test_tensor = torch_tensor(y_test, dtype = torch_float())
# building model using Torch, 3 filters of 5x5, no padding, stride = 1
# followed by 2 Fully connected layers of 50 neurons each
print("Building and training Torch CNN Model")
torch_cnn_model <- nn_sequential(
  nn_conv2d(1, 1, kernel_size=c(5, 5)), # 28x28 => 24x24
  nn_relu(),
  nn_conv2d(1, 1, kernel_size=c(5, 5)), # 24x24 => 20x20
  nn_relu(),
  nn_conv2d(1, 1, kernel_size=c(5, 5)), # 20x20 => 16x16
  nn_relu(),
  nn_flatten(),
  nn_linear(16*16, 50),
  nn_sigmoid(),
  nn_linear(50, 50),
  nn_sigmoid(),
  nn_linear(50, 10)
)
# creating loss and optimizer parameters nd training the model
criterion = nn_cross_entropy_loss()  
optimizer = optim_adam(torch_cnn_model$parameters, lr = 0.001)
# training phase
print("Initiating Torch CNN model Training")
tic()
for(i in 1:1000){
  y_pred <- torch_cnn_model(X_train_tensor)
  loss <- criterion(y_pred, y_train_tensor)
  optimizer$zero_grad()
  loss$backward()
  optimizer$step()
  # train time: 3.26102121671041 x 60 
}
toc()
print("Torch MLP model finished training")

# CARET MODELS
# insert model_code for the ML algorithm of your choosing:
# few examples (each algorithm can be created by different models tags, each
# example will include model tag)
# k-nearest neighbors = "knn"
# linear regression = "lm"
# Decision Tree (CART) = "rpart2"
# SVM (e.g. Radial Kernel) = "svmRadial"
# Random Forest = "ranger"
# Gradient Boosting = "gbm"
# MLP Back Propagation = "mlpML"
# if a model requires a library installation, RStudio will notify you and give
# you the option to do so
# for more models, visit https://topepo.github.io/caret/train-models-by-tag.html

# handling dataset
df <- data(iris) # iris dataset
iris_df <- iris
iris_foo <- iris
x <- 0
# enlargening the dataset from 150 data to 30000
print("enlargening dataset")
tic()
for(i in 1:199){
  iris_df <- rbind(iris_df, iris_foo)
}
print("Enlargening Successful!")
set.seed(42)
toc()
# Decision Tree Classifier
print("Decision Tree model Classifier, max depth = 5")
tic()
caret::train(Species ~ ., data = iris_df,
      method =  "rpart2",
      trControl = trainControl(method="none"),
      tuneGrid = expand.grid(maxdepth=5),
      tuneLength = 1
      )
toc() # 0.36 seconds

# Random Forest Classifier
print("Random Forest model Classifier")
tic()
caret::train(Species ~ ., data = iris_df,
             tuneLength = 1,
             trControl = trainControl(method="none"),
             method =  "rf"
)
toc() # 5.39 seconds

# Gradient Boosting Classifier
print("Gradient Boosting Classifier")
tic()
caret::train(Species ~ ., data = iris_df,
             method =  "gbm",
             trControl = trainControl(method="none"),
             tuneLength = 1,
             verbose = FALSE
)
toc() # 1.78 seconds

# AdaBoost Classifier
print("AdaBoost Classifier")
tic()
caret::train(Species ~ ., data = iris_df,
             method =  "AdaBag",
             trControl = trainControl(method="none"),
             tuneLength = 1,
             verbose = FALSE
)
toc() # 4.69 seconds


# SVM Classifier (one versus one), C = 1 and RBF kernel
print("SVM model classifierm C = 1, RBF kernel")
tic()
caret::train(Species ~ ., data = iris_df,
             method =  "svmRadial",
             tuneLength = 1,
             trControl = trainControl(method="none"),
             tuneGrid = expand.grid(C = 1, sigma = 8),
             verbose = FALSE
)
toc() # 2.08 seconds
