import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.mnist
import smile.base.rbf.RBF
import smile.classification.*
import smile.data.formula.Formula
import java.util.*
import smile.io.Read
import smile.math.kernel.Gaussian
import smile.math.kernel.GaussianKernel
import java.lang.Math.pow


// KotlinDL related functions
// libraries that creates an MLP model to train, test and evaluate
fun kdl_MLP(train: OnHeapDataset, test: OnHeapDataset) {
    // creating a MLP model of 28 input neurons, 2 Hidden layers of 50 neurons and 1 output layer of 10 neurons
    val model = Sequential.of(
        Input(28,28,1),
        Flatten(),
        Dense(50, activation = Activations.Sigmoid),
        // Dense(50, activation = Activations.Sigmoid),
        Dense(10, activation = Activations.Softmax)
    )
    // train and test the model
    print("Initialize training of MLP model")
    kdl_train_and_test(train, test, model) // time: 20.025 lossValue: 1.5560485124588013, Acc: 0.9036144614219666
}

// function that creates a CNN model to train, test and evaluate
fun kdl_CNN(train: OnHeapDataset, test: OnHeapDataset) {
    // creating the CNN Neural Network Model of 3 filter layers of 5x5 followed by to Fully Connected layers of
    // 50 neurons each.
    val model = Sequential.of(
        Input(28,28,1),
        Conv2D(1, kernelSize=intArrayOf(5, 5), activation=Activations.Relu), // 28x28 => 24x24
        Conv2D(1, kernelSize=intArrayOf(5, 5), activation=Activations.Relu), // 24x24 => 20x20
        Conv2D(1, kernelSize=intArrayOf(5, 5), activation=Activations.Relu), // 20x20 => 16x16
        Flatten(),
        Dense(50, activation = Activations.Sigmoid),
        Dense(50, activation = Activations.Sigmoid),
        Dense(10, activation = Activations.Softmax)
    )
    // train and test the model
    print("Initialize training of CNN model")
    kdl_train_and_test(train, test, model) // time: 20.025 lossValue: 1.5560485124588013, Acc: 0.9036144614219666
}

// function that trains, tests and evaluates a KotlinDL Deep Learning (MLP/CNN) model
fun kdl_train_and_test(train: OnHeapDataset, test: OnHeapDataset, model: Sequential) {
    // method that use model's functions
    model.use{
        // creating the metric compiler
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()

        // training data and count the time needed
        val start = Date().getTime()
        print("Training Model")
        it.fit(dataset = train, epochs = 1000)
        print("end of training model")
        val end = Date().getTime()
        val duration = (end - start)/1000.0
        print("Model trained Time required: ")
        print(duration)
        println(" seconds!")

        // test model's accuracy
        val accuracy = it.evaluate(test)
        print("accuracy: ")
        println(accuracy)
    }
}

fun main(args: Array<String>) {
    // mnist data management
    // taking the entire mnist dataset, which included their own train and test set with kotlindl
    val (train_whole, test_whole) = mnist()
    // splitting the data into a user specific amount. For this instance, we take 1000 data from both train and test
    val desired_size = 1000.0
    val actual_size = train_whole.xSize()*1.0
    val splitRatio = desired_size / actual_size
    var (train, dontTrain) = train_whole.split(splitRatio = splitRatio)
    var (test, dontTest) = test_whole.split(splitRatio = splitRatio)
    // iris data management
    // train and test libraries
    // Kotlin DL
    // kdl_MLP(train, test) // MLP
    // kdl_CNN(train, test) // CNN
    // smile
    // val smile_mlp = mlp(train.x, train.y., epochs = 1000)
    // Smile Iris Classification
    // Read data and split to train/test set
    var start = System.nanoTime()
    var end = System.nanoTime()
    // making foo dataframe to enlargen the original with through iteration. Original iris dataframe has 150 instances.
    var iris_foo = Read.arff("https://storm.cis.fordham.edu/~gweiss/data-mining/weka-data/iris.arff")
    // must double the dataframe first because method union changes datatype of class column
    iris_foo= iris_foo.union(iris_foo)
    var iris = iris_foo
    // enlargen the dataset instances. From 150 data into 30000
    // note: array used to copy itself is double the actual, so, it requires half iterations - 1
    for (i in 1 .. 99) {
        iris = iris.union(iris_foo)
    }
    val train_data = iris.drop("class").toArray()
    val train_labels = iris.column("class").toIntArray()

    // Decision Tree model training
    println("Initializing Decision Tree model training max depth = 5")
    start = System.nanoTime()
    val decision_tree_model = cart(Formula.lhs("class"), iris, maxDepth = 5) // time: 0.1427886
    end = System.nanoTime()
    print("Decision training complete: Time required: ")
    println((end-start)/pow(10.0,9.0))

    // Random Forest Model
    println("Initializing Random Forest model training, max depth = 3")
    start = System.nanoTime()
    val random_forest_model = randomForest(Formula.lhs("class"), iris, maxDepth = 3) // time: 1.624469
    end = System.nanoTime()
    print("Random Forest training complete: Time required: ")
    println((end-start)/pow(10.0,9.0))

    // Gradient Boosting Model
    println("Initializing Gradient boost model training, max depth = 3")
    start = System.nanoTime()
    val gradient_boost_model = gbm(Formula.lhs("class"), iris, maxDepth = 3) // time: 14.8248096
    end = System.nanoTime()
    print("Random Forest training complete: Time required: ")
    println((end-start)/pow(10.0,9.0))

    // Ada Boosting Model
    println("Initializing Ada boost model training")
    start = System.nanoTime()
    val ada_boost_model = adaboost(Formula.lhs("class"), iris) // time: 2.3199194
    end = System.nanoTime()
    print("Ada boost training complete: Time required: ")
    println((end-start)/pow(10.0,9.0))

    // SVM (one versus one) Model
    println("Initializing 1v1 SVM model training, C = 1, kernel = Gaussian (sigma = 8)")
    val kernel = GaussianKernel(8.0)
    start = System.nanoTime()
    val svm_model = ovo(train_data, train_labels, { x, y -> svm(x, y, kernel, 1.0) }) // time: 17.1880007
    end = System.nanoTime()
    val duration = (end-start)/pow(10.0,9.0)
    print("SVM training complete: Time required: ")
    println((end-start)/pow(10.0,9.0))

}