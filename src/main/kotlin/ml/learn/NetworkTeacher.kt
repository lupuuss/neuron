package ml.learn

import ml.spine.Layer
import ml.spine.OutputLayer

abstract class NetworkTeacher {

    var trainingSet: List<Pair<DoubleArray, DoubleArray>> = emptyList()
    var verificationSet: List<Pair<DoubleArray, DoubleArray>> = emptyList()

    abstract fun teach(hiddenLayer: List<Layer>, outputLayer: OutputLayer)

    abstract fun verify(hiddenLayer: List<Layer>, outputLayer: OutputLayer)
}