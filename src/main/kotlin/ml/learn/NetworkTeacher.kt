package ml.learn

import ml.spine.Network
import java.lang.IllegalArgumentException

abstract class NetworkTeacher(
    val alpha: Double,
    val beta: Double
) {

    init {
        if (beta !in 0.0..1.0) {
            throw IllegalArgumentException("Beta must have value in range [0; 1]!")
        }
    }

    enum class Mode {
        Online, Offline
    }

    var trainingSet: List<Pair<List<Double>, List<Double>>> = emptyList()
    var verificationSet: List<Pair<List<Double>, List<Double>>> = emptyList()

    abstract fun teach(network: Network)

    abstract fun verify(network: Network): List<Double>
}