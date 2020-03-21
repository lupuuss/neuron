package ml.learn

import ml.spine.Network

abstract class NetworkTeacher(
    val alpha: Double
) {

    enum class Mode {
        Online, Offline
    }

    var trainingSet: List<Pair<List<Double>, List<Double>>> = emptyList()
    var verificationSet: List<Pair<List<Double>, List<Double>>> = emptyList()

    abstract fun teach(network: Network)

    abstract fun verify(network: Network): List<Double>
}