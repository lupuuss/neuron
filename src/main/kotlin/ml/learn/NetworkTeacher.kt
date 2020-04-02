package ml.learn

import ml.difference
import ml.spine.Network
import java.lang.IllegalArgumentException

abstract class NetworkTeacher(
    var alpha: Double,
    var beta: Double
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

    abstract val metric: String

    abstract fun teach(network: Network)

    fun verify(network: Network): List<Double> {

        val errors = mutableListOf<List<Double>>()

        for ((input, expected) in verificationSet) {
            errors.add(network.answer(input) difference expected)
        }

        val errorVector = ArrayList<Double>(errors.first().size)

        for (i in errors.first().indices) {

            var sum = 0.0

            for (j in errors.indices) {

                sum += errors[j][i].let { it * it }
            }

            errorVector.add(sum / (errors.first().size * 2))
        }

        return errorVector
    }

    companion object {

        @JvmStatic
        fun get(mode: Mode, alpha: Double, beta: Double): NetworkTeacher = when (mode) {
            Mode.Online -> OnlineNetworkTeacher(alpha, beta)
            Mode.Offline -> OfflineNetworkTeacher(alpha, beta)
        }
    }
}