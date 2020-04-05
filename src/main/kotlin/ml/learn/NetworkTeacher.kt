package ml.learn

import ml.difference
import ml.input.LearnData
import ml.spine.Network
import java.lang.IllegalArgumentException

abstract class NetworkTeacher(
    var alpha: Double,
    var beta: Double
) {

    var name: String? = null

    init {
        if (beta !in 0.0..1.0) {
            throw IllegalArgumentException("Beta must have value in range [0; 1]!")
        }
    }

    enum class Mode {
        Online, Offline
    }

    var trainingSet: LearnData = emptyList()
    var verificationSet: LearnData = emptyList()

    abstract val metric: String

    abstract fun teach(network: Network)

    private fun verify(network: Network, verificationSet: LearnData): List<Double> {

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

    fun verify(network: Network): List<Double> = verify(network, verificationSet)

    fun verifyTraining(network: Network): List<Double> = verify(network, trainingSet)

    companion object {

        @JvmStatic
        fun get(mode: Mode, alpha: Double, beta: Double): NetworkTeacher = when (mode) {
            Mode.Online -> OnlineNetworkTeacher(alpha, beta)
            Mode.Offline -> OfflineNetworkTeacher(alpha, beta)
        }
    }
}