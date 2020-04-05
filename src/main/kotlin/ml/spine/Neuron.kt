package ml.spine

import java.lang.IllegalArgumentException
import kotlin.random.Random

class Neuron(
    private val inputs: Int,
    val activation: Activation,
    val hasBias: Boolean = true
) {
    data class Out(
        val activation: Double,
        val rawValue: Double
    )

    val weights: MutableList<Double> = MutableList(inputs) { Random.nextDouble(-1.0, 1.0) }

    var previousWeights: List<Double> = weights.toList() // copy of weights

    var bias: Double = if (hasBias) Random.nextDouble(-1.0, 1.0) else 0.0

    var previousBias: Double = bias // copy of bias

    fun activate(input: List<Double>): Out {

        if (input.size != inputs) {
            throw IllegalArgumentException("Neuron requires $inputs values on input! Passed ${input.size} values.")
        }

        var x = 0.0

        for (i in input.indices) {
            x += weights[i] * input[i]
        }

        x += bias

        return Out(activation.function(x), x)
    }

    fun weightsAndBiasBackup(): Pair<MutableList<Double>, Double> = weights.toMutableList() to bias

    fun setPrevious(backup: Pair<MutableList<Double>, Double>) {

        previousWeights = backup.first
        previousBias = backup.second
    }
}