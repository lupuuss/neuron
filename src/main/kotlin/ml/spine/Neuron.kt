package ml.spine

import java.lang.IllegalArgumentException
import kotlin.random.Random

class Neuron(
    val inputs: Int,
    val activationFunction: (Double) -> Double,
    val hasBias: Boolean = true
) {
    data class Out(
        val activation: Double,
        val rawValue: Double
    )

    val weights = MutableList(inputs) { Random.nextDouble(-1.0, 1.0) }

    var bias: Double = Random.nextDouble(-1.0, 1.0)

    fun activate(input: List<Double>): Out {

        if (input.size != inputs) {
            throw IllegalArgumentException("Neuron requires $inputs values on input! Passed ${input.size} values.")
        }

        var x = 0.0

        for (i in input.indices) {
            x += weights[i] * input[i]
        }

        x += bias

        return Out(activationFunction(x), x)
    }
}