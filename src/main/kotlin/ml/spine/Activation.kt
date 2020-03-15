package ml.spine

import kotlin.math.exp

data class Activation(
    val function: (Double) -> Double,
    val derivative: (Neuron.Out) -> Double
) {
    companion object {

        @JvmStatic
        fun sigmoid(): Activation = Activation(
            { x -> 1.0 / (1.0 + exp(-x)) },
            { (activation) -> activation * (1 - activation) }
        )
    }
}
