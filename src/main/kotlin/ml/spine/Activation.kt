package ml.spine

import kotlin.math.exp

@Suppress("unused")
enum class Activation(
    val function: (Double) -> Double,
    val derivative: (Neuron.Out) -> Double
) {
    Sigmoid(
        { x -> 1.0 / (1.0 + exp(-x)) },
        { (activation) -> activation * (1 - activation) }
    ),
    Identity(
        { x -> x },
        { 1.0 }
    );

    operator fun component1(): (Double) -> Double = function
    operator fun component2(): (Neuron.Out) -> Double = derivative
}