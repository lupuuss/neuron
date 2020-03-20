package ml.spine

@Suppress("MemberVisibilityCanBePrivate")
open class Layer(
    neuronsCount: Int,
    inputsCount: Int,
    val activation: Activation,
    val hasBias: Boolean = true
) {

    val neurons: MutableList<Neuron> = if (hasBias) {
        MutableList(neuronsCount) { Neuron(inputsCount, activation.function) }
    } else {
        MutableList(neuronsCount) { Neuron(inputsCount, activation.function).also { it.bias = 0.0 } }
    }

    fun activate(input: List<Double>): List<Neuron.Out> {

        val result = ArrayList<Neuron.Out>(input.size)

        for (i in neurons.indices) {

            result.add(neurons[i].activate(input))
        }

        return result
    }
}