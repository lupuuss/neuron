package ml.spine

@Suppress("MemberVisibilityCanBePrivate")
open class Layer(
    neuronsCount: Int,
    inputsCount: Int,
    val activation: Activation,
    val hasBias: Boolean = true
) : Iterable<Neuron> {

    val neurons: MutableList<Neuron> = MutableList(neuronsCount) { Neuron(inputsCount, activation, hasBias) }

    val indices get() = neurons.indices

    val size = neurons.size

    @Transient
    lateinit var lastOutput: List<Neuron.Out>
        private set

    fun activate(input: List<Double>): List<Neuron.Out> {

        val result = ArrayList<Neuron.Out>(input.size)

        for (i in neurons.indices) {

            result.add(neurons[i].activate(input))
        }

        lastOutput = result
        return result
    }

    override fun iterator(): Iterator<Neuron> {

        return neurons.iterator()
    }
}

operator fun Layer.set(i: Int, neuron: Neuron) {
    this.neurons[i] = neuron
}

operator fun Layer.get(i: Int): Neuron = this.neurons[i]