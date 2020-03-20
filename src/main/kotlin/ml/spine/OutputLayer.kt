package ml.spine

@Suppress("unused")
class OutputLayer(
    neuronsCount: Int,
    inputsCount: Int,
    activation: Activation,
    hasBias: Boolean = true
) : Layer(neuronsCount, inputsCount, activation, hasBias) {

    fun activateAndCompare(input: List<Double>, expected: List<Double>): Pair<List<Neuron.Out>, List<Double>> {

        val result = this.activate(input)
        val differences = ArrayList<Double>(input.size)

        for (i in result.indices) {
            differences[i] = result[i].activation - expected[i]
        }

        return result to differences
    }
}