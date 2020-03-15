package ml.spine

@Suppress("unused")
class OutputLayer(
    neuronsCount: Int,
    inputsCount: Int,
    activation: Activation,
    hasBias: Boolean = true
) : Layer(neuronsCount, inputsCount, activation, hasBias) {

    fun activateAndCompare(input: DoubleArray, expected: DoubleArray): Pair<Array<Neuron.Out>, DoubleArray> {

        val result = this.activate(input)
        val differences = DoubleArray(input.size)

        for (i in result.indices) {
            differences[i] = result[i].activation - expected[i]
        }

        return result to differences
    }
}