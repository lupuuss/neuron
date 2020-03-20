package ml.learn

import ml.spine.Activation
import ml.spine.Neuron

class SingleNeuronTeacher(private val alpha: Double, private val activation: Activation) {

    fun teach(neuron: Neuron, input: List<List<Double>>, expectedOutput: List<Double>) {

        val outputDifferences = MutableList(input.size) { 0.0 }
        val neuronOutputs = ArrayList<Neuron.Out>(input.size)
        val n = input.size

        for (i in input.indices) {

            neuronOutputs.add(neuron.activate(input[i]))
            outputDifferences[i] = neuronOutputs[i].activation - expectedOutput[i]
        }

        for (i in neuron.weights.indices) {

            var derivativeValue = 0.0
            for (j in input.indices) {

                derivativeValue += outputDifferences[j] * activation.derivative(neuronOutputs[j]) * input[j][i]

            }

            derivativeValue *= 1.0 / n.toDouble()

            neuron.weights[i] = neuron.weights[i] - alpha * derivativeValue
        }

        // bias part

        var derivativeValue = 0.0

        for (j in input.indices) {

            derivativeValue += outputDifferences[j] * activation.derivative(neuronOutputs[j])

        }

        derivativeValue *= 1.0 / n.toDouble()

        neuron.bias = neuron.bias - alpha * derivativeValue
    }

    fun verify(neuron: Neuron, inputArray: List<List<Double>>, expectedOutput: List<Double>): Double {

        var error = 0.0
        for (i in inputArray.indices) {
            val diff = neuron.activate(inputArray[i]).activation - expectedOutput[i]
            error += diff * diff
        }

        return error / inputArray.size
    }
}