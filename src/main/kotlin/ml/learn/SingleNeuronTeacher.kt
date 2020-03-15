package ml.learn

import ml.spine.Activation
import ml.spine.Neuron

class SingleNeuronTeacher(private val alpha: Double, private val activation: Activation) {

    fun teach(neuron: Neuron, inputArray: Array<DoubleArray>, expectedOutput: DoubleArray) {

        val outputDifferences = DoubleArray(inputArray.size)
        val neuronOutputs = ArrayList<Neuron.Out>(inputArray.size)
        val n = inputArray.size
        
        for (i in inputArray.indices) {

            neuronOutputs.add(neuron.activate(inputArray[i]))
            outputDifferences[i] = neuronOutputs[i].activation - expectedOutput[i]
        }

        for (i in neuron.weights.indices) {

            var derivativeValue = 0.0
            for (j in inputArray.indices) {

                derivativeValue += outputDifferences[j] * activation.derivative(neuronOutputs[j]) * inputArray[j][i]

            }

            derivativeValue *= 1.0 / n.toDouble()

            neuron.weights[i] = neuron.weights[i] - alpha * derivativeValue
        }

        // bias part

        var derivativeValue = 0.0

        for (j in inputArray.indices) {

            derivativeValue += outputDifferences[j] * activation.derivative(neuronOutputs[j])

        }

        derivativeValue *= 1.0 / n.toDouble()

        neuron.bias = neuron.bias - alpha * derivativeValue
    }

    fun verify(neuron: Neuron, inputArray: Array<DoubleArray>, expectedOutput: DoubleArray): Double {

        var error = 0.0
        for (i in inputArray.indices) {
            val diff = neuron.activate(inputArray[i]).activation - expectedOutput[i]
            error += diff * diff
        }

        return error / inputArray.size
    }
}