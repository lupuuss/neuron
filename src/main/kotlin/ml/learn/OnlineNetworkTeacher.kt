package ml.learn

import ml.difference
import ml.spine.*

class OnlineNetworkTeacher(alpha: Double) : NetworkTeacher(alpha) {

    private fun teachSingleNeuron(neuron: Neuron, input: List<Double>, error: Double) {

        for (i in neuron.weights.indices) {

            neuron.weights[i] = neuron.weights[i] - alpha * error * input[i]
        }

        if (neuron.hasBias) {
            neuron.bias = neuron.bias - alpha * error
        }
    }

    private fun teachHiddenLayer(
        layer: Layer,
        followingLayer: Layer,
        input: List<Double>,
        followingLayerErrors: List<Double>
    ): MutableList<Double> {

        val (_, derivative) = layer.activation

        val layerErrors = ArrayList<Double>(layer.size)

        for (i in layer.indices) {

            var errorSum = 0.0

            for (j in followingLayer.indices) {
                errorSum += followingLayerErrors[j] * followingLayer[j].previousWeights[i]
            }

            val weightsBackup = layer[i].weightsBackup()

            teachSingleNeuron(
                layer[i],
                input,
                (errorSum * derivative(layer.lastOutput[i])).also { layerErrors.add(it) }
            )

            layer[i].previousWeights = weightsBackup
        }

        return layerErrors
    }

    private fun teachOutputLayer(
        outputLayer: OutputLayer,
        outputLayerInput: List<Double>,
        differences: List<Double>
    ): MutableList<Double> {

        val derivative = outputLayer.activation.derivative
        val layerErrors = ArrayList<Double>(outputLayer.size)

        for (i in outputLayer.indices) {

            val weightsBackup = outputLayer[i].weightsBackup()

            teachSingleNeuron(
                outputLayer[i],
                outputLayerInput,
                (differences[i] * derivative(outputLayer.lastOutput[i])).also { layerErrors.add(it) }
            )

            outputLayer[i].previousWeights = weightsBackup
        }

        return layerErrors
    }

    override fun teach(network: Network) {

        for ((input, expected) in trainingSet.shuffled()) {

            network.answer(input)

            val differences = network.outputLayer.lastOutput.map { it.activation } difference expected

            var followingLayerErrors: MutableList<Double> = teachOutputLayer(
                network.outputLayer,
                network.hiddenLayers.lastOrNull()?.lastOutput?.map { it.activation } ?: input,
                differences
            )

            if (network.hiddenLayers.size == 0) {
                continue
            }

            var followingLayer: Layer = network.outputLayer

            for (layerIndex in network.hiddenLayers.size - 1 downTo 1) {


                followingLayerErrors = teachHiddenLayer(
                    network.hiddenLayers[layerIndex],
                    followingLayer,
                    network.hiddenLayers[layerIndex - 1].lastOutput.map { it.activation },
                    followingLayerErrors
                )

                followingLayer = network.hiddenLayers[layerIndex]
            }

            teachHiddenLayer(
                network.hiddenLayers.first(),
                followingLayer,
                input,
                followingLayerErrors
            )
        }
    }

    override fun verify(network: Network): List<Double> {

        val errors = mutableListOf<List<Double>>()

        for ((input, expected) in verificationSet) {
            errors.add(network.answer(input) difference expected)
        }

        val errorVector = ArrayList<Double>(errors.first().size)

        for (i in errors.first().indices) {

            var sum = 0.0

            for (j in errors.indices) {

                sum += errors[j][i].let { it * it }
            }

            errorVector.add(sum / errors.first().size)
        }

        return errorVector
    }

}