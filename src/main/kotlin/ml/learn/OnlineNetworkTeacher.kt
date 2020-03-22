package ml.learn

import ml.difference
import ml.spine.*

class OnlineNetworkTeacher(alpha: Double, beta: Double) : NetworkTeacher(alpha, beta) {

    override val metric: String = "Epochs"

    private fun teachSingleNeuron(neuron: Neuron, input: List<Double>, error: Double) {

        for (i in neuron.weights.indices) {

            val momentumValue = neuron.weights[i] - neuron.previousWeights[i]
            neuron.weights[i] = neuron.weights[i] - alpha * error * input[i] + beta * momentumValue
        }

        if (neuron.hasBias) {
            val biasMomentum = neuron.bias - neuron.previousBias
            neuron.bias = neuron.bias - alpha * error + beta * biasMomentum
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

            val backup = layer[i].weightsAndBiasBackup()

            teachSingleNeuron(
                layer[i],
                input,
                (errorSum * derivative(layer.lastOutput[i])).also { layerErrors.add(it) }
            )

            layer[i].setPrevious(backup)
        }

        return layerErrors
    }

    private fun teachOutputLayer(
        outputLayer: Layer,
        outputLayerInput: List<Double>,
        differences: List<Double>
    ): MutableList<Double> {

        val derivative = outputLayer.activation.derivative
        val layerErrors = ArrayList<Double>(outputLayer.size)

        for (i in outputLayer.indices) {

            val backup = outputLayer[i].weightsAndBiasBackup()

            teachSingleNeuron(
                outputLayer[i],
                outputLayerInput,
                (differences[i] * derivative(outputLayer.lastOutput[i])).also { layerErrors.add(it) }
            )

            outputLayer[i].setPrevious(backup)
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
}