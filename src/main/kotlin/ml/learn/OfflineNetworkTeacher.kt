package ml.learn

import ml.difference
import ml.spine.*

class OfflineNetworkTeacher(alpha: Double, beta: Double) : NetworkTeacher(alpha, beta) {

    override val metric: String = "Iterations"

    private fun collectAnswers(network: Network): List<List<List<Neuron.Out>>> {

        val result = MutableList(network.hiddenLayers.size + 1) {
            MutableList(trainingSet.size) { listOf<Neuron.Out>() }
        }

        for (i in trainingSet.indices) {

            val (input, _) = trainingSet[i]

            network.answer(input)

            for (j in network.hiddenLayers.indices) {
                result[j][i] = network.hiddenLayers[j].lastOutput
            }

            result.last()[i] = network.outputLayer.lastOutput
        }

        return result
    }

    private fun teachSingleNeuron(
        neuron: Neuron,
        neuronIndex: Int,
        errors: List<List<Double>>,
        inputs: List<List<Double>>
    ) {

        val n = inputs.size

        for (weightIndex in neuron.weights.indices) {

            var derivativeValue = 0.0
            for (inputIndex in inputs.indices) {

                derivativeValue += errors[inputIndex][neuronIndex] * inputs[inputIndex][weightIndex]

            }

            derivativeValue *= 1.0 / n

            val momentumValue = neuron.weights[weightIndex] - neuron.previousWeights[weightIndex]

            neuron.weights[weightIndex] = neuron.weights[weightIndex] - alpha * derivativeValue + beta * momentumValue
        }

        if (!neuron.hasBias) return

        val biasDerivative = errors.map { it[neuronIndex] }.sum() / n
        val biasMomentum = neuron.bias - neuron.previousBias
        neuron.bias = neuron.bias - alpha * biasDerivative + beta * biasMomentum
    }

    private fun calcErrorPropagation(
        layer: Layer,
        followingLayer: Layer,
        followingErrors: List<List<Double>>
    ): MutableList<MutableList<Double>> {
        val errors = MutableList<MutableList<Double>>(followingErrors.size) { mutableListOf() }

        for (i in errors.indices) {

            for (j in layer.indices) {

                var errorSum = 0.0
                for (k in followingLayer.indices) {
                    errorSum += followingErrors[i][k] * followingLayer[k].weights[j]
                }

                errors[i].add(errorSum)
            }
        }

        return errors
    }

    private fun calcOutputLayerError(outputLayerOutputs: List<List<Neuron.Out>>): MutableList<MutableList<Double>> {

        val errors = MutableList<MutableList<Double>>(outputLayerOutputs.size) { mutableListOf() }

        for (i in trainingSet.indices) {

            val (_, expected) = trainingSet[i]

            errors[i] = (outputLayerOutputs[i].map { it.activation } difference expected).toMutableList()
        }

        return errors
    }

    private fun multiplyErrorsByDerivative(
        errors: MutableList<MutableList<Double>>,
        layer: Layer,
        outputs: List<List<Neuron.Out>>
    ) {

        for (i in errors.indices) {

            for (j in errors.first().indices) {

                val derivative = layer[j].activation.derivative

                errors[i][j] *= derivative(outputs[i][j])
            }
        }
    }

    private fun teachLayer(
        layer: Layer,
        errors: MutableList<MutableList<Double>>,
        outputs: List<List<Neuron.Out>>,
        inputs: List<List<Double>>
    ) {

        multiplyErrorsByDerivative(errors, layer, outputs)

        for (i in layer.indices) {

            val backup = layer[i].weightsAndBiasBackup()

            teachSingleNeuron(layer[i], i, errors, inputs)

            layer[i].setPrevious(backup)
        }
    }

    override fun teach(network: Network) {
        val answers = collectAnswers(network)

        val inputs = trainingSet.map { it.first }
        val outputLayerInput = answers.getOrNull(answers.lastIndex - 1)
            ?.map { list ->
                list.map { it.activation }
            } ?: inputs

        var followingLayerErrors = calcOutputLayerError(answers.last())

        teachLayer(network.outputLayer, followingLayerErrors, answers.last(), outputLayerInput)

        var followingLayer = network.outputLayer

        if (network.hiddenLayers.size == 0) {
            return
        }

        for (i in network.hiddenLayers.size - 1 downTo 0) {

            val errors = calcErrorPropagation(network.hiddenLayers[i], followingLayer, followingLayerErrors)

            val layerInput = answers.getOrNull(i - 1)?.map { list -> list.map { it.activation } } ?: inputs

            teachLayer(
                network.hiddenLayers[i],
                errors,
                answers[i],
                layerInput
            )

            followingLayer = network.hiddenLayers[i]
            followingLayerErrors = errors
        }
    }
}

