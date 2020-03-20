package ml.spine

import ml.learn.NetworkTeacher
import java.lang.Exception

class NoActivationFunctionException() : Exception(
    "Activation function not chosen! Use setDefaultActivation or pass as argument to function."
)

class Network private constructor(
    private val teacher: NetworkTeacher,
    private val hiddenLayers: MutableList<Layer> = mutableListOf(),
    private val outputLayer: OutputLayer
) {

    fun teach() {
        teacher.teach(hiddenLayers, outputLayer)
    }

    fun answer(input: List<Double>): List<Double> {

        var lastOutput = input

        for (hiddenLayer in hiddenLayers) {

            lastOutput = hiddenLayer.activate(lastOutput).map { it.activation }
        }

        return outputLayer.activate(lastOutput).map { it.activation }
    }

    class Builder(
        private val inputs: Int,
        private val teacher: NetworkTeacher
    ) {

        private var defaultActivation: Activation? = null
        private val hiddenLayers: MutableList<Layer> = mutableListOf()

        fun setDefaultActivation(activation: Activation): Builder {
            this.defaultActivation = activation
            return this
        }

        fun hiddenLayer(neurons: Int, hasBias: Boolean, activation: Activation? = null): Builder {

            val neuronsInputs = hiddenLayers.lastOrNull()?.neurons?.size ?: inputs

            hiddenLayers.add(
                Layer(
                    neurons,
                    neuronsInputs,
                    activation ?: defaultActivation ?: throw NoActivationFunctionException(),
                    hasBias
                )
            )

            return this
        }

        fun outputLayer(neurons: Int, hasBias: Boolean, activation: Activation? = null) = Network(
            teacher,
            hiddenLayers,
            OutputLayer(
                neurons,
                hiddenLayers.last().neurons.size,
                activation ?: defaultActivation ?: throw NoActivationFunctionException(),
                hasBias
            )
        )
    }
}