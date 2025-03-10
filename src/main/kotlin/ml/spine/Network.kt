package ml.spine

class NoActivationFunctionException : Exception(
    "Activation function not chosen! Use setDefaultActivation or pass as argument to function."
)

class Network private constructor(
    val name: String,
    val hiddenLayers: MutableList<Layer>,
    val outputLayer: Layer
) {

    fun answer(input: List<Double>): List<Double> {

        var lastOutput = input

        for (hiddenLayer in hiddenLayers) {

            lastOutput = hiddenLayer.activate(lastOutput).map { it.activation }
        }

        return outputLayer.activate(lastOutput).map { it.activation }
    }

    class Builder {

        private var inputsCount: Int? = null
        private var name: String? = null
        private var defaultActivation: Activation? = null
        private val hiddenLayers: MutableList<Layer> = mutableListOf()

        fun inputs(inputs: Int): Builder {
            inputsCount = inputs
            return this
        }

        fun name(name: String): Builder {
            this.name = name
            return this
        }

        fun setDefaultActivation(activation: Activation): Builder {
            this.defaultActivation = activation
            return this
        }

        fun hiddenLayer(neurons: Int, hasBias: Boolean, activation: Activation? = null): Builder {

            if (inputsCount == null) throw IllegalStateException("You have to set networks inputs count before hidden layers")

            val neuronsInputs = hiddenLayers.lastOrNull()?.neurons?.size ?: inputsCount!!

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

        fun outputLayer(neurons: Int, hasBias: Boolean, activation: Activation? = null): Network {

            if (inputsCount == null) throw IllegalStateException("You have to set networks inputs count before output layer")

            return Network(
                name ?: throw IllegalStateException("You have to set name for the network!"),
                hiddenLayers,
                Layer(
                    neurons,
                    hiddenLayers.lastOrNull()?.neurons?.size ?: inputsCount!!,
                    activation ?: defaultActivation ?: throw NoActivationFunctionException(),
                    hasBias
                )
            )
        }
    }
}