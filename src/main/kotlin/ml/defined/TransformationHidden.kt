package ml.defined

import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.input.DataParser
import ml.learn.NetworkTeacher
import ml.round
import ml.spine.Activation
import ml.spine.Network
import java.util.*

class TransformationHidden(config: Config) : NetworksLearning(config) {

    override val errorGoal: Double = 0.07
    override val stepsLimit: Int = 1_000_000

    private val sharedTeacher = NetworkTeacher.get(config.teacherMode, 0.1, 0.7)

    override fun setup() {
        val data = DataParser(config.separator, 4, 4).parse(Scanner(config.inputs[0]))
        sharedTeacher.verificationSet = data
        sharedTeacher.trainingSet = data
    }

    override fun buildTeachers(): List<NetworkTeacher> = generateSequence { sharedTeacher }.take(2).toList()

    override fun buildNetworks(): List<Network> = listOf(
        Network.Builder()
            .inputs(4)
            .name("HiddenTransformationBias")
            .setDefaultActivation(Activation.Sigmoid)
            .hiddenLayer(2, true)
            .outputLayer(4, true),
        Network.Builder()
            .inputs(4)
            .name("HiddenTransformationNoBias")
            .setDefaultActivation(Activation.Sigmoid)
            .hiddenLayer(2, false)
            .outputLayer(4, false)
    )

    override fun unfreezing(): List<Network> = baseUnfreezing("HiddenTransformationBias", "HiddenTransformationNoBias")

    override fun allNetworksReady(restored: Boolean) {

        networks.forEach { network ->

            println(network.name)

            for ((input, _) in sharedTeacher.verificationSet) {

                print("Output layer: ")
                println(network.answer(input).map { it.round(3) })

                print("Hidden layer: ")
                println(network.hiddenLayers[0].lastOutput.map { it.activation.round(3) })
                println()
            }
        }

    }
}