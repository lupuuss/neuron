package ml.defined

import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.round
import ml.spine.Activation
import ml.spine.Network
import java.util.stream.Stream
import kotlin.streams.toList

class Transformation(config: Config) : NetworksLearning(config) {

    override val errorGoal: Double = 0.000001
    override val stepsLimit: Int = 50_000

    private val sharedTeacher = NetworkTeacher.get(config.teacherMode, 0.1, 0.0)

    override fun setup() {

        val data = dataParser.parse(config.inputs[0], 4, 4)
        sharedTeacher.trainingSet = data
        sharedTeacher.verificationSet = data

        asyncRunner = true
    }

    private val commonName = "Transformtion"

    override fun buildNetworks(): List<Network> {

        val biased = (1..3).map { hiddenNeurons ->
            Network.Builder()
                .name("${commonName}_Bias_$hiddenNeurons")
                .setDefaultActivation(Activation.Sigmoid)
                .inputs(4)
                .hiddenLayer(hiddenNeurons, true)
                .outputLayer(4, true)
        }

        val noBias = (1..3).map {
            Network.Builder()
                .name("${commonName}_NoBias_$it")
                .setDefaultActivation(Activation.Sigmoid)
                .inputs(4)
                .hiddenLayer(it, false)
                .outputLayer(4, false)
        }

        return Stream.concat(biased.stream(), noBias.stream()).toList()
    }

    override fun buildTeachers(): List<NetworkTeacher> = generateSequence { sharedTeacher }.take(6).toList()

    override fun unfreezing(): List<Network> {

        val unfreezedNetworks = mutableListOf<Network>()

        for (i in 1..3) {

            NetworkFreezer.unfreezeFile("${commonName}_Bias_$i")?.let {
                unfreezedNetworks.add(it)
            }
        }

        for (i in 1..3) {

            NetworkFreezer.unfreezeFile("${commonName}_NoBias_$i")?.let {
                unfreezedNetworks.add(it)
            }
        }

        return if (unfreezedNetworks.size != 6) {
            emptyList()
        } else {
            unfreezedNetworks
        }
    }

    override fun eachLearningStep(network: Network, teacher: NetworkTeacher, errorVector: List<Double>, steps: Int) {
        errorCollector.collect(network, errorVector, steps)
    }

    override fun allNetworksReady(restored: Boolean) {

        println("Networks answers: ")
        for (network in networks) {

            println("\t${network.name}: ")

            for ((input, _) in sharedTeacher.verificationSet) {
                println("\t\t $input -> ${network.answer(input).map { it.round(3) }}")
            }
        }

        if (restored) return

        val errors = errorCollector.getNetworksPlotableErrorMap()

        val biasErrors = errors
            .filter { it.key.name.contains("_Bias") }
            .map { it.key.name.last() + " neurons" to it.value }
            .toMap()
            .toSortedMap()

        val noBiasErrors = errors
            .filter { it.key.name.contains("_NoBias") }
            .map { it.key.name.last() + " neurons" to it.value }
            .toMap()
            .toSortedMap()

        plotMultiple(biasErrors, "Bias") {
            styler.xAxisDecimalPattern = "###,###,###,###"
            styler.yAxisDecimalPattern = "0.00"
        }
        plotMultiple(noBiasErrors, "No bias") {
            styler.xAxisDecimalPattern = "###,###,###,###"
            styler.yAxisDecimalPattern = "0.00"
        }

    }
}