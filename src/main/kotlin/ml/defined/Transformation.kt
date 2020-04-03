package ml.defined

import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.input.DataParser
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.quickPlotDisplay
import ml.round
import ml.spine.Activation
import ml.spine.Network
import org.knowm.xchart.style.markers.None
import java.util.*
import java.util.stream.Stream
import kotlin.streams.toList

class Transformation(config: Config) : NetworksLearning(config) {

    override val errorGoal: Double = 0.000001
    override val stepsLimit: Int = 50_000

    private val sharedTeacher = NetworkTeacher.get(config.teacherMode, 0.1, 0.0)

    override fun setup() {
        val parser = DataParser(config.separator, 4, 4)
        val data = parser.parse(Scanner(config.inputs.first()))

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

    override fun eachLearningStep(network: Network, errorVector: List<Double>, steps: Int) {
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

        val errors = errorCollector.getAveragePlotableErrorMap()

        val biasErrors = errors.filter { it.first.name.contains("_Bias") }.sortedBy { it.first.name }
        val noBiasErrors = errors.filter { it.first.name.contains("_NoBias") }.sortedBy { it.first.name }

        plotsErrors(biasErrors, "Bias")
        plotsErrors(noBiasErrors, "No bias")

    }

    private fun plotsErrors(data: List<Pair<Network, Map<Double, Double>>>, title: String) {

        val (firstNetwork, firstErrors) = data.first()
        val remToPlot = data.stream().skip(1).toList().toMap()

        firstErrors.quickPlotDisplay(generatePlotName(firstNetwork)) { _ ->

            this.title = title
            styler.xAxisDecimalPattern = "###,###,###,###"
            styler.yAxisDecimalPattern = "0.00"

            remToPlot.forEach { (network, data) ->

                val plotName = generatePlotName(network)
                addSeries(plotName, data.keys.toDoubleArray(), data.values.toDoubleArray())
                seriesMap[plotName]!!.marker = None()
            }
        }
    }

    private fun generatePlotName(network: Network) = "${network.name.last()} neurons"
}