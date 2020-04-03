package ml.defined

import ml.input.DataParser
import ml.defined.exceptions.UnfulfilledExpectationsException
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.quickPlotDisplay
import ml.spine.Activation
import ml.spine.Network
import ml.step
import org.knowm.xchart.style.markers.Circle
import org.knowm.xchart.style.markers.None
import java.awt.Color
import java.util.*

class Approximation(config: Config) : DefinedLearning(config) {

    override val errorGoal: Double = 0.1
    override val stepsLimit: Int = 100_000

    private val commonName = "Approximation"
    private var trainingDataName = ""

    private val alpha = if (config.teacherMode == NetworkTeacher.Mode.Offline) 0.1 else 0.0005
    private val beta = if (config.teacherMode == NetworkTeacher.Mode.Offline) 0.7 else 0.5
    private val sharedTeacher: NetworkTeacher = NetworkTeacher.get(config.teacherMode, alpha, beta)

    override fun setup() {

        if (config.inputs.size != 2) {
            throw UnfulfilledExpectationsException("Approximation task requires training data and verification data (2 files)")
        }

        asyncRunner = true
        trainingDataName = config.inputs.first().nameWithoutExtension

        val parser = DataParser(config.separator, 1, 1)

        sharedTeacher.trainingSet = parser.parse(Scanner(config.inputs[0]))
        sharedTeacher.verificationSet = parser.parse(Scanner(config.inputs[1]))
    }

    private fun generateName(n: Int) = "${commonName}_${trainingDataName}_$n"

    override fun buildNetworks(): List<Network> = listOf(5, 10, 19).map {
        Network.Builder()
            .name(generateName(it))
            .inputs(1)
            .hiddenLayer(it, true, Activation.Sigmoid)
            .outputLayer(1, true, Activation.Identity)
    }

    override fun eachLearningStep(network: Network, errorVector: List<Double>, steps: Int) {
        errorCollector.collect(network, errorVector, steps)
    }

    override fun buildTeachers(): List<NetworkTeacher> = generateSequence { sharedTeacher }.take(6).toList()

    override fun unfreezing(): List<Network> {

        val result = mutableListOf<Network>()

        listOf(5, 10, 19).forEach { neurons ->
            NetworkFreezer.unfreezeFile(generateName(neurons))?.let {
                result.add(it)
            }
        }

        return result
    }

    private fun networkPlotDataXY(network: Network, range: Iterator<Double>): Map<Double, Double> {

        val result = mutableMapOf<Double, Double>()

        for (x in range) {
            result[x] = network.answer(listOf(x)).first()
        }

        return result
    }

    override fun allNetworksReady(restored: Boolean) {

        val range = -3.0..4.0 step 0.1
        val firstPlot = networkPlotDataXY(networks.first(), range)

        firstPlot.quickPlotDisplay(networks.first().name) { _ ->

            title = "$commonName | File: ${config.inputs[0].name} | Learning coefficient: $alpha | Momentum: $beta"

            for (network in networks.stream().skip(1)) {

                val plot = networkPlotDataXY(network, range)

                addSeries(network.name, plot.keys.toDoubleArray(), plot.values.toDoubleArray())
                seriesMap[network.name]?.marker = None()
            }

            addSeries(
                "Verification data",
                sharedTeacher.verificationSet.map { it.first.first() }.toDoubleArray(),
                sharedTeacher.verificationSet.map { it.second.first() }.toDoubleArray()
            )
            seriesMap["Verification data"]?.apply {
                lineColor = Color(0, 0, 0, 0)
                marker = Circle()
            }

            addSeries(
                "Training data",
                sharedTeacher.trainingSet.map { it.first.first() }.toDoubleArray(),
                sharedTeacher.trainingSet.map { it.second.first() }.toDoubleArray()
            )
            seriesMap["Training data"]?.apply {
                lineColor = Color(0, 0, 0, 0)
                marker = Circle()
            }
        }

        if (restored) {
            return
        }

        val (firstNetwork, firstPlotData) = errorCollector.getAveragePlotableErrorMap().first()

        firstPlotData.quickPlotDisplay(firstNetwork.name) { _ ->

            for ((network, data) in errorCollector.getAveragePlotableErrorMap().stream().skip(1)) {

                addSeries(network.name, data.keys.toDoubleArray(), data.values.toDoubleArray())
                seriesMap[network.name]?.marker = None()
            }

            this.styler.yAxisMax = 100.0
        }
    }
}