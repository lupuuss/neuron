package ml.defined

import ml.addSeries
import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.defined.base.UnfulfilledExpectationsException
import ml.learn.NetworkTeacher
import ml.spine.Activation
import ml.spine.Network
import org.knowm.xchart.style.markers.None
import java.util.stream.Stream
import kotlin.streams.asStream
import kotlin.streams.toList

class Approximation(config: Config) : NetworksLearning(config) {

    override val errorGoal: Double = 0.1
    override val stepsLimit: Int = 100_000

    private val commonName = "Approximation"

    private val alpha = if (config.teacherMode == NetworkTeacher.Mode.Offline) 0.01 else 0.0005
    private val beta = if (config.teacherMode == NetworkTeacher.Mode.Offline) 0.9 else 0.5

    private val sharedTeacher1: NetworkTeacher = NetworkTeacher.get(config.teacherMode, alpha, beta)
    private val sharedTeacher2: NetworkTeacher = NetworkTeacher.get(config.teacherMode, alpha, beta)

    private val markedToErrorCollecting = mutableListOf<Network>()

    override fun setup() {

        if (config.inputs.size != 3) {
            throw UnfulfilledExpectationsException("Approximation task requires training data (2 files) and verification data (1 file) ")
        }

        asyncRunner = true

        val verificationData = dataParser.parse(config.inputs[2], 1, 1)

        sharedTeacher1.trainingSet = dataParser.parse(config.inputs[0], 1, 1)
        sharedTeacher1.verificationSet = verificationData

        sharedTeacher2.trainingSet = dataParser.parse(config.inputs[1], 1, 1)
        sharedTeacher2.verificationSet = verificationData
    }

    private fun generateName(n: Int, prefix: String, trainingDataName: String) =
        "${commonName}_${prefix}_${trainingDataName}_$n"

    override fun buildNetworks(): List<Network> {
        val allNetworks = mutableListOf<Network>()

        for (trainingSetIndex in 1..2) {

            val networks1 = listOf(1, 5, 9, 14, 17).map { neurons ->
                Network.Builder()
                    .name(generateName(neurons, "result", trainingSetIndex.toString()))
                    .inputs(1)
                    .hiddenLayer(neurons, true, Activation.Sigmoid)
                    .outputLayer(1, true, Activation.Identity)
            }

            val networks2 = listOf(1, 5, 19).map { neurons ->
                Network.Builder()
                    .name(generateName(neurons, "error", trainingSetIndex.toString()))
                    .inputs(1)
                    .hiddenLayer(neurons, true, Activation.Sigmoid)
                    .outputLayer(1, true, Activation.Identity)
            }

            markedToErrorCollecting.addAll(networks2)

            allNetworks.addAll(networks1)
            allNetworks.addAll(networks2)
        }

        return allNetworks
    }

    override fun buildTeachers(): List<NetworkTeacher> = Stream.concat(
        generateSequence { sharedTeacher1 }.take(8).asStream(),
        generateSequence { sharedTeacher2 }.take(8).asStream()
    ).toList()

    override fun eachLearningStep(network: Network, teacher: NetworkTeacher, errorVector: List<Double>, steps: Int) {

        if (markedToErrorCollecting.contains(network)) {
            errorCollector.collect(network, errorVector, steps)
            errorCollector.collect("${network.name}_test", teacher.verifyTest(network), steps)
        }
    }

    override fun unfreezing(): List<Network> {
        val names = mutableListOf<String>()
        for (trainingSetIndex in 1..2) {

            val networks1 = listOf(1, 5, 9, 14, 17).map { neurons ->
                generateName(neurons, "result", trainingSetIndex.toString())
            }

            val networks2 = listOf(1, 5, 19).map { neurons ->
                generateName(neurons, "error", trainingSetIndex.toString())
            }

            names.addAll(networks1)
            names.addAll(networks2)
        }

        return baseUnfreezing(names)
    }

    private fun networkPlotDataXY(network: Network, range: Iterator<Double>): Map<Double, Double> {

        val result = mutableMapOf<Double, Double>()

        for (x in range) {
            result[x] = network.answer(listOf(x)).first()
        }

        return result
    }

    override fun allNetworksReady(restored: Boolean) {

/*
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
                sharedTeacher1.verificationSet.map { it.first.first() }.toDoubleArray(),
                sharedTeacher1.verificationSet.map { it.second.first() }.toDoubleArray()
            )
            seriesMap["Verification data"]?.apply {
                lineColor = Color(0, 0, 0, 0)
                marker = Circle()
            }

            addSeries(
                "Training data",
                sharedTeacher1.trainingSet.map { it.first.first() }.toDoubleArray(),
                sharedTeacher1.trainingSet.map { it.second.first() }.toDoubleArray()
            )
            seriesMap["Training data"]?.apply {
                lineColor = Color(0, 0, 0, 0)
                marker = Circle()
            }
        }
*/

        if (restored) return

        val errors = errorCollector.getNetworksPlotableErrorMap()
        val trainingError = errorCollector.getNamedPlotableErrorMap()

        plotsErrors(
            errors.filter { it.first.name.contains("error_1") },
            "Errors for file 1",
            { it.name.last() + " neurons" }
        ) {
            styler.yAxisMax = 400.0
            trainingError
                .filter { it.first.contains("error_1") }
                .forEach { (name, data) ->

                    this.addSeries(name, data)
                    this.seriesMap[name]!!.marker = None()
                }
        }

        plotsErrors(
            errors.filter { it.first.name.contains("error_2") },
            "Errors for file 2",
            { it.name.last() + " neurons" }
        ) {
            styler.yAxisMax = 400.0
            trainingError
                .filter { it.first.contains("error_2") }
                .forEach { (name, data) ->

                    this.addSeries(name, data)
                    this.seriesMap[name]!!.marker = None()
                }
        }
    }
}