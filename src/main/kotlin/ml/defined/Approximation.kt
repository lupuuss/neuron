package ml.defined

import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.defined.base.UnfulfilledExpectationsException
import ml.learn.NetworkTeacher
import ml.spine.Activation
import ml.spine.Network
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
            errorCollector.collect("${network.name}_training", teacher.verifyTraining(network), steps)
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

        if (restored) return

        val errors = errorCollector.getNetworksPlotableErrorMap().map { it.first.name to it.second }
        val trainingError = errorCollector.getNamedPlotableErrorMap()

        plotMultiple(preparePlotDataForFile(1, errors, trainingError), "File 1") {
            styler.xAxisDecimalPattern = "###,###,###,###"
            styler.yAxisDecimalPattern = "0.00"
            styler.yAxisMax = 300.0
        }
        plotMultiple(preparePlotDataForFile(2, errors, trainingError), "File 2") {
            styler.xAxisDecimalPattern = "###,###,###,###"
            styler.yAxisDecimalPattern = "0.00"
            styler.yAxisMax = 300.0
        }
    }

    private fun preparePlotDataForFile(
        fileN: Int,
        verSet: List<Pair<String, Map<Double, Double>>>,
        trainingSet: List<Pair<String, Map<Double, Double>>>
    ): List<Pair<String, Map<Double, Double>>> {

        val fileVerPlotData = verSet
            .filter { it.first.contains("error_$fileN") }
            .map { it.first.substringAfter("error_${fileN}_") + " neurons (verification set)" to it.second }
            .sortedBy { it.first.substringBefore(" ").toDouble() }

        val fileTrainingPlotData = trainingSet
            .filter { it.first.contains("error_$fileN") }
            .map {
                val n = it.first.substringAfter("error_${fileN}_").substringBefore("_training")
                "$n neurons (training set)" to it.second
            }.sortedBy { it.first.substringBefore(" ").toDouble() }

        return fileVerPlotData + fileTrainingPlotData
    }
}