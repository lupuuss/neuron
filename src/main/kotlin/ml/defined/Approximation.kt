package ml.defined

import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.defined.base.UnfulfilledExpectationsException
import ml.learn.NetworkTeacher
import ml.spine.Activation
import ml.spine.Network
import ml.step
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

    private fun getRange() = -2.0..3.0 step 0.1

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


    override fun allNetworksReady() {

        val result1Networks = networks.filter { it.name.contains("result_1") }
        val result2Networks = networks.filter { it.name.contains("result_2") }

        plotNetworks(result1Networks, 1)
        plotNetworks(result2Networks, 2)

        val errors = errorCollector.getNetworksPlotableErrorMap().map { it.key.name to it.value }.toMap()
        val trainingError = errorCollector.getNamedPlotableErrorMap()

        for (i in 1..2) {
            plotMultiple(preparePlotDataForFile(i, errors, trainingError), "File $i") {
                styler.xAxisDecimalPattern = "###,###,###,###"
                styler.yAxisDecimalPattern = "0.00"
                styler.yAxisMax = 300.0
            }
        }

    }

    private fun preparePlotDataForFile(
        fileN: Int,
        verSet: Map<String, Map<Double, Double>>,
        trainingSet: Map<String, Map<Double, Double>>
    ): Map<String, Map<Double, Double>> {

        val fileVerPlotData = verSet
            .filter { it.key.contains("error_$fileN") }
            .map { it.key.substringAfter("error_${fileN}_") + " neurons (verification set)" to it.value }
            .sortedBy { it.first.substringBefore(" ").toDouble() }
            .toMap()

        val fileTrainingPlotData = trainingSet
            .filter { it.key.contains("error_$fileN") }
            .map {
                val n = it.key.substringAfter("error_${fileN}_").substringBefore("_training")
                "$n neurons (training set)" to it.value
            }.sortedBy { it.first.substringBefore(" ").toDouble() }

        return (fileVerPlotData + fileTrainingPlotData).toMap()
    }

    private fun plotNetworks(networks: List<Network>, fileN: Int) {
        val plots: MutableMap<String, Map<Double, Double>> = mutableMapOf()

        for (network in networks) {
            val name = network.name.substringAfter("result_${fileN}_") + " neurons"
            plots[name] = networkPlotDataXY(network, getRange())
        }

        plotMultiple(plots, "File $fileN results")
    }
}