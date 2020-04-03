package ml.defined

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

class Transformation(config: Config) : DefinedLearning(config) {

    override val errorGoal: Double = 0.001
    override val stepsLimit: Int = 20_000

    private val sharedTeacher = NetworkTeacher.get(config.teacherMode, 0.1, 0.1)

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

            for ((input, expected) in sharedTeacher.verificationSet) {
                println("\t\t $input -> ${network.answer(input).map { it.round(3) }} | $expected")
            }
        }

        if (restored) return

        val errors = errorCollector.getAveragePlotableErrorMap().sortedWith(kotlin.Comparator { o1, o2 ->
            when {
                o1.first.name.last() != o2.first.name.last() -> {
                    o1.first.name.last().compareTo(o2.first.name.last())
                }
                o1.first.name.contains("NoBias") -> {
                    -1
                }
                else -> {
                    1
                }
            }

        })

        val (firstNetwork, firstErrors) = errors.first()
        val remToPlot = errors.stream().skip(1).toList().toMap()

        firstErrors.quickPlotDisplay(firstNetwork.name) { _ ->

            styler.xAxisDecimalPattern = "###,###,###,###"
            styler.yAxisDecimalPattern = "0.00"

            remToPlot.forEach { (network, data) ->

                addSeries(network.name, data.keys.toDoubleArray(), data.values.toDoubleArray())
                seriesMap[network.name]!!.marker = None()
            }
        }
    }
}