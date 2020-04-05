package ml.defined

import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.input.DataParser
import ml.learn.NetworkTeacher
import ml.output.Classification
import ml.plotMultiple
import ml.spine.Activation
import ml.spine.Network
import java.util.*
import java.util.concurrent.ConcurrentHashMap

class Iris(config: Config) : NetworksLearning(config) {
    override val errorGoal: Double = 0.1
    override val stepsLimit: Int = 30_000

    private lateinit var data: List<Pair<List<Double>, List<Double>>>

    private lateinit var fieldMask: List<Boolean>
    private lateinit var fieldMaskString: String

    private val teacherMode = NetworkTeacher.Mode.Offline
    private val alpha = 0.03
    private val beta = 0.5

    private val localTeachers = listOf(
        NetworkTeacher.get(teacherMode, alpha, beta),
        NetworkTeacher.get(teacherMode, alpha, beta)
    )

    private val classificationErrorTraining: MutableMap<Network, MutableMap<Int, Double>> = ConcurrentHashMap()
    private val classificationErrorVerification: MutableMap<Network, MutableMap<Int, Double>> = ConcurrentHashMap()

    override fun setup() {

        dataParser.lineTransformer = DataParser.irisTransformer

        println("Fields mask: ")
        val userInput = Scanner(System.`in`)

        while (!userInput.hasNext("[0,1]{4}")) {
            userInput.nextLine()
            println("Invalid mask!")
        }

        fieldMaskString = userInput.nextLine()
        fieldMask = fieldMaskString.map { it == '1' }

        data = dataParser.parse(config.inputs[0], 4, 3).map {
            val result = mutableListOf<Double>()

            for (i in it.first.indices) {
                if (fieldMask[i]) {
                    result.add(it.first[i])
                }
            }

            result to it.second
        }

    }

    override fun buildNetworks(): List<Network> {

        val allNetworks = mutableListOf<Network>()

        for (i in listOf(50, 80)) {

            allNetworks.addAll(
                listOf(1, 5, 9, 13, 17).map { hiddenNeurons ->
                    Network.Builder()
                        .name("Iris_${fieldMaskString}_${i}_$hiddenNeurons")
                        .inputs(fieldMask.count { it })
                        .setDefaultActivation(Activation.Sigmoid)
                        .hiddenLayer(hiddenNeurons, true)
                        .outputLayer(3, true)
                }
            )
        }

        return allNetworks
    }

    override fun buildTeachers(): List<NetworkTeacher> {
        localTeachers[0].apply {
            trainingSet = data.filterIndexed { index, _ -> index % 2 != 0 }
            verificationSet = data.filterIndexed { index, _ -> index % 2 == 0 }
        }

        localTeachers[1].apply {
            trainingSet = data.filterIndexed { index, _ -> index % 5 != 0 }
            verificationSet = data.filterIndexed { index, _ -> index % 5 == 0 }
        }

        return generateSequence { localTeachers[0] }.take(5).toList() +
                generateSequence { localTeachers[1] }.take(5).toList()
    }

    private fun saveClassificationLevel(network: Network, teacher: NetworkTeacher, steps: Int) {
        classificationErrorTraining
            .getOrPut(network, { mutableMapOf() })[steps] = Classification.calcLevel(network, teacher.trainingSet)

        classificationErrorVerification
            .getOrPut(network, { mutableMapOf() })[steps] = Classification.calcLevel(network, teacher.verificationSet)
    }

    override fun beforeLearning(network: Network, teacher: NetworkTeacher) {
        saveClassificationLevel(network, teacher, 0)
    }

    override fun eachLearningStep(network: Network, teacher: NetworkTeacher, errorVector: List<Double>, steps: Int) {
        saveClassificationLevel(network, teacher, steps)
    }

    override fun afterLearning(network: Network, teacher: NetworkTeacher, errorVector: List<Double>, steps: Int) {
        saveClassificationLevel(network, teacher, steps)
    }

    override fun allNetworksReady() {

        classificationErrorTraining.map { it.key to it.value.entries.last() }.forEach {
            println("${it.first.name} ${it.second.value}%")
        }
        println()
        classificationErrorVerification.map { it.key to it.value.entries.last() }.forEach {
            println("${it.first.name} ${it.second.value}%")
        }

        for (percentage in listOf(50, 80)) {

            val plotData = mutableMapOf<String, Map<Double, Double>>()

            val prefix = "_${percentage}_"

            plotData.putAll(
                classificationErrorVerification
                    .filter { it.key.name.contains(prefix) }
                    .map { entry ->
                        val name = entry.key.name.substringAfter(prefix) + " neurons (verification)"
                        val mappedValue = entry.value.map { it.key.toDouble() to it.value }.toMap()
                        name to mappedValue
                    }
                    .sortedBy { it.first.substringBefore(" ").toDouble() }
                    .toMap()
            )


            plotData.putAll(
                classificationErrorTraining
                    .filter { it.key.name.contains(prefix) }
                    .map { entry ->
                        val name = entry.key.name.substringAfter(prefix) + " neurons (training)"
                        val mappedValue = entry.value.map { it.key.toDouble() to it.value }.toMap()
                        name to mappedValue
                    }
                    .sortedBy { it.first.substringBefore(" ").toDouble() }
                    .toMap()
            )

            plotMultiple(plotData, "Training data $percentage% | Fields mask: $fieldMaskString")
        }
    }

}