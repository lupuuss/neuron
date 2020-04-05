package ml.defined

import ml.defined.base.ClusterLearning
import ml.defined.base.Config
import ml.input.DataParser
import ml.input.LearnData
import ml.learn.NetworkTeacher
import ml.output.Classification
import ml.spine.Activation
import ml.spine.Network
import ml.standardDeviation

class Iris100(config: Config) : ClusterLearning(config) {

    override val errorGoal: Double = 0.3
    override val stepsLimit: Int = 20_000

    private val alpha = 0.01
    private val beta = 0.9
    private val teacherMode = NetworkTeacher.Mode.Offline

    private val classificationLevelMapTraining: MutableMap<NetworkTeacher, Pair<Double, Double>> = mutableMapOf()
    private val classificationLevelMapVerification: MutableMap<NetworkTeacher, Pair<Double, Double>> = mutableMapOf()

    private lateinit var verificationData: LearnData
    private lateinit var trainingData: LearnData

    override fun setup() {
        dataParser.lineTransformer = DataParser.irisTransformer

        val data = dataParser.parse(config.inputs[0], 4, 3)

        trainingData = data.filterIndexed { index, _ -> index % 2 == 0 }
        verificationData = data.filterIndexed { index, _ -> index % 2 != 0 }
    }

    override fun buildTeachers(): List<NetworkTeacher> {
        var i = 1
        return generateSequence {
            NetworkTeacher.get(teacherMode, alpha, beta).apply {
                name = "neurons_${i++}"
                verificationSet = verificationData
                trainingSet = trainingData
            }
        }.take(20).toList()
    }

    override fun buildClusters(): List<List<Network>> {
        val clusters = mutableListOf<List<Network>>()

        for (neurons in 1..20) {
            val cluster = mutableListOf<Network>()

            for (i in 1..100) {
                cluster.add(
                    Network.Builder()
                        .inputs(4)
                        .name("Iris_Progress_${neurons}_$i")
                        .setDefaultActivation(Activation.Sigmoid)
                        .hiddenLayer(neurons, true)
                        .outputLayer(3, true)
                )
            }

            clusters.add(cluster)
        }

        return clusters
    }

    override fun afterLearningCluster(teacher: NetworkTeacher, cluster: List<Network>) {

        val trainingPercentages = cluster.map { Classification.calcLevel(it, teacher.trainingSet) }
        val trainingAvg = trainingPercentages.average()

        classificationLevelMapTraining[teacher] =
            trainingAvg to trainingPercentages.standardDeviation(trainingAvg)

        val verificationPercentages = cluster.map { Classification.calcLevel(it, teacher.verificationSet) }
        val verificationAvg = verificationPercentages.average()

        classificationLevelMapVerification[teacher] =
            verificationAvg to verificationPercentages.standardDeviation(verificationAvg)
    }

    override fun allNetworksReady() {

        classificationLevelMapVerification.forEach { (key, value) ->

            println("${key.name} $value%")
        }

        classificationLevelMapTraining.forEach { (key, value) ->
            println("${key.name} ${value.first}% +- ${value.second}")
        }
    }
}