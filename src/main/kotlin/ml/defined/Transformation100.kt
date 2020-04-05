package ml.defined

import ml.defined.base.ClusterLearning
import ml.defined.base.Config
import ml.input.LearnData
import ml.learn.NetworkTeacher
import ml.round
import ml.spine.Activation
import ml.spine.Network

class Transformation100(config: Config) : ClusterLearning(config) {

    override val errorGoal: Double = 0.001
    override val stepsLimit: Int = 1_000_000

    private val teacherMode = NetworkTeacher.Mode.Online

    private lateinit var data: LearnData

    override fun setup() {
        data = dataParser.parse(config.inputs[0], 4, 4)
    }

    override fun buildTeachers(): List<NetworkTeacher> {

        val teachers = mutableListOf<NetworkTeacher>()
        val coefficients = listOf(0.2, 0.4, 0.6, 0.95)

        for (alpha in coefficients) {

            for (beta in coefficients) {
                teachers.add(
                    NetworkTeacher.get(teacherMode, alpha, beta).apply {
                        trainingSet = data
                        verificationSet = data
                    }
                )
            }
        }

        return teachers
    }

    override fun buildClusters(): List<List<Network>> = teachers.map { teacher ->
        var i = 0
        generateSequence {
            Network.Builder()
                .inputs(4)
                .name("Transformation_${teacher.alpha}_${teacher.beta}_${i++}")
                .setDefaultActivation(Activation.Sigmoid)
                .hiddenLayer(2, true)
                .outputLayer(4, true)
        }.take(100).toList()
    }

    override fun allNetworksReady() {
        for (teacher in teachers) {
            val meanData = clusterErrorCollector.meanData(teacher)
            println("${teacher.alpha} ${teacher.beta} ${meanData.squaredError} ${meanData.rootSquareError.round(4)} ${meanData.iterations}")
        }
    }
}