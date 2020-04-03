package ml.defined

import ml.defined.base.ClusterLearning
import ml.defined.base.Config
import ml.input.DataParser
import ml.learn.NetworkTeacher
import ml.spine.Activation
import ml.spine.Network
import java.util.*

class Transformation100(config: Config) : ClusterLearning(config) {

    override val errorGoal: Double = 0.001
    override val stepsLimit: Int = 1_000_000

    private lateinit var data: List<Pair<List<Double>, List<Double>>>

    override fun setup() {
        data = dataParser.parse(config.inputs[0], 4, 4)
    }

    override fun buildTeachers(): List<NetworkTeacher> {

        val teachers = mutableListOf<NetworkTeacher>()
        val coefficients = listOf(0.2, 0.4, 0.6)

        for (alpha in coefficients) {

            for (beta in coefficients) {
                teachers.add(
                    NetworkTeacher.get(config.teacherMode, alpha, beta).apply {
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

    override fun afterLearningCluster(teacher: NetworkTeacher, cluster: List<Network>) {

    }

    override fun allNetworksReady(restored: Boolean) {
        for (teacher in teachers) {
            println("${teacher.alpha} ${teacher.beta} ${clusterErrorCollector.meanData(teacher)}")
        }
    }
}