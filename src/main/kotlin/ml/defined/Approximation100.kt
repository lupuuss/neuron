package ml.defined

import ml.defined.base.ClusterLearning
import ml.defined.base.Config
import ml.learn.NetworkTeacher
import ml.output.ClusterErrorCollector
import ml.spine.Activation
import ml.spine.Network

class Approximation100(config: Config) : ClusterLearning(config) {

    override val errorGoal: Double = 0.001
    override val stepsLimit: Int = 10_000

    private lateinit var training1: List<Pair<List<Double>, List<Double>>>
    private lateinit var training2: List<Pair<List<Double>, List<Double>>>
    private lateinit var verification: List<Pair<List<Double>, List<Double>>>


    override fun setup() {
        training1 = dataParser.parse(config.inputs[0], 1, 1)
        training2 = dataParser.parse(config.inputs[1], 1, 1)
        verification = dataParser.parse(config.inputs[2], 1, 1)
    }

    override fun buildTeachers(): List<NetworkTeacher> {
        val teachers1 = generateSequence {
            NetworkTeacher.get(config.teacherMode, 0.01, 0.8).apply {
                trainingSet = training1
                verificationSet = verification
            }
        }.take(10).toList()

        val teachers2 = generateSequence {
            NetworkTeacher.get(config.teacherMode, 0.01, 0.8).apply {
                trainingSet = training2
                verificationSet = verification
            }
        }.take(10).toList()

        return teachers1 + teachers2
    }

    override fun buildClusters(): List<List<Network>> {

        val clusters = mutableListOf<List<Network>>()

        for (fileIndex in 1..2) {

            for (neurons in 11..20) {

                val cluster = mutableListOf<Network>()

                for (i in 1..100) {

                    cluster.add(
                        Network.Builder()
                            .inputs(1)
                            .name("Approximation_file_${fileIndex}_${neurons}_$i")
                            .hiddenLayer(neurons, true, Activation.Sigmoid)
                            .outputLayer(1, true, Activation.Identity)
                    )
                }

                clusters.add(cluster)
            }
        }

        return clusters
    }

    override fun afterLearningCluster(teacher: NetworkTeacher, cluster: List<Network>) {
        println(cluster.joinToString { it.name })
    }
}