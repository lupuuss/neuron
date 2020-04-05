package ml.defined

import ml.defined.base.ClusterLearning
import ml.defined.base.Config
import ml.defined.base.UnfulfilledExpectationsException
import ml.input.LearnData
import ml.learn.NetworkTeacher
import ml.round
import ml.spine.Activation
import ml.spine.Network
import ml.standardDeviation

class Approximation100(config: Config) : ClusterLearning(config) {

    override val errorGoal: Double = 0.001
    override val stepsLimit: Int = 50_000

    private val teacherMode = NetworkTeacher.Mode.Offline
    private val alpha = 0.01
    private val beta = 0.8

    private lateinit var training: List<LearnData>
    private lateinit var verification: LearnData

    private val trainingErrorMap: MutableMap<NetworkTeacher, Double> = mutableMapOf()
    private val trainingSD: MutableMap<NetworkTeacher, Double> = mutableMapOf()

    override fun setup() {

        if (config.inputs.size != 3) {
            throw UnfulfilledExpectationsException("Approximation task requires training data (2 files) and verification data (1 file) ")
        }

        training = listOf(dataParser.parse(config.inputs[0], 1, 1), dataParser.parse(config.inputs[1], 1, 1))
        verification = dataParser.parse(config.inputs[2], 1, 1)
    }

    override fun buildTeachers(): List<NetworkTeacher> {
        var i = 1
        val teachers1 = generateSequence {
            NetworkTeacher.get(teacherMode, alpha, beta).apply {
                trainingSet = training[0]
                verificationSet = verification
                name = "File_1_${i++}"
            }
        }.take(20).toList()

        i = 1
        val teachers2 = generateSequence {
            NetworkTeacher.get(teacherMode, alpha, beta).apply {
                trainingSet = training[1]
                verificationSet = verification
                name = "File_2_${i++}"
            }
        }.take(20).toList()

        return teachers1 + teachers2
    }

    override fun buildClusters(): List<List<Network>> {

        val clusters = mutableListOf<List<Network>>()

        for (fileIndex in 1..2) {

            for (neurons in 1..20) {

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
        val errors = cluster.map { teacher.verifyTraining(it).average() }
        val avg = errors.average()
        trainingErrorMap[teacher] = avg
        trainingSD[teacher] = errors.standardDeviation(avg)
    }

    override fun allNetworksReady() {

        for (teacher in teachers) {

            print("${teacher.name} = Training Error: ${trainingErrorMap[teacher]!!.round(3)}")
            print("+- ${trainingSD[teacher]!!.round(3)}")

            clusterErrorCollector.meanData(teacher).let { (squared, sd, iter) ->
                println(" | Verification Error: $squared +- $sd Iters: $iter")
            }
        }
    }
}