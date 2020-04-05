package ml.defined

import ml.defined.base.ClusterLearning
import ml.defined.base.Config
import ml.learn.NetworkTeacher
import ml.spine.Network

class Iris100(config: Config) : ClusterLearning(config) {

    override val errorGoal: Double = 0.3
    override val stepsLimit: Int = 10_000

    private val sharedTeacher = NetworkTeacher.get(NetworkTeacher.Mode.Offline, 0.01, 0.1)

    override fun setup() {

    }

    override fun buildTeachers(): List<NetworkTeacher> {
        TODO("Not yet implemented")
    }

    override fun buildClusters(): List<List<Network>> {
        TODO("Not yet implemented")
    }
}