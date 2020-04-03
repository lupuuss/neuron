package ml.defined.base

import kotlinx.coroutines.*
import ml.learn.NetworkTeacher
import ml.output.ClusterErrorCollector
import ml.output.ConsoleLoader
import ml.spine.Network

abstract class ClusterLearning(config: Config) : Learning(config) {

    protected val clusters: MutableList<List<Network>> = mutableListOf()

    protected val clusterErrorCollector = ClusterErrorCollector()

    protected abstract fun buildClusters(): List<List<Network>>

    protected open fun beforeLearningCluster(teacher: NetworkTeacher, cluster: List<Network>) {}

    protected open fun afterLearningCluster(teacher: NetworkTeacher, cluster: List<Network>) {}

    override fun run() {

        setup()

        teachers.addAll(buildTeachers())
        clusters.addAll(buildClusters())

        val awaits = mutableListOf<Deferred<Triple<NetworkTeacher, List<Double>, Int>>>()

        for (teacherIndex in teachers.indices) {

            beforeLearningCluster(teachers[teacherIndex], clusters[teacherIndex])

            for (network in clusters[teacherIndex]) {
                val teacher = teachers[teacherIndex]

                val await = GlobalScope.async {
                    val (errors, steps) = learningProcess(network, teacher)

                    Triple(teacher, errors, steps)
                }

                awaits.add(await)
            }
        }

        val clustersCounters = teachers.map { it to 0 }.toMap().toMutableMap()
        val all = awaits.size
        val loader = ConsoleLoader(50)

        runBlocking {
            while (awaits.size != 0) {

                val finished = awaits.filter { it.isCompleted }
                awaits.removeAll(finished)

                finished.map {
                    it.await()
                }.forEach { (teacher, errors, steps) ->
                    clustersCounters.compute(teacher) { _, counter -> counter!! + 1 }
                    clusterErrorCollector.putErrors(teacher, errors, steps)
                }

                val clustersDone = clustersCounters.entries.filterIndexed { index, entry ->
                    clusters[index].size == entry.value
                }

                clustersDone.forEach {
                    clustersCounters.remove(it.key)
                    afterLearningCluster(it.key, clusters[teachers.indexOf(it.key)])
                }

                loader.update((all - awaits.size).toDouble() / all.toDouble())
                loader.print()

                delay(300)
            }
        }

        loader.close()

        allNetworksReady(false)
    }
}