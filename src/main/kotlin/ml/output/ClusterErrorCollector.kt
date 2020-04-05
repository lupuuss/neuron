package ml.output

import ml.learn.NetworkTeacher
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.roundToInt
import kotlin.math.sqrt

class ClusterErrorCollector {

    private val errorsMap: MutableMap<NetworkTeacher, MutableList<Pair<List<Double>, Int>>> = ConcurrentHashMap()

    fun putErrors(teacher: NetworkTeacher, errors: List<Double>, steps: Int) {

        errorsMap.getOrPut(teacher) { mutableListOf() }.add(errors to steps)
    }

    fun meanData(teacher: NetworkTeacher): MeanData {
        val iters = errorsMap[teacher]!!.map { it.second }.average().roundToInt()
        val squaredErrorAvg = errorsMap[teacher]!!.map { it.first.average() }.average()
        val variation = errorsMap[teacher]!!.map {
            (it.first.average() - squaredErrorAvg).let { it * it }
        }.sum() / errorsMap.size
        val standardDeviation = sqrt(variation)

        return MeanData(
            squaredErrorAvg,
            standardDeviation,
            iters
        )
    }
}

data class MeanData(
    val squaredError: Double,
    val standardDeviation: Double,
    val iterations: Int
)