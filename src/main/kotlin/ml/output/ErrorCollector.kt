package ml.output

import ml.spine.Network

class ErrorCollector {

    val errorMap: MutableMap<Network, MutableMap<Int, List<Double>>> = mutableMapOf()

    fun collect(network: Network, error: List<Double>, steps: Int) {
        errorMap.getOrPut(network, { mutableMapOf() })[steps] = error
    }

    fun getAveragePlotableErrorMap(): List<Pair<Network, Map<Double, Double>>> =
        errorMap.map { (network, errors) ->
            network to errors.map { (steps, errorVec) -> steps.toDouble() to errorVec.average() }.toMap()
        }
}