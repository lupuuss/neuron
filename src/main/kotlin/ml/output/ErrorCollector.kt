package ml.output

import ml.spine.Network
import java.util.concurrent.ConcurrentHashMap

class ErrorCollector {

    private val errorMap: MutableMap<Network, MutableMap<Int, List<Double>>> = ConcurrentHashMap()
    private val namedMap: MutableMap<String, MutableMap<Int, List<Double>>> = ConcurrentHashMap()

    fun collect(network: Network, error: List<Double>, steps: Int) {
        errorMap.getOrPut(network, { mutableMapOf() })[steps] = error
    }

    fun collect(name: String, error: List<Double>, steps: Int) {
        namedMap.getOrPut(name, { mutableMapOf() })[steps] = error
    }

    fun getNetworksPlotableErrorMap(): List<Pair<Network, Map<Double, Double>>> =
        errorMap.map { (network, errors) ->
            network to errors.map { (steps, errorVec) -> steps.toDouble() to errorVec.average() }.toMap()
        }

    fun getNamedPlotableErrorMap(): List<Pair<String, Map<Double, Double>>> =
        namedMap.map { (name, errors) ->
            name to errors.map { (steps, errorVec) -> steps.toDouble() to errorVec.average() }.toMap()
        }
}