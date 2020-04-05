package ml.output

import ml.spine.Network
import kotlin.math.roundToInt

object Classification {

    fun calcLevel(network: Network, data: List<Pair<List<Double>, List<Double>>>): Double {
        var good = 0
        for ((input, expected) in data) {

            val answer = network.answer(input).map { it.roundToInt().toDouble() }

            if (answer == expected) {
                good++
            }
        }

        return (good.toDouble() / data.size.toDouble()) * 100
    }
}