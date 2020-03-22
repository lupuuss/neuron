package ml

import java.util.*

class DataParser(
    private var separator: String,
    private var inputs: Int,
    private var expected: Int
) {

    fun parse(scanner: Scanner): List<Pair<List<Double>, List<Double>>> {

        val result = mutableListOf<Pair<List<Double>, List<Double>>>()

        scanner.forEach { line ->
            val values = line.split(separator).map { it.toDouble() }

            result.add(values.subList(0, inputs) to values.subList(inputs, inputs + expected))
        }

        return result
    }
}