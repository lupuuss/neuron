@file:Suppress("unused")

package ml

import java.lang.IllegalArgumentException
import kotlin.math.pow
import kotlin.math.round

fun Pair<DoubleArray, DoubleArray>.meanSquaredError(): Double {
    if (first.size != second.size) {
        throw IllegalArgumentException("Incompatible sizes of arrays (${first.size} != ${second.size})")
    }

    var sum = 0.0

    for (i in first.indices) {
        sum += (first[i] - second[i]).let { it * it }
    }

    return sum / first.size.toDouble()
}

fun Boolean.asInt(): Int = if (this) 1 else 0

fun <K, V> Map<K, V>.firstValue(): V {

    return this.entries.first().value
}

operator fun ClosedFloatingPointRange<Double>.iterator(): Iterator<Double> {

    return this step 1.0
}

infix fun ClosedFloatingPointRange<Double>.step(step: Double): Iterator<Double> {

    return object : Iterator<Double> {

        private val start = this@step.start
        private val end = this@step.endInclusive
        private var next = this@step.start

        override fun hasNext(): Boolean {
            val hadNext = next < end

            if (!hadNext) {
                next = start
            }

            return hadNext
        }

        override fun next(): Double = next.also { this.next = it + step }

    }
}

infix fun List<Double>.difference(expected: List<Double>): List<Double> {
    val result = MutableList(this.size) { 0.0 }

    for (i in this.indices) {
        result[i] = this[i] - expected[i]
    }

    return result
}

fun Double.round(n: Int): Double = 10.0.pow(n).let { round(this * it) / it }