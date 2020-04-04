package ml.output

import kotlin.math.roundToInt

class ConsoleLoader(private val characters: Int) {

    private var progress = 0.0

    fun update(value: Double) {
        progress = value
    }

    private fun repeat(char: Char, times: Int): String =
        generateSequence { char }.take(times).joinToString(separator = "")

    private fun fixedSizePercentage(n: Int): String {
        val percentage = ((progress * 1000).roundToInt() / 10.0).toString()
        val size = percentage.length

        return repeat(' ', n - size) + percentage
    }

    fun print() {
        val progress = (characters.toDouble() * progress).roundToInt()
        val spaces = characters - progress
        print("\r[${repeat('=', progress)}>${repeat(' ', spaces)}] ${fixedSizePercentage(5)}%")
    }

    fun close() {
        println()
    }
}