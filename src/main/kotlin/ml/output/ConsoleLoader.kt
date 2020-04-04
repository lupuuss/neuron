package ml.output

import ml.ConsoleAnimation
import kotlin.math.roundToInt

class ConsoleLoader(
    private val characters: Int,
    var animation: Boolean = true,
    private val mode: Mode
) {

    enum class Mode {
        Percentage, Fraction
    }

    private var current = 0
    private var max = 0
    private val animationIter = ConsoleAnimation.frames()

    fun update(current: Int, max: Int) {
        this.current = current
        this.max = max
    }

    private fun repeat(char: Char, times: Int): String =
        generateSequence { char }.take(times).joinToString(separator = "")

    private fun fixedSizePercentage(progress: Double, n: Int): String {
        val percentage = ((progress * 1000).roundToInt() / 10.0).toString()
        val size = percentage.length

        return repeat(' ', n - size) + percentage + "%"
    }

    fun print() {
        var progressString = "\rLearning:"
        val progress = current.toDouble() / max.toDouble()

        if (characters != 0) {
            val barProgress = (characters.toDouble() * progress).roundToInt()
            val spaces = characters - barProgress
            progressString += " [${repeat('=', barProgress)}>${repeat(' ', spaces)}]"
        }

        if (animation) {
            progressString += " (${animationIter.next()})"
        }

        progressString += when (mode) {
            Mode.Percentage -> " " + fixedSizePercentage(progress, 5)
            Mode.Fraction -> " [$current/$max]"
        }

        print(progressString)
    }

    fun close() {
        print("\r")
    }
}