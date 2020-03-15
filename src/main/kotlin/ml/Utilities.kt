@file:Suppress("unused")

package ml

import org.knowm.xchart.BitmapEncoder
import org.knowm.xchart.QuickChart
import org.knowm.xchart.SwingWrapper
import org.knowm.xchart.XYChart
import java.lang.Exception
import java.lang.IllegalArgumentException

fun Map<Double, Double>.quickPlot(seriesName: String): XYChart = QuickChart.getChart(
    "",
    "",
    "",
    seriesName,
    keys.toDoubleArray(),
    values.toDoubleArray()
)

fun Map<Double, Double>.quickPlotSwingWrapped(seriesName: String): SwingWrapper<XYChart> = SwingWrapper(
    QuickChart.getChart(
        "",
        "",
        "",
        seriesName,
        keys.toDoubleArray(),
        values.toDoubleArray()
    )
)

fun Map<Double, Double>.quickPlotDisplay(
    seriesName: String,
    plotManipulation: (XYChart, SwingWrapper<XYChart>) -> Unit
) {

    val chart = this.quickPlot(seriesName)
    val wrapper = SwingWrapper(chart)
    plotManipulation(chart, wrapper)
    wrapper.displayChart()
}

fun Map<Double, Double>.quickPlotDisplay(seriesName: String, plotManipulation: (XYChart) -> Unit) {

    this.quickPlotDisplay(seriesName) { chart, _ -> plotManipulation(chart) }
}

fun Map<Double, Double>.quickPlotSave(seriesName: String, path: String, plotManipulation: (XYChart) -> Unit) {

    this.quickPlot(seriesName).also(plotManipulation).saveAs(path)
}

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

fun XYChart.saveAs(path: String) {

    val format = try {
        BitmapEncoder.BitmapFormat.valueOf(path.let { it.substring(it.lastIndexOf('.') + 1).toUpperCase() })
    } catch (e: Exception) {
        println("Bitmap format could not be guessed based on {$path}. Error: ${e.message}")
        BitmapEncoder.BitmapFormat.JPG
    }

    BitmapEncoder.saveBitmap(
        this,
        path,
        format
    )
}

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

        override fun hasNext(): Boolean = next < end

        override fun next(): Double = next.also { this.next = it + step }

    }
}