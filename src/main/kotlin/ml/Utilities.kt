@file:Suppress("unused")

package ml

import org.knowm.xchart.*
import org.knowm.xchart.style.markers.None
import java.lang.Exception
import java.lang.IllegalArgumentException

object QuickPlotSettings {
    var width: Int = 800
    var height: Int = 600
    var axisTitleX = ""
    var axsisTitleY = ""
}

fun Map<Double, Double>.quickPlot(seriesName: String): XYChart = XYChartBuilder()
    .width(QuickPlotSettings.width)
    .height(QuickPlotSettings.height)
    .xAxisTitle(QuickPlotSettings.axisTitleX)
    .yAxisTitle(QuickPlotSettings.axsisTitleY)
    .build()
    .apply {
        this.addSeries(seriesName, keys.toDoubleArray(), values.toDoubleArray())
        this.seriesMap[seriesName]?.marker = None()
    }

fun Map<Double, Double>.quickPlotSwingWrapped(seriesName: String): SwingWrapper<XYChart> = SwingWrapper(
    this.quickPlot(seriesName)
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