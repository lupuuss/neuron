@file:Suppress("unused")

package ml

import ml.spine.Network
import org.knowm.xchart.*
import org.knowm.xchart.style.markers.None
import java.lang.Exception
import kotlin.streams.toList

object QuickPlotSettings {
    var width: Int = 800
    var height: Int = 600
    var axisTitleX: String = ""
    var axsisTitleY: String = ""
}

fun XYChart.saveAs(path: String) {

    val format = try {
        BitmapEncoder.BitmapFormat.valueOf(path.let { it.substring(it.lastIndexOf('.') + 1).toUpperCase() })
    } catch (e: Exception) {
        println("Failed to infer bitmap format {$path}. Error: ${e.message}")
        BitmapEncoder.BitmapFormat.JPG
    }

    BitmapEncoder.saveBitmap(
        this,
        path,
        format
    )
}

fun Map<Double, Double>.quickPlot(seriesName: String): XYChart = XYChartBuilder()
    .width(QuickPlotSettings.width)
    .height(QuickPlotSettings.height)
    .xAxisTitle(QuickPlotSettings.axisTitleX)
    .yAxisTitle(QuickPlotSettings.axsisTitleY)
    .build()
    .apply {

        if (isNotEmpty()) {

            this.addSeries(seriesName, keys.toDoubleArray(), values.toDoubleArray())
            this.seriesMap[seriesName]?.marker = None()
        }
    }

fun Map<Double, Double>.quickPlotSwingWrapped(seriesName: String): SwingWrapper<XYChart> = SwingWrapper(
    this.quickPlot(seriesName)
)

fun Map<Double, Double>.quickPlotDisplay(
    seriesName: String,
    plotManipulation: XYChart.(SwingWrapper<XYChart>) -> Unit
) {

    val chart = this.quickPlot(seriesName)
    val wrapper = SwingWrapper(chart)
    plotManipulation(chart, wrapper)
    wrapper.displayChart()
}

fun Map<Double, Double>.quickPlotDisplay(seriesName: String, plotManipulation: XYChart.() -> Unit) {

    this.quickPlotDisplay(seriesName) { _ -> plotManipulation(this) }
}

fun XYChart.addSeries(title: String, map: Map<Double, Double>) {
    this.addSeries(title, map.keys.toDoubleArray(), map.values.toDoubleArray())
}

fun Map<Double, Double>.quickPlotSave(seriesName: String, path: String, plotManipulation: XYChart.() -> Unit) {

    this.quickPlot(seriesName).also(plotManipulation).saveAs(path)
}

fun plotMultiple(
    data: Map<String, Map<Double, Double>>,
    title: String,
    chartEdit: (XYChart.() -> Unit)? = null
) {

    val (firstName, firstErrors) = data.entries.first()
    val remToPlot = data.toList().stream().skip(1).toList().toMap()

    firstErrors.quickPlotDisplay(firstName) { _ ->

        this.title = title

        remToPlot.forEach { (name, data) ->

            addSeries(name, data.keys.toDoubleArray(), data.values.toDoubleArray())
            seriesMap[name]!!.marker = None()
        }

        chartEdit?.invoke(this)
    }
}

fun networkPlotDataXY(network: Network, range: Iterator<Double>): Map<Double, Double> {

    val result = mutableMapOf<Double, Double>()

    for (x in range) {
        result[x] = network.answer(listOf(x)).first()
    }

    return result
}
