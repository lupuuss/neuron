package ml

import ml.learn.SingleNeuronTeacher
import ml.spine.Activation
import ml.spine.Neuron
import org.knowm.xchart.style.markers.None
import java.awt.Color
import java.lang.Exception
import java.nio.file.Paths
import java.util.*

fun parseInputData(inputScanner: Scanner): Pair<Array<DoubleArray>, DoubleArray> {

    val inputData = mutableListOf<Double>()
    val expectedData = mutableListOf<Double>()

    while (inputScanner.hasNextLine()) {

        val values = inputScanner.nextLine().split(";")

        val inDouble = values[0].toDouble()
        val outDouble = values[1].toDouble()

        inputData.add(inDouble)
        expectedData.add(outDouble)
    }

    inputScanner.close()

    return inputData.map { doubleArrayOf(it) }.toTypedArray() to expectedData.toDoubleArray()
}

fun getNeuronActivationPoints(neuron: Neuron, range: Iterator<Double>): Map<Double, Double> {

    val values = mutableMapOf<Double, Double>()

    for (i in range) {
        values[i] = neuron.activate(doubleArrayOf(i)).activation
    }

    return values
}

fun main(args: Array<String>) {

    // Preparation

    val (inputScanner, outputWriter) = try {

        Scanner(Paths.get(args[0]).toFile().inputStream()) to Paths.get(args[1]).toFile().outputStream().bufferedWriter()

    } catch (e: Exception) {

        throw e
    }


    val teacher = SingleNeuronTeacher(0.1, Activation.sigmoid())
    val neuron = Neuron(1, Activation.sigmoid().function)
    val errorChangeData = mutableMapOf<Int, Double>()

    val (input, expected) = parseInputData(inputScanner)
    val errorGoal = 0.001
    var i = 0

    val neuronFunPlotsDataMap = mutableMapOf<Int, Map<Double, Double>?>(
        0 to null,
        50 to null,
        100 to null,
        250 to null,
        1_000 to null,
        5_000 to null,
        25_000 to null,
        50_000 to null,
        100_000 to null
    )

    // Learning process

    do {

        if (neuronFunPlotsDataMap.containsKey(i)) {
            neuronFunPlotsDataMap[i] = getNeuronActivationPoints(neuron, -2.0..2.0 step 0.01)
        }

        teacher.teach(neuron, input, expected)
        errorChangeData[i] = teacher.verify(neuron, input, expected)
        println("${neuron.weights[0]} ${neuron.bias} ${errorChangeData[i]}")

    } while (errorChangeData[i++] ?: 0.0 > errorGoal)

    outputWriter.write("${neuron.weights[0]} ${neuron.bias}")
    outputWriter.close()

    // Plots

    neuronFunPlotsDataMap[i] = getNeuronActivationPoints(neuron, -2.0..2.0 step 0.01)

    neuronFunPlotsDataMap.firstValue()!!.quickPlotSave("Before learning", "progress.png") { chart ->

        chart.title = "Learning progress"
        chart.yAxisTitle = "f(x)"
        chart.xAxisTitle = "x"

        neuronFunPlotsDataMap
            .entries
            .stream()
            .skip(1)
            .forEach { chartData ->
                chart.addSeries(
                    "${chartData.key} iterations",
                    chartData.value!!.keys.toDoubleArray(),
                    chartData.value!!.values.toDoubleArray()
                )
            }

        chart.seriesMap.entries.forEach { it.value.marker = None() }

    }

    neuronFunPlotsDataMap.firstValue()!!.quickPlotSave("Before learning", "result.png") { chart ->

        val afterLearningData = neuronFunPlotsDataMap.entries.last().value!!
        chart.addSeries(
            "After learning",
            afterLearningData.keys.toDoubleArray(),
            afterLearningData.values.toDoubleArray()
        )

        chart.seriesMap["Before learning"]?.apply {
            this.lineColor = Color.RED
        }

        chart.seriesMap["After learning"]?.apply {
            this.marker = None()
            this.lineColor = Color.GREEN
        }

        chart.addSeries("Training data", input.map { it.first() }.toDoubleArray(), expected)

        chart.seriesMap["Training data"]?.apply {
            this.lineColor = Color(0, 0, 0, 0)
            this.markerColor = Color.BLACK
        }
    }

    errorChangeData
        .map { it.key.toDouble() to it.value }
        .toMap()
        .quickPlotSave("Error change", "error.png") { chart ->
            chart.title = "Quaility"
            chart.xAxisTitle = "Iterations"
            chart.yAxisTitle = "Error value"
            chart.styler.yAxisDecimalPattern = "0.00"
            chart.styler.xAxisDecimalPattern = "###,###,###,##0"
        }
}