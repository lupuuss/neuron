package ml

import ml.freeze.NetworkFreezer
import ml.learn.OnlineNetworkTeacher
import ml.spine.Activation
import ml.spine.Network
import ml.spine.Neuron
import java.awt.Color
import java.lang.Exception
import java.nio.file.Paths
import java.util.*

fun parseInputData(inputScanner: Scanner): List<Pair<List<Double>, List<Double>>> {

    val data = mutableListOf<Pair<List<Double>, List<Double>>>()

    while (inputScanner.hasNextLine()) {

        val values = inputScanner.nextLine().split(";")

        val inDouble = values[0].toDouble()
        val outDouble = values[1].toDouble()

        data.add(listOf(inDouble) to listOf(outDouble))
    }

    inputScanner.close()

    return data
}

fun getNeuronActivationPoints(neuron: Neuron, range: Iterator<Double>): Map<Double, Double> {

    val values = mutableMapOf<Double, Double>()

    for (i in range) {
        values[i] = neuron.activate(listOf(i)).activation
    }

    return values
}

fun main(args: Array<String>) {

    // Preparation

    val (inputScanner, _) = try {

        Scanner(Paths.get(args[0]).toFile().inputStream()) to
                Paths.get(args[1]).toFile().outputStream().bufferedWriter()

    } catch (e: Exception) {

        throw e
    }

    val trainingData = parseInputData(inputScanner)

    val teacher = OnlineNetworkTeacher(0.1)
    val network = Network.Builder()
        .name("cw3")
        .inputs(1)
        .setDefaultActivation(Activation.Sigmoid)
        .hiddenLayer(3, true)
        .outputLayer(1, true)

    teacher.trainingSet = trainingData
    teacher.verificationSet = trainingData

    var i = 0
    do {

        teacher.teach(network)
        val errorVector = teacher.verify(network)
        i++
    } while (errorVector.stream().allMatch { it > 0.001 })

    println("Iterations: $i")

    val plotData = mutableMapOf<Double, Double>()

    for (x in -1.0..3.0 step 0.1) {
        plotData[x] = network.answer(listOf(x)).first()
    }

    plotData.quickPlotDisplay("Result") { _ ->
        title = "ML"
        addSeries("training points", trainingData.map { it.first.first() }, trainingData.map { it.second.first() })
        seriesMap["training points"]!!.lineColor = Color(0, 0, 0, 0)
    }

    NetworkFreezer.freeze(network)

}