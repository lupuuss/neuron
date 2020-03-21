package ml

import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.learn.OfflineNetworkTeacher
import ml.learn.OnlineNetworkTeacher
import ml.output.NetworkProgressPrinter
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

    val teacher = NetworkTeacher.get(NetworkTeacher.Mode.Offline, 0.5, 0.9)
    val network = Network.Builder()
        .name("cw3")
        .inputs(1)
        .setDefaultActivation(Activation.Sigmoid)
        .hiddenLayer(4, true)
        .hiddenLayer(3, true)
        .hiddenLayer(2, true)
        .outputLayer(1, true)

    teacher.trainingSet = trainingData
    teacher.verificationSet = trainingData

    var i = 0

    val networkPrinter = NetworkProgressPrinter(System.out)
    networkPrinter.stepMetric = "Epochs"

    val time = System.currentTimeMillis()

    do {

        teacher.teach(network)
        val errorVector = teacher.verify(network)
        i++

        networkPrinter.updateData(errorVector.first(), i)

    } while (errorVector.stream().allMatch { it > 0.0001 })

    networkPrinter.close()

    println()
    println("Elapsed time: ${System.currentTimeMillis() - time} ms")

    val plotData = mutableMapOf<Double, Double>()

    for (x in -1.0..3.0 step 0.01) {
        plotData[x] = network.answer(listOf(x)).first()
    }

    plotData.quickPlotDisplay("Result") { _ ->
        title = "ML"
        addSeries("training points", trainingData.map { it.first.first() }, trainingData.map { it.second.first() })
        seriesMap["training points"]!!.lineColor = Color(0, 0, 0, 0)
    }

    NetworkFreezer.freeze(network)

}