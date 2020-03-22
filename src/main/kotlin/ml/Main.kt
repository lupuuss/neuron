package ml

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.arguments.default
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.option
import com.github.ajalt.clikt.parameters.types.file
import com.github.ajalt.clikt.parameters.types.int
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.spine.Activation
import ml.spine.Network
import java.awt.Color
import java.lang.Exception
import java.nio.file.Paths
import java.util.*

class Main : CliktCommand() {

    private val inputFile by argument(help = "Destination of training data set.").file(
        mustExist = true,
        mustBeReadable = true
    )

    private val mode: String by argument(help = "Learning mode [online/offline]. Default value = 'online'").default("online")

    private val separator: String by option("--separator", "-s", help = "Sets data separator. Default value = ';'").default(";")

    private val inputs: Int by option("--inputs", "-i", help = "Sets number of inputs in training data. Default value = 1")
        .int()
        .default(1)

    private val expected: Int by option("--expected", "-e", help = "Sets number of expected output values. Default value = 1")
        .int()
        .default(1)

    override fun run() {

        val mode = NetworkTeacher.Mode.valueOf(
            mode
                .replace("-", "")
                .replace("_", "")
                .toLowerCase()
                .capitalize()
        )
        val trainingData = DataParser(separator, inputs, expected).parse(Scanner(inputFile.inputStream()))

        val teacher = NetworkTeacher.get(mode, 0.5, 0.9)
        val network = Network.Builder()
            .name("ćwiczenie3")
            .inputs(1)
            .setDefaultActivation(Activation.Sigmoid)
            .hiddenLayer(2, true)
            .outputLayer(1, true)

        teacher.trainingSet = trainingData
        teacher.verificationSet = trainingData

        var i = 0

        val networkPrinter = NetworkProgressPrinter(System.out)
        networkPrinter.mode = NetworkProgressPrinter.Mode.Async
        networkPrinter.type = NetworkProgressPrinter.Type.InPlace
        networkPrinter.stepMetric = teacher.metric
        networkPrinter.formatter = { error, _, _ -> "$error" }

        val time = System.currentTimeMillis()

        do {

            teacher.teach(network)
            val errorVector = teacher.verify(network)
            i++

            networkPrinter.updateData(errorVector.first(), i)

        } while (errorVector.stream().allMatch { it > 0.00001 })

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
}

fun main(args: Array<String>): Unit = Main().main(args)