package ml.defined

import ml.DataParser
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.quickPlotDisplay
import ml.saveAs
import ml.spine.Activation
import ml.spine.Network
import ml.step
import org.knowm.xchart.style.markers.None
import java.awt.Color
import java.util.*

class Exercise3(
    config: Config
) : DefinedLearning(config) {

    private var beforeLearningData: MutableMap<Double, Double> = mutableMapOf()
    private var afterLearningData: MutableMap<Double, Double> = mutableMapOf()
    private val errorChange: MutableMap<Int, Double> = mutableMapOf()

    override var errorGoal: Double = 0.001

    private val alpha = if (config.teacherMode == NetworkTeacher.Mode.Online) 0.3 else 0.9
    private val beta = if (config.teacherMode == NetworkTeacher.Mode.Online) 0.2 else 0.9

    override val teacher: NetworkTeacher = NetworkTeacher.get(config.teacherMode, alpha, beta)

    override fun dataLoading() {
        val parser = DataParser(config.separator, 1, 1)
        val data = parser.parse(Scanner(config.input))

        teacher.trainingSet = data
        teacher.verificationSet = data
    }

    override fun buildNetworks(): MutableList<Network> = mutableListOf(
        Network.Builder()
            .name("exercise3")
            .setDefaultActivation(Activation.Sigmoid)
            .inputs(1)
            .hiddenLayer(2, true)
            .outputLayer(1, true)
    )

    override fun unfreezing(): MutableList<Network> {
        return NetworkFreezer.unfreezeFile("exercise3")?.let { mutableListOf(it) } ?: mutableListOf()
    }

    private fun getDataPlotXY(network: Network, range: Iterator<Double>): MutableMap<Double, Double> {

        val plotData = mutableMapOf<Double, Double>()

        for (x in range) {
            plotData[x] = network.answer(listOf(x)).first()
        }

        return plotData
    }

    override fun beforeLearning(network: Network) {
        beforeLearningData = getDataPlotXY(network, -1.0..3.0 step 0.1)
    }

    override fun eachLearningStep(
        network: Network,
        progressPrinter: NetworkProgressPrinter,
        errorVector: List<Double>,
        steps: Int
    ) {
        progressPrinter.updateData(errorVector, steps)
        errorChange[steps] = errorVector.first()
    }

    override fun afterLearning(network: Network, errorVector: List<Double>?, steps: Int?, restored: Boolean) {
        afterLearningData = getDataPlotXY(network, -1.0..3.0 step 0.1)
    }

    override fun allNetworksReady() {

        val type = config.teacherMode.toString().toLowerCase()

        errorChange.map { it.key.toDouble() to it.value }.toMap().quickPlotDisplay("Error change") { _ ->
            title = "Quality for $type. Momentum = $beta Alpha = $alpha."
            xAxisTitle = "Iterations"
            yAxisTitle = "Error value"
            styler.yAxisDecimalPattern = "0.000"
            styler.xAxisDecimalPattern = "###,###,###,###"

            saveAs("error_$type.png")
        }

        beforeLearningData.quickPlotDisplay("Before learning") { _ ->
            title = "Result for $type. Momentum = $beta Alpha = $alpha"
            xAxisTitle = "x"
            yAxisTitle = "Network answer"

            seriesMap["Before learning"]!!.lineColor = Color.RED

            addSeries(
                "After learning",
                afterLearningData.keys.toDoubleArray(),
                afterLearningData.values.toDoubleArray()
            )
            seriesMap["After learning"]!!.apply {
                lineColor = Color.GREEN
                marker = None()
            }

            addSeries(
                "Training points",
                teacher.trainingSet.map { it.first.first() },
                teacher.trainingSet.map { it.second.first() }
            )
            seriesMap["Training points"]!!.apply {
                lineColor = Color(0, 0, 0, 0)
                markerColor = Color.BLACK
            }
            saveAs("result_$type.png")
        }
    }
}