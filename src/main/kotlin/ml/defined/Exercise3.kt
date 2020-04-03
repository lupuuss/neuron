package ml.defined

import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.input.DataParser
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
) : NetworksLearning(config) {

    private var beforeLearningData: MutableMap<Double, Double> = mutableMapOf()
    private var afterLearningData: MutableMap<Double, Double> = mutableMapOf()

    override val errorGoal: Double = 0.001
    override val stepsLimit: Int = 5_000_000

    private val alpha = 0.3
    private val beta = 0.3
    private val data: MutableList<Pair<List<Double>, List<Double>>> = mutableListOf()
    private var progressPrinter: NetworkProgressPrinter? = null

    override fun setup() {
        val parser = DataParser(config.separator, 1, 1)
        data.addAll(parser.parse(Scanner(config.inputs.first())))
    }

    override fun buildNetworks(): List<Network> = listOf(
        Network.Builder()
            .name("exercise3")
            .setDefaultActivation(Activation.Sigmoid)
            .inputs(1)
            .hiddenLayer(2, true)
            .outputLayer(1, true)
    )

    override fun buildTeachers(): List<NetworkTeacher> = listOf(
        NetworkTeacher.get(config.teacherMode, alpha, beta).apply {
            trainingSet = data
            verificationSet = data
        }
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

    override fun beforeLearning(network: Network, teacher: NetworkTeacher) {
        beforeLearningData = getDataPlotXY(network, -1.0..3.0 step 0.1)
        progressPrinter = requiresProgressPrinter().apply {
            stepMetric = teacher.metric
        }
    }

    override fun eachLearningStep(
        network: Network,
        errorVector: List<Double>,
        steps: Int
    ) {
        progressPrinter?.updateData(errorVector, steps)
        errorCollector.collect(network, errorVector, steps)
    }

    override fun afterLearning(network: Network, errorVector: List<Double>?, steps: Int?) {

        progressPrinter?.close()
        progressPrinter = null

        afterLearningData = getDataPlotXY(network, -1.0..3.0 step 0.1)
    }

    override fun allNetworksReady(restored: Boolean) {

        val type = config.teacherMode.toString().toLowerCase()
        val errorChange = errorCollector.getAveragePlotableErrorMap().first().second

        if (!restored) {
            errorChange.quickPlotDisplay("Error change") { _ ->
                title = "Quality for $type. Momentum = $beta Alpha = $alpha."
                xAxisTitle = "Iterations"
                yAxisTitle = "Error value"
                styler.yAxisDecimalPattern = "0.000"
                styler.xAxisDecimalPattern = "###,###,###,###"

                saveAs("error_$type.png")
            }
        }

        beforeLearningData.quickPlotDisplay("Before learning") { _ ->
            title = "Result for $type. Momentum = $beta Alpha = $alpha"
            xAxisTitle = "x"
            yAxisTitle = "Network answer"

            seriesMap["Before learning"]?.lineColor = Color.RED

            addSeries(
                "After learning",
                afterLearningData.keys.toDoubleArray(),
                afterLearningData.values.toDoubleArray()
            )
            seriesMap["After learning"]?.apply {
                lineColor = Color.GREEN
                marker = None()
            }

            addSeries(
                "Training points",
                teachers.first().trainingSet.map { it.first.first() },
                teachers.first().trainingSet.map { it.second.first() }
            )
            seriesMap["Training points"]?.apply {
                lineColor = Color(0, 0, 0, 0)
                markerColor = Color.BLACK
            }
            saveAs("result_$type.png")
        }
    }

}