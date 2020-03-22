package ml.defined

import ml.DataParser
import ml.freeze.NetworkFreezer
import ml.output.NetworkProgressPrinter
import ml.quickPlotDisplay
import ml.spine.Activation
import ml.spine.Network
import ml.step
import java.util.*

class Exercise3(
    config: Config
) : DefinedLearning(config) {

    private var beforeLearningData: MutableMap<Double, Double> = mutableMapOf()
    private var afterLearningData: MutableMap<Double, Double> = mutableMapOf()
    private val errorChange: MutableMap<Int, Double> = mutableMapOf()

    override fun dataSetup() {
        val parser = DataParser(config.separator, 1, 1)
        val data = parser.parse(Scanner(config.input))

        teacher.trainingSet = data
        teacher.verificationSet = data
    }

    override fun buildNetwork(): Network = Network.Builder()
        .name("exercise3")
        .setDefaultActivation(Activation.Sigmoid)
        .inputs(1)
        .hiddenLayer(2, true)
        .outputLayer(1, true)

    override fun unfreezing(): Network? = NetworkFreezer.unfreezeFile("exercise3")

    private fun getDataPlotXY(network: Network, range: Iterator<Double>): MutableMap<Double, Double> {

        val plotData = mutableMapOf<Double, Double>()

        for (x in range) {
            plotData[x] = network.answer(listOf(x)).first()
        }

        return plotData
    }

    override fun beforeLearning() {
        beforeLearningData = getDataPlotXY(network, -1.0..3.0 step 0.1)
    }

    override fun eachLearningStep(progressPrinter: NetworkProgressPrinter, errorVector: List<Double>, steps: Int) {
        progressPrinter.updateData(errorVector, steps)
        errorChange[steps] = errorVector.first()
    }

    override fun afterLearning(errorVector: List<Double>?, steps: Int?, restored: Boolean) {
        afterLearningData = getDataPlotXY(network, -1.0..3.0 step 0.1)

        errorChange.map { it.key.toDouble() to it.value }.toMap().quickPlotDisplay("Error change") { _ ->
            title = "Quality"
            xAxisTitle = "Iterations"
            yAxisTitle = "Error value"
            styler.yAxisDecimalPattern = "0.000"
            styler.xAxisDecimalPattern = "###,###,###,###"
        }
    }
}