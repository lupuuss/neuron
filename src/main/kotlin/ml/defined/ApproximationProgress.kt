package ml.defined

import ml.addSeries
import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.defined.base.UnfulfilledExpectationsException
import ml.learn.NetworkTeacher
import ml.spine.Activation
import ml.spine.Network
import ml.step
import org.knowm.xchart.style.markers.Circle
import java.awt.Color

class ApproximationProgress(config: Config) : NetworksLearning(config) {

    override val errorGoal: Double = 0.01
    override val stepsLimit: Int = 70_000

    private val localTeachers = listOf(
        NetworkTeacher.get(config.teacherMode, 0.01, 0.8),
        NetworkTeacher.get(config.teacherMode, 0.01, 0.8)
    )

    private val plotData: List<MutableMap<Int, Map<Double, Double>>> = listOf(mutableMapOf(), mutableMapOf())
    private val plotSteps = listOf(100, 300, 10_000, 30_000)

    private fun getRange() = -4.0..3.0 step 0.1

    override fun setup() {

        if (config.inputs.size != 3) {
            throw UnfulfilledExpectationsException("Approximation task requires training data (2 files) and verification data (1 file) ")
        }

        val verification = dataParser.parse(config.inputs[2], 1, 1)

        for (index in localTeachers.indices) {
            localTeachers[index].verificationSet = verification
            localTeachers[index].trainingSet = dataParser.parse(config.inputs[index], 1, 1)
        }

        asyncRunner = true
    }

    override fun buildTeachers(): List<NetworkTeacher> = localTeachers

    override fun buildNetworks(): List<Network> = listOf(
        Network.Builder()
            .inputs(1)
            .name("Approxmiation_set_1")
            .hiddenLayer(15, true, Activation.Sigmoid)
            .outputLayer(1, true, Activation.Identity),
        Network.Builder()
            .inputs(1)
            .name("Approximation_set_2")
            .hiddenLayer(15, true, Activation.Sigmoid)
            .outputLayer(1, true, Activation.Identity)
    )

    override fun unfreezing(): List<Network> = baseUnfreezing("Approximation_set_1", "Approximation_set_2")

    private fun pickPlotData(network: Network, steps: Int) {

        if (network.name.contains("set_1")) {
            plotData[0][steps] = networkPlotDataXY(network, getRange())
        } else {
            plotData[1][steps] = networkPlotDataXY(network, getRange())
        }
    }

    override fun beforeLearning(network: Network, teacher: NetworkTeacher) {
        pickPlotData(network, 0)
    }

    override fun eachLearningStep(network: Network, teacher: NetworkTeacher, errorVector: List<Double>, steps: Int) {

        if (plotSteps.contains(steps)) {
            pickPlotData(network, steps)
        }
    }

    override fun afterLearning(network: Network, teacher: NetworkTeacher?, errorVector: List<Double>?, steps: Int?) {

        pickPlotData(network, steps ?: 0)
    }

    override fun allNetworksReady(restored: Boolean) {

        val verificationData = localTeachers[0].verificationSet.map { it.first.first() to it.second.first() }.toMap()

        for (i in localTeachers.indices) {

            plotResults(
                plotData[i],
                localTeachers[i].trainingSet.map { it.first.first() to it.second.first() }.toMap(),
                verificationData,
                i + 1
            )
        }
    }

    private fun plotResults(
        results: MutableMap<Int, Map<Double, Double>>,
        trainingData: Map<Double, Double>,
        verificationData: Map<Double, Double>,
        fileN: Int
    ) {


        plotMultiple(results.map { (steps, plot) -> "$steps iterations" to plot }.toMap(), "File $fileN results") {
            styler.yAxisMax = 60.0

            addSeries("training data", trainingData)
            addSeries("verification data", verificationData)
            seriesMap["training data"]?.apply {
                lineColor = Color(0, 0, 0, 0)
                marker = Circle()
                markerColor = Color.RED
            }

            seriesMap["verification data"]?.apply {
                lineColor = Color(0, 0, 0, 0)
                markerColor = Color.GREEN
                marker = Circle()
            }
        }

    }
}