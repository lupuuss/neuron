package ml.defined

import ml.addSeries
import ml.defined.base.ApproximationBase
import ml.defined.base.Config
import ml.defined.base.NetworksLearning
import ml.defined.base.UnfulfilledExpectationsException
import ml.learn.NetworkTeacher
import ml.networkPlotDataXY
import ml.plotMultiple
import ml.spine.Activation
import ml.spine.Network
import ml.step
import org.knowm.xchart.style.markers.Circle
import java.awt.Color

class ApproximationProgress(config: Config) : ApproximationBase(config) {

    override val errorGoal: Double = 0.01
    override val stepsLimit: Int = 70_000

    // Separate teacher for each learning data file.
    private val localTeachers = listOf(
        NetworkTeacher.get(teacherMode, alpha, beta),
        NetworkTeacher.get(teacherMode, alpha, beta)
    )

    private val plotData: List<MutableMap<Int, Map<Double, Double>>> = listOf(mutableMapOf(), mutableMapOf())
    private val plotSteps = listOf(100, 300, 10_000, 30_000)

    private fun getRange() = -3.0..4.0 step 0.1

    override fun buildTeachers(): List<NetworkTeacher> {

        for (i in localTeachers.indices) {

            localTeachers[i].apply {
                verificationSet = verification
                trainingSet = training[i]
            }
        }

        return localTeachers
    }

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

    override fun afterLearning(network: Network, teacher: NetworkTeacher, errorVector: List<Double>, steps: Int) {

        pickPlotData(network, steps)
    }

    override fun allNetworksReady() {

        val verificationData = verification.map { it.first.first() to it.second.first() }.toMap()

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
            styler.yAxisMax = 80.0

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