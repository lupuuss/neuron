package ml.defined

import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.spine.Network

abstract class DefinedLearning(
    protected val config: Config
) {

    lateinit var network: Network
    protected val teacher: NetworkTeacher = NetworkTeacher.get(config.teacherMode, config.alpha, config.beta)

    protected abstract fun dataSetup()

    protected abstract fun buildNetwork(): Network

    protected abstract fun unfreezing(): Network?

    protected open fun beforeLearning() {}

    private fun learningProcess(): Pair<List<Double>, Int> {

        var i = 0

        var errorVector: List<Double>
        val progressPrinter = NetworkProgressPrinter(System.out)

        progressPrinter.stepMetric = teacher.metric
        progressPrinter.formatter = config.printFormatter
        progressPrinter.mode = config.printMode
        progressPrinter.type = config.printType

        do {

            teacher.teach(network)
            errorVector = teacher.verify(network)
            i++
            eachLearningStep(progressPrinter, errorVector, i)

        } while (errorVector.stream().allMatch { it > config.errorGoal })

        progressPrinter.close()

        return errorVector to i
    }

    protected open fun eachLearningStep(
        progressPrinter: NetworkProgressPrinter,
        errorVector: List<Double>,
        steps: Int
    ) {
    }

    protected open fun afterLearning(errorVector: List<Double>?, steps: Int?, restored: Boolean) {}

    fun run() {

        dataSetup()

        val restoredNetwork = unfreezing()

        if (config.alwaysFresh || restoredNetwork == null) {

            network = buildNetwork()

            beforeLearning()
            val (errorVector, steps) = learningProcess()
            afterLearning(errorVector, steps, false)

        } else {

            network = restoredNetwork
            afterLearning(null, null, true)
        }
    }
}