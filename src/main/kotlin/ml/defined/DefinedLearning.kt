package ml.defined

import ml.learn.NetworkTeacher
import ml.output.NetworkProgressPrinter
import ml.spine.Network

/**
 * Implements standard learning process for multiple neural networks.
 */
@Suppress("MemberVisibilityCanBePrivate")
abstract class DefinedLearning(
    protected val config: Config
) {

    enum class Type {
        Exercise3
    }

    protected val networks: MutableList<Network> = mutableListOf()
    protected abstract val teacher: NetworkTeacher
    protected abstract var errorGoal: Double

    /**
     * Should load all necessary data for learning. It is called only once on start.
     */
    protected abstract fun dataLoading()

    /**
     * Should build all neural networks that will be used. It is called only once on start.
     */
    protected abstract fun buildNetworks(): MutableList<Network>

    /**
     * Should load neural networks from disk. It is called only once on start.
     */
    protected abstract fun unfreezing(): MutableList<Network>

    /**
     * It is called before learning process for each network. On this step network is unlearned.
     * It might be not called if a neural network was unfreezed.
     */
    protected open fun beforeLearning(network: Network) {}

    /**
     * Performs standard learning process for each network.
     * It might be not called if a neural network was unfreezed.
     */
    private fun learningProcess(network: Network): Pair<List<Double>, Int> {

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
            eachLearningStep(network, progressPrinter, errorVector, i)

        } while (errorVector.stream().allMatch { it > errorGoal })

        progressPrinter.close()

        return errorVector to i
    }

    /**
     * It's called after every step of learning.
     */
    protected open fun eachLearningStep(
        network: Network,
        progressPrinter: NetworkProgressPrinter,
        errorVector: List<Double>,
        steps: Int
    ) {
    }

    /**
     * It's called for each network after its learning process is done.
     */
    protected open fun afterLearning(network: Network, errorVector: List<Double>?, steps: Int?, restored: Boolean) {}

    /**
     * It's called when every network is learned.
     */
    protected open fun allNetworksReady() {}

    /**
     * Initializes whole learning process.
     */
    fun run() {

        dataLoading()

        val restoredNetwork = unfreezing()

        if (config.alwaysFresh || restoredNetwork.isEmpty()) {

            networks.addAll(buildNetworks())

            val time = System.currentTimeMillis()
            for (network in networks) {

                beforeLearning(network)

                val networkTime = System.currentTimeMillis()

                val (errorVector, steps) = learningProcess(network)

                println("Network '${network.name}' learning time: ${System.currentTimeMillis() - networkTime}")

                afterLearning(network, errorVector, steps, false)
            }

            println("Total elapsed time: ${System.currentTimeMillis() - time}")

        } else {

            networks.addAll(restoredNetwork)

            for (network in networks) {
                afterLearning(network, null, null, true)
            }
        }


        allNetworksReady()
    }

    companion object {

        @JvmStatic
        fun get(type: Type, config: Config): DefinedLearning = when (type) {
            Type.Exercise3 -> Exercise3(config)
        }
    }
}