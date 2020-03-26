package ml.defined

import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.output.ErrorCollector
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
        Exercise3, Transformation
    }

    protected var asyncRunner: Boolean = false

    protected val networks: MutableList<Network> = mutableListOf()

    protected val teachers: MutableList<NetworkTeacher> = mutableListOf()

    protected val errorCollector = ErrorCollector()
    protected abstract val errorGoal: Double
    protected abstract val stepsLimit: Int

    protected fun requiresProgressPrinter(): NetworkProgressPrinter {

        if (asyncRunner) {
            println("[WARNING] Its not required to print progress while asynchronous learning!")
        }

        return NetworkProgressPrinter(System.out).apply {
            formatter = config.printFormatter
            mode = config.printMode
            type = config.printType
        }
    }

    /**
     * Should load all necessary data for learning. It is called only once on start.
     */
    protected abstract fun dataLoading()

    /**
     * Should build all neural networks that will be used. It is called only once on start.
     */
    protected abstract fun buildNetworks(): List<Network>

    /**
     * Should build all teachers that will be used. It is called only once on start.
     * Amount of teachers must be the same as networks.
     */
    protected abstract fun buildTeachers(): List<NetworkTeacher>

    /**
     * Should load neural networks from disk. It is called only once on start.
     */
    protected abstract fun unfreezing(): List<Network>

    /**
     * It is called before learning process for each network. On this step network is unlearned.
     * It might be not called if a neural network was unfreezed.
     */
    protected open fun beforeLearning(network: Network, teacher: NetworkTeacher) {}

    /**
     * Performs standard learning process for each network.
     * It might be not called if a neural network was unfreezed.
     */
    private fun learningProcess(network: Network, teacher: NetworkTeacher): Pair<List<Double>, Int> {

        var i = 0

        var errorVector: List<Double>

        do {

            teacher.teach(network)
            errorVector = teacher.verify(network)
            i++
            eachLearningStep(network, errorVector, i)

        } while (!errorVector.stream().allMatch { it < errorGoal } && i < stepsLimit)
        return errorVector to i
    }

    /**
     * It's called after every step of learning.
     */
    protected open fun eachLearningStep(network: Network, errorVector: List<Double>, steps: Int) {}

    /**
     * It's called for each network after its learning process is done.
     */
    protected open fun afterLearning(network: Network, errorVector: List<Double>?, steps: Int?) {}

    /**
     * It's called when every network is learned.
     */
    protected open fun allNetworksReady(restored: Boolean) {}

    private fun singleNetworkLog(network: Network, errors: List<Double>, steps: Int, startTime: Long) {

        println(
            "Network '${network.name}' learning time: ${System.currentTimeMillis() - startTime} ms" +
                    "\n\tSteps: $steps" +
                    "\n\tError: $errors" +
                    "\n\tAvgError: ${errors.average()}" +
                    if (steps == stepsLimit) "\n\t[Warning] Network reached steps limit!" else ""
        )
    }

    private fun syncRunner() {

        for ((index, network) in networks.withIndex()) {

            beforeLearning(network, teachers[index])

            val networkTime = System.currentTimeMillis()

            val (errorVector, steps) = learningProcess(network, teachers[index])

            singleNetworkLog(network, errorVector, steps, networkTime)

            afterLearning(network, errorVector, steps)
        }
    }

    private fun asyncRunner() {

        val awaits = networks.mapIndexed { index, network ->

            val teacher = teachers[index]

            beforeLearning(network, teacher)

            GlobalScope.async {

                val time = System.currentTimeMillis()
                val (errorVector, steps) = learningProcess(network, teacher)

                network to Triple(errorVector, steps, time)
            }
        }.toMutableList()

        runBlocking {

            while (awaits.size != 0) {

                val finished = awaits.filter { it.isCompleted }
                awaits.removeAll(finished)

                finished.forEach {

                    val (network, result) = it.await()
                    val (errorVector, steps, time) = result

                    singleNetworkLog(network, errorVector, steps, time)

                    afterLearning(network, errorVector, steps)
                }

                delay(100)
            }

        }
    }

    /**
     * Initializes whole learning process.
     */
    fun run() {

        dataLoading()

        val restoredNetwork = unfreezing()


        val restored = if (config.alwaysFresh || restoredNetwork.isEmpty()) {

            networks.addAll(buildNetworks())
            teachers.addAll(buildTeachers())

            val time = System.currentTimeMillis()

            if (asyncRunner) {
                asyncRunner()
            } else {
                syncRunner()
            }

            println("Total elapsed time: ${System.currentTimeMillis() - time}")

            if (config.freeze) {
                networks.forEach {
                    NetworkFreezer.freeze(it, false)
                }
            }

            false

        } else {

            networks.addAll(restoredNetwork)

            for (network in networks) {
                afterLearning(network, null, null)
            }

            true
        }

        allNetworksReady(restored)
    }

    companion object {

        @JvmStatic
        fun get(type: Type, config: Config): DefinedLearning = when (type) {
            Type.Exercise3 -> Exercise3(config)
            Type.Transformation -> Transformation(config)
        }
    }
}