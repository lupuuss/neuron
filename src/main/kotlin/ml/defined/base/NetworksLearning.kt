package ml.defined.base

import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import ml.ConsoleAnimation
import ml.freeze.NetworkFreezer
import ml.learn.NetworkTeacher
import ml.output.ErrorCollector
import ml.output.NetworkProgressPrinter
import ml.quickPlotDisplay
import ml.spine.Network
import org.knowm.xchart.XYChart
import org.knowm.xchart.style.markers.None
import kotlin.streams.toList

/**
 * Implements standard learning process for multiple neural networks.
 */
@Suppress("MemberVisibilityCanBePrivate")
abstract class NetworksLearning(config: Config) : Learning(config) {

    protected var asyncRunner: Boolean = false

    protected val networks: MutableList<Network> = mutableListOf()

    protected val errorCollector: ErrorCollector = ErrorCollector()

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

    protected fun baseUnfreezing(names: List<String>): List<Network> {
        val unfreezed = mutableListOf<Network>()

        for (name in names) {
            NetworkFreezer.unfreezeFile(name)?.let { unfreezed.add(it) }
        }

        return if (unfreezed.size == names.size) {
            unfreezed
        } else {
            emptyList()
        }
    }

    protected fun baseUnfreezing(vararg values: String): List<Network> = baseUnfreezing(values.toList())

    /**
     * Should load neural networks from disk. It is called only once on start.
     */
    protected abstract fun unfreezing(): List<Network>

    /**
     * Should build all neural networks that will be used. It is called only once on start.
     */
    protected abstract fun buildNetworks(): List<Network>

    /**
     * It is called before learning process for each network. On this step network is unlearned.
     * It might be not called if a neural network was unfreezed.
     */
    protected open fun beforeLearning(network: Network, teacher: NetworkTeacher) {}

    /**
     * It's called for each network after its learning process is done.
     */
    protected open fun afterLearning(network: Network, errorVector: List<Double>?, steps: Int?) {}


    private fun syncRunner() {

        for ((index, network) in networks.withIndex()) {

            beforeLearning(network, teachers[index])

            val networkTime = System.currentTimeMillis()

            val (errorVector, steps) = learningProcess(network, teachers[index])

            afterLearning(network, errorVector, steps)

            singleNetworkLog(network, errorVector, steps, networkTime)
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

        val animation = ConsoleAnimation.frames()

        runBlocking {

            val all = awaits.size

            while (awaits.size != 0) {

                val finished = awaits.filter { it.isCompleted }
                awaits.removeAll(finished)

                finished.forEach {

                    val (network, result) = it.await()
                    val (errorVector, steps, time) = result

                    afterLearning(network, errorVector, steps)
                    print("\r")
                    singleNetworkLog(network, errorVector, steps, time)
                }

                val progress = all - awaits.size

                print("\rLearning: ${animation.next()} [$progress/$all]")
                delay(250)
            }

            print("\r")

        }
    }

    private fun singleNetworkLog(network: Network, errors: List<Double>, steps: Int, startTime: Long) {

        println(
            "Network '${network.name}' learning time: ${System.currentTimeMillis() - startTime} ms" +
                    "\n\tSteps: $steps" +
                    "\n\tError: $errors" +
                    "\n\tAvgError: ${errors.average()}" +
                    if (steps == stepsLimit) "\n\t[Warning] Network reached steps limit!" else ""
        )
    }

    protected fun plotMultiple(
        data: List<Pair<String, Map<Double, Double>>>,
        title: String,
        chartEdit: (XYChart.() -> Unit)? = null
    ) {

        val (firstName, firstErrors) = data.first()
        val remToPlot = data.stream().skip(1).toList().toMap()

        firstErrors.quickPlotDisplay(firstName) { _ ->

            this.title = title

            remToPlot.forEach { (name, data) ->

                addSeries(name, data.keys.toDoubleArray(), data.values.toDoubleArray())
                seriesMap[name]!!.marker = None()
            }

            chartEdit?.invoke(this)
        }
    }

    /**
     * Initializes whole learning process.
     */
    @Throws(UnfulfilledExpectationsException::class)
    override fun run() {

        setup()

        val restoredNetwork = unfreezing()

        teachers.addAll(buildTeachers())

        val restored = if (config.alwaysFresh || restoredNetwork.isEmpty()) {

            networks.addAll(buildNetworks())

            val time = System.currentTimeMillis()

            if (asyncRunner) {
                asyncRunner()
            } else {
                syncRunner()
            }

            println("Total elapsed time: ${System.currentTimeMillis() - time} ms")

            if (config.freeze) {
                networks.forEach {
                    NetworkFreezer.freeze(it, false)
                }
            }

            false

        } else {

            println("Networks unfreezed!")

            networks.addAll(restoredNetwork)

            for (network in networks) {
                afterLearning(network, null, null)
            }

            true
        }

        allNetworksReady(restored)
    }
}