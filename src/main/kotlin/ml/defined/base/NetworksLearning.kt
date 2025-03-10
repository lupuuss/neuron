package ml.defined.base

import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import ml.learn.NetworkTeacher
import ml.output.ConsoleLoader
import ml.output.ErrorCollector
import ml.spine.Network


/**
 * Implements standard learning process for multiple neural networks.
 */
@Suppress("MemberVisibilityCanBePrivate")
abstract class NetworksLearning(config: Config) : Learning(config) {

    protected val networks: MutableList<Network> = mutableListOf()

    protected val errorCollector: ErrorCollector = ErrorCollector()

    /**
     * Should build all neural networks that will be used. It is called only once on start.
     */
    protected abstract fun buildNetworks(): List<Network>

    /**
     * It is called before learning process for each network. On this step network is unlearned.
     */
    protected open fun beforeLearning(network: Network, teacher: NetworkTeacher) {}

    override fun learningProcess(network: Network, teacher: NetworkTeacher): Pair<List<Double>, Int> {

        var i = 0

        var errorVector: List<Double>

        do {

            teacher.teach(network)
            errorVector = teacher.verify(network)
            i++
            eachLearningStep(network, teacher, errorVector, i)

        } while (!errorVector.stream().allMatch { it < errorGoal } && i < stepsLimit)
        return errorVector to i
    }

    /**
     * It's called after every step of learning.
     */
    protected open fun eachLearningStep(
        network: Network,
        teacher: NetworkTeacher,
        errorVector: List<Double>,
        steps: Int
    ) {
    }

    /**
     * It's called for each network after its learning process is done.
     */
    protected open fun afterLearning(
        network: Network,
        teacher: NetworkTeacher,
        errorVector: List<Double>,
        steps: Int
    ) {
    }

    private fun asyncRunner() {

        val awaits = networks.mapIndexed { index, network ->

            val teacher = teachers[index]

            beforeLearning(network, teacher)

            GlobalScope.async {

                val time = System.currentTimeMillis()
                val (errorVector, steps) = learningProcess(network, teacher)

                network to Result(teacher, steps, errorVector, time)
            }
        }.toMutableList()

        val loader = ConsoleLoader(0, ConsoleLoader.Mode.Fraction)

        runBlocking {

            val all = awaits.size

            while (awaits.size != 0) {

                val finished = awaits.filter { it.isCompleted }
                awaits.removeAll(finished)

                finished.forEach {

                    val (network, result) = it.await()

                    afterLearning(network, result.teacher, result.errorVector, result.steps)
                    print("\r")
                    singleNetworkLog(network, result.errorVector, result.steps, result.time)
                }

                val progress = all - awaits.size

                loader.update(progress, all)
                loader.print()

                delay(250)
            }

            loader.close()
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

    /**
     * Initializes whole learning process.
     */
    @Throws(UnfulfilledExpectationsException::class)
    override fun run() {

        setup()

        teachers.addAll(buildTeachers())
        networks.addAll(buildNetworks())

        val time = System.currentTimeMillis()

        asyncRunner()

        println("Total elapsed time: ${System.currentTimeMillis() - time} ms")

        allNetworksReady()
    }
}

private data class Result(
    val teacher: NetworkTeacher,
    val steps: Int,
    val errorVector: List<Double>,
    val time: Long
)