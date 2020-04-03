package ml.defined.base

import ml.defined.*
import ml.defined.exceptions.UnfulfilledExpectationsException
import ml.input.ParsingException
import ml.learn.NetworkTeacher
import ml.spine.Network

abstract class Learning(
    protected val config: Config
) {
    enum class Type {
        Exercise3, Transformation, Approximation, Iris
    }

    protected abstract val errorGoal: Double
    protected abstract val stepsLimit: Int

    protected val teachers: MutableList<NetworkTeacher> = mutableListOf()

    /**
     * Should load all necessary data for learning. It is called only once on start.
     */
    @Throws(UnfulfilledExpectationsException::class, ParsingException::class)
    protected abstract fun setup()

    /**
     * Should build all teachers that will be used. It is called only once on start.
     * Amount of teachers must be the same as networks.
     */
    protected abstract fun buildTeachers(): List<NetworkTeacher>

    /**
     * It is called before learning process for each network. On this step network is unlearned.
     * It might be not called if a neural network was unfreezed.
     */
    protected open fun beforeLearning(network: Network, teacher: NetworkTeacher) {}

    /**
     * Performs standard learning process for each network.
     * It might be not called if a neural network was unfreezed.
     */
    protected fun learningProcess(network: Network, teacher: NetworkTeacher): Pair<List<Double>, Int> {

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

    abstract fun run()

    companion object {

        @JvmStatic
        fun get(type: Type, config: Config): NetworksLearning = when (type) {
            Type.Exercise3 -> Exercise3(
                config
            )
            Type.Transformation -> Transformation(
                config
            )
            Type.Approximation -> Approximation(
                config
            )
            Type.Iris -> Iris(config)
        }
    }
}