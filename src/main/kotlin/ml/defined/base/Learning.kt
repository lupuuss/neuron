package ml.defined.base

import ml.defined.*
import ml.input.DataParser
import ml.learn.NetworkTeacher
import ml.spine.Network

abstract class Learning(
    protected val config: Config
) {
    enum class Type {
        Exercise3,
        Transformation, Transformation100, TransformationHidden,
        Approximation, Approximation100, ApproximationProgress,
        Iris,
    }

    protected abstract val errorGoal: Double
    protected abstract val stepsLimit: Int

    protected val dataParser: DataParser = DataParser(config.separator)
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
     * Performs standard learning process for each network.
     */
    protected abstract fun learningProcess(network: Network, teacher: NetworkTeacher): Pair<List<Double>, Int>

    /**
     * It's called when every network is learned.
     */
    protected open fun allNetworksReady(restored: Boolean) {}

    abstract fun run()

    companion object {

        @JvmStatic
        fun get(type: Type, config: Config): Learning = when (type) {
            Type.Exercise3 -> Exercise3(config)
            Type.Transformation -> Transformation(config)
            Type.Approximation -> Approximation(config)
            Type.Approximation100 -> Approximation100(config)
            Type.ApproximationProgress -> ApproximationProgress(config)
            Type.Iris -> Iris(config)
            Type.Transformation100 -> Transformation100(config)
            Type.TransformationHidden -> TransformationHidden(config)
        }
    }
}