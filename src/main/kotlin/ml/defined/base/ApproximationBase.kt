package ml.defined.base

import ml.input.LearnData
import ml.learn.NetworkTeacher

abstract class ApproximationBase(config: Config) : NetworksLearning(config) {

    protected val alpha = 0.01
    protected val beta = 0.9
    protected val teacherMode = NetworkTeacher.Mode.Offline


    protected lateinit var training: List<LearnData>
    protected lateinit var verification: LearnData

    override fun setup() {

        if (config.inputs.size != 3) {
            throw UnfulfilledExpectationsException("Approximation task requires training data (2 files) and verification data (1 file) ")
        }

        training = listOf(dataParser.parse(config.inputs[0], 1, 1), dataParser.parse(config.inputs[1], 1, 1))
        verification = dataParser.parse(config.inputs[2], 1, 1)
    }

}