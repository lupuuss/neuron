package ml.freeze

import com.google.gson.Gson
import com.google.gson.GsonBuilder
import ml.spine.Network
import java.io.File
import java.lang.IllegalArgumentException
import java.nio.charset.Charset
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

object NetworkFreezer {

    private val charset: Charset = Charsets.UTF_8

    private val gson = GsonBuilder()
        .setPrettyPrinting()
        .create()

    private val defaultPath = ".\\freezer"
    private val extension = ".frz"
    private var rootDirectory: Path = Paths.get(defaultPath).also { validatePathAsRoot(it) }

    private fun validatePathAsRoot(path: Path) {

        if (!path.toFile().exists()) {

            Files.createDirectory(path)

        } else if (!Files.isDirectory(path)) {

            throw IllegalArgumentException("$path is not a directory!")
        }
    }

    fun setFreezerPath(path: String) {

        this.rootDirectory = Paths.get(path)
    }

    fun freeze(network: Network) {
        val file = File(rootDirectory.toFile(), "${network.name}$extension")
        file.outputStream().write(gson.toJson(network).toByteArray(charset))
    }

    fun unfreezeFile(networkName: String): Network {
        val fileName = if (networkName.endsWith(extension)) networkName else "$networkName$extension"
        val file = File(rootDirectory.toFile(), fileName)

        return gson.fromJson(String(file.inputStream().readAllBytes(), charset), Network::class.java)
    }
}