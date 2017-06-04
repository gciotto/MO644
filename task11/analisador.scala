import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object Analisador {

  // Args = path/to/text0.txt path/to/text1.txt
  def main(args: Array[String]) {

    // create Spark context with Spark configuration
    val sc = new SparkContext(new SparkConf().setAppName("Contagem de Palavra"))

    println("TEXT1")

    // read first text file and split into lines
    val lines1 = sc.textFile(args(0))

    // TODO: contar palavras do texto 1 e imprimir as 5 palavras com as maiores ocorrencias (ordem DECRESCENTE)
    // imprimir na cada linha: "palavra=numero"

    // Splits it up into words and replaces punctuation
    val words =  lines1.flatMap(line => line.split(" ")).replaceAll("[,.!?:;]","")

    // Counts words
    val counts =  words.filter(word => word.length > 3).map(word => (word, 1)).reduceByKey{case (x, y) => x + y}.sortBy(_._2)

    counts.take(5).foreach(x => println(x._1 + "=" + x._2)))

    println("TEXT2")

    // read second text file and split each document into words
    val lines2 = sc.textFile(args(1))

    // TODO: contar palavras do texto 2 e imprimir as 5 palavras com as maiores ocorrencias (ordem DECRESCENTE)
    // imprimir na cada linha: "palavra=numero"

    println("COMMON")

    // TODO: comparar resultado e imprimir na ordem ALFABETICA todas as palavras que aparecem MAIS que 100 vezes nos 2 textos
    // imprimir na cada linha: "palavra"

  }

}
