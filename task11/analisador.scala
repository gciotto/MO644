// Gustavo Ciotto Pinton - task #11
// RA 117136

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

    // Splits it up into words and replaces punctuation
    val words1 =  lines1.flatMap(line => line.split(" "))

    // Counts words
    val filteredWords1 = words1.map(word => word.toLowerCase.replaceAll("[,.!?:;]","")).filter(word => word.length > 3)

    val reducedWords1 = filteredWords1.map(word => (word, 1)).reduceByKey{case (x, y) => x + y}

    val sortedWords1 = reducedWords1.sortBy(- _._2)

    // Prints 5-most used words in text 1
    sortedWords1.take(5).foreach(x => println(x._1 + "=" + x._2))

    println("TEXT2")

    // read second text file and split each document into words
    val lines2 = sc.textFile(args(1))

    // Splits it up into words and replaces punctuation
    val words2 =  lines2.flatMap(line => line.split(" "))

    // Counts words
    val filteredWords2 = words2.map(word => word.toLowerCase.replaceAll("[,.!?:;]","")).filter(word => word.length > 3)

    val reducedWords2 = filteredWords2.map(word => (word, 1)).reduceByKey{case (x, y) => x + y}

    val sortedWords2 = reducedWords2.sortBy(- _._2)

    // Prints 5-most used words in text 1
    sortedWords2.take(5).foreach(x => println(x._1 + "=" + x._2))

    println("COMMON")

    val filtered100Words1 = reducedWords1.filter (x => x._2 > 100)
    val filtered100Words2 = reducedWords2.filter (x => x._2 > 100)

    val sameWords = filtered100Words1.keys.intersection(filtered100Words2.keys)
 
    // takeOrdered() gets all key sorted
    sameWords.takeOrdered(sameWords.count().toInt).foreach(x => println(x))

  }

}

/*

Resultados:

spark-submit --class Analisador target/scala-2.11/analisador_2.11-0.1.jar alice30.txt wizoz10.txt
TEXT1
said=456
alice=377
that=234
with=172
very=139
TEXT2
they=390
that=350
dorothy=340
said=331
with=274
COMMON
little
said
that
they
this
with

spark-submit --class Analisador target/scala-2.11/analisador_2.11-0.1.jar alice30.txt warw10.txt
TEXT1
said=456
alice=377
that=234
with=172
very=139
TEXT2
that=759
with=448
were=365
from=326
they=302
COMMON
little
said
that
they
this
with

spark-submit --class Analisador target/scala-2.11/analisador_2.11-0.1.jar alice30.txt gmars11.txt
TEXT1
said=456
alice=377
that=234
with=172
very=139
TEXT2
that=1235
with=633
from=599
upon=523
were=458
COMMON
said
that
they
this
with

spark-submit --class Analisador target/scala-2.11/analisador_2.11-0.1.jar warw10.txt wizoz10.txt
TEXT1
that=759
with=448
were=365
from=326
they=302
TEXT2
they=390
that=350
dorothy=340
said=331
with=274
COMMON
came
could
from
have
little
said
that
them
then
there
they
this
were
with

spark-submit --class Analisador target/scala-2.11/analisador_2.11-0.1.jar warw10.txt gmars11.txt
TEXT1
that=759
with=448
were=365
from=326
they=302
TEXT2
that=1235
with=633
from=599
upon=523
were=458
COMMON
about
again
been
black
came
could
from
have
into
said
that
their
them
then
there
they
this
through
time
upon
were
with

spark-submit --class Analisador target/scala-2.11/analisador_2.11-0.1.jar wizoz10.txt gmars11.txt
TEXT1
they=390
that=350
dorothy=340
said=331
with=274
TEXT2
that=1235
with=633
from=599
upon=523
were=458
COMMON
came
could
from
great
have
said
that
them
then
there
they
this
were
when
will
with
would


*/
