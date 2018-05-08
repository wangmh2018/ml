package org.dataalgorithms.chap13.scala

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * kNN algorithm in scala/spark.
 * 
 * @author Gaurav Bhardwaj (gauravbhardwajemail@gmail.com)
 *
 * @editor Mahmoud Parsian (mahmoud.parsian@yahoo.com)
 *
 */
object kNN {
  //定义主函数
  def main(args: Array[String]): Unit = {
    //如果输入的参数小于五个，则输出"Usage: kNN <k-knn> <d-dimension> <R-input-dir> <S-input-dir> <output-dir>"
    //参数分别为k个近邻，数据维度，训练集和测试集路径，输出路径     
    if (args.size < 5) {
      println("Usage: kNN <k-knn> <d-dimension> <R-input-dir> <S-input-dir> <output-dir>")
      sys.exit(1)
    }
    //    设置环境变量参数，把程序名设置为kNN    
    val sparkConf = new SparkConf().setAppName("kNN")
    //    建立sc对象     
    val sc = new SparkContext(sparkConf)
    //    将近邻k和数据维度d转化为整型
    val k = args(0).toInt 
    val d = args(1).toInt0
    //    将训练集、测试集路径和输出路径保存到常量中     
    val inputDatasetR = args(2)
    val inputDatasetS = args(3)
    val output = args(4)
    //    将K和d发往每个executor
    val broadcastK = sc.broadcast(k);
    val broadcastD = sc.broadcast(d)
    //    从变量保存的地址中读取训练集和测试集 
    val R = sc.textFile(inputDatasetR)
    val S = sc.textFile(inputDatasetS)

    /**
     *计算两点的距离
     * Calculate the distance between 2 points
     * 训练集和测试集的维度都为d
     * @param rAsString as r1,r2, ..., rd
     * @param sAsString as s1,s2, ..., sd
     * @param d as dimention
     */
    //     函数名为calculateDistance，函数返回类型为双精度类型。
    //     参数为rAsString（训练集，类型为字符串类型），sAsString（测试集，类型为字符串类型）
    //     d为样本维度，类型为整型，
    def calculateDistance(rAsString: String, sAsString: String, d: Int): Double = {
    //     将训练集和测试集以逗号分隔，并经过map算子，将每一个元素转为双精度类型
      val r = rAsString.split(",").map(_.toDouble)
      val s = sAsString.split(",").map(_.toDouble)
    //     如果长度不为d，则返回双精度的空值
      if (r.length != d || s.length != d) Double.NaN else {
    //     否则计算距离
    //     以下是关于zipped作用的示例    
    //     scala> val values = List.range(1, 5)
    //     values: List[Int] = List(1, 2, 3, 4)
    //     scala> val sumOfSquares = (values, values).zipped map (_ * _) sum
    //     sumOfSquares: Int = 30 
    //     take取出前d项进行操作   
        math.sqrt((r, s).zipped.take(d).map { case (ri, si) => math.pow((ri - si), 2) }.reduce(_ + _))
      }
    }
    //     两个RDD进行笛卡尔积合并
    val cart = R cartesian S
    //     
    val knnMapped = cart.map(cartRecord => {
      val rRecord = cartRecord._1
      val sRecord = cartRecord._2
      val rTokens = rRecord.split(";")
      val rRecordID = rTokens(0)
      val r = rTokens(1) // r.1, r.2, ..., r.d
      val sTokens = sRecord.split(";")
      val sClassificationID = sTokens(1)
      val s = sTokens(2) // s.1, s.2, ..., s.d
      val distance = calculateDistance(r, s, broadcastD.value)
      (rRecordID, (distance, sClassificationID))
    })

    // note that groupByKey() provides an expensive solution 
    // [you must have enough memory/RAM to hold all values for 
    // a given key -- otherwise you might get OOM error], but
    // combineByKey() and reduceByKey() will give a better 
    // scale-out performance
    val knnGrouped = knnMapped.groupByKey()

    val knnOutput = knnGrouped.mapValues(itr => {
      val nearestK = itr.toList.sortBy(_._1).take(broadcastK.value)
      val majority = nearestK.map(f => (f._2, 1)).groupBy(_._1).mapValues(list => {
        val (stringList, intlist) = list.unzip
        intlist.sum
      })
      majority.maxBy(_._2)._1
    })

    // for debugging purpose
    knnOutput.foreach(println)
    
    // save output
    knnOutput.saveAsTextFile(output)

    // done!
    sc.stop()
  }
}
