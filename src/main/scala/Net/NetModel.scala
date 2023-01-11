package CNN

import scala.io.{BufferedSource, Source}
import scala.runtime.ScalaRunTime._
import java.io._

class NetModel{
    var net_structure = ""
    var in_height = Array(1)
    var in_channel = Array(1)
    var out_channel = Array(1)
    var last_in_channel = Array(1)
    var pre_in_channel = Array(1)
    var kernel_size_row = Array(1)
    var kernel_size_col = Array(1)
    var kernel_stride_row = Array(1)
    var kernel_stride_col = Array(1)
    var pooling_kernel_size_row = Array(1)
    var pooling_kernel_size_col = Array(1)
    var pooling_kernel_stride_row = Array(1)
    var pooling_kernel_stride_col = Array(1)
    var fc_in_size = Array(1)
    var fc_out_size = Array(1)
    var mini_fc_len = Array(1)
    var unified_cycle = 1
    var multiple = Array(1)

    def str_to_int(str_array: Array[String]): Array[Int] ={
        var number_array = new Array[Int](str_array.size) 
        for(i <- 0 until str_array.size){
            number_array(i) = str_array(i).toInt
        }
        return number_array
    }

    def read_file(path: String): Unit ={
        val file: BufferedSource = Source.fromFile(path)
        for (line <- file.getLines()){
            var line_list = line.split(":")
            line_list(0) match{
                case "net_structure" =>{
                    net_structure = line_list(1)
                }
                case "in_height" =>{
                    in_height = str_to_int(line_list(1).split(","))
                }
                case "in_channel" =>{
                    in_channel = str_to_int(line_list(1).split(","))
                }
                case "out_channel" =>{
                    out_channel = str_to_int(line_list(1).split(","))
                }
                case "last_in_channel" =>{
                    last_in_channel = str_to_int(line_list(1).split(","))
                }
                case "pre_in_channel" =>{
                    pre_in_channel = str_to_int(line_list(1).split(","))
                }
                case "kernel_size_row" =>{
                    kernel_size_row = str_to_int(line_list(1).split(","))
                }
                case "kernel_size_col" =>{
                    kernel_size_col = str_to_int(line_list(1).split(","))
                }
                case "kernel_stride_row" =>{
                    kernel_stride_row = str_to_int(line_list(1).split(","))
                }
                case "kernel_stride_col" =>{
                    kernel_stride_col = str_to_int(line_list(1).split(","))
                }
                case "pooling_kernel_size_row" =>{
                    pooling_kernel_size_row = str_to_int(line_list(1).split(","))
                }
                case "pooling_kernel_size_col" =>{
                    pooling_kernel_size_col = str_to_int(line_list(1).split(","))
                }
                case "pooling_kernel_stride_row" =>{
                    pooling_kernel_stride_row = str_to_int(line_list(1).split(","))
                }
                case "pooling_kernel_stride_col" =>{
                    pooling_kernel_stride_col = str_to_int(line_list(1).split(","))
                }
                case "fc_in_size" =>{
                    fc_in_size = str_to_int(line_list(1).split(","))
                }
                case "fc_out_size" =>{
                    fc_out_size = str_to_int(line_list(1).split(","))
                }
                case "mini_fc_len" =>{
                    mini_fc_len = str_to_int(line_list(1).split(","))
                }
                case "unified_cycle" =>{
                    unified_cycle = line_list(1).toInt
                }
                case "multiple" =>{
                    multiple = str_to_int(line_list(1).split(","))
                }
                case _ => ;
            }
        }
        file.close()
    }

    def print_this(): Unit ={
        println("net_structure:" + net_structure)
        print("last_in_channel:")
        println(stringOf(last_in_channel))
        print("unified_cycle:")
        println(unified_cycle)
        print("multiple:")
        println(stringOf(multiple))
    }
}


class BaseNetModel{
    var net_structure = ""
    var in_height = Array(1)
    var in_channel = Array(1)
    var out_channel = Array(1)
    var mini_in_channel = Array(1)
    var kernel_size_row = Array(1)
    var kernel_size_col = Array(1)
    var kernel_stride_row = Array(1)
    var kernel_stride_col = Array(1)
    var pooling_kernel_size_row = Array(1)
    var pooling_kernel_size_col = Array(1)
    var pooling_kernel_stride_row = Array(1)
    var pooling_kernel_stride_col = Array(1)
    var fc_in_size = Array(1)
    var fc_out_size = Array(1)
    var mini_fc_len = Array(1)
    var unified_cycle = 1

    def str_to_int(str_array: Array[String]): Array[Int] ={
        var number_array = new Array[Int](str_array.size) 
        for(i <- 0 until str_array.size){
            number_array(i) = str_array(i).toInt
        }
        return number_array
    }

    def read_file(path: String): Unit ={
        val file: BufferedSource = Source.fromFile(path)
        for (line <- file.getLines()){
            var line_list = line.split(":")
            line_list(0) match{
                case "net_structure" =>{
                    net_structure = line_list(1)
                }
                case "in_height" =>{
                    in_height = str_to_int(line_list(1).split(","))
                }
                case "in_channel" =>{
                    in_channel = str_to_int(line_list(1).split(","))
                }
                case "out_channel" =>{
                    out_channel = str_to_int(line_list(1).split(","))
                }
                case "mini_in_channel" =>{
                    mini_in_channel = str_to_int(line_list(1).split(","))
                }
                case "kernel_size_row" =>{
                    kernel_size_row = str_to_int(line_list(1).split(","))
                }
                case "kernel_size_col" =>{
                    kernel_size_col = str_to_int(line_list(1).split(","))
                }
                case "kernel_stride_row" =>{
                    kernel_stride_row = str_to_int(line_list(1).split(","))
                }
                case "kernel_stride_col" =>{
                    kernel_stride_col = str_to_int(line_list(1).split(","))
                }
                case "pooling_kernel_size_row" =>{
                    pooling_kernel_size_row = str_to_int(line_list(1).split(","))
                }
                case "pooling_kernel_size_col" =>{
                    pooling_kernel_size_col = str_to_int(line_list(1).split(","))
                }
                case "pooling_kernel_stride_row" =>{
                    pooling_kernel_stride_row = str_to_int(line_list(1).split(","))
                }
                case "pooling_kernel_stride_col" =>{
                    pooling_kernel_stride_col = str_to_int(line_list(1).split(","))
                }
                case "fc_in_size" =>{
                    fc_in_size = str_to_int(line_list(1).split(","))
                }
                case "fc_out_size" =>{
                    fc_out_size = str_to_int(line_list(1).split(","))
                }
                case "mini_fc_len" =>{
                    mini_fc_len = str_to_int(line_list(1).split(","))
                }
                case "unified_cycle" =>{
                    unified_cycle = line_list(1).toInt
                }
                case _ => ;
            }
        }
        file.close()
    }

    def print_this(): Unit ={
        println("net_structure:" + net_structure)
        print("mini_in_channel:")
        println(stringOf(mini_in_channel))
        print("unified_cycle:")
        println(unified_cycle)
    }
}