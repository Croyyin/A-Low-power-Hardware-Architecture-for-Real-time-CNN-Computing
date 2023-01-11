package CNN

import scala.util.Random
import chisel3._
import chiseltest._
import org.scalatest._
import scala.io.Source
import chisel3.util._
import chisel3.experimental._
import scala.util.control.Breaks._
import chiseltest.experimental.TestOptionBuilder._
import chiseltest.internal.WriteVcdAnnotation
import scala.util.control._
import Array._
import scala.math.floor
import scala.math.pow
import scala.io.Source
import java.io.PrintWriter





class Batch_Adder_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    behavior of "test"
    it should "test channel adder" in {
        test(new Batch_Adder(7)){ c=> 
            var i = 0
            var previous = 0
            for(i <- 0 until 18){
                var input_p = previous
                if(i % 7 == 0){
                    previous = 0
                }
                c.io.current_input.poke((i % 7).S)
                c.io.previous_input.poke(input_p.S)
                var result = (i % 7) + previous
                c.io.result.expect(result.S)
                previous = result
                c.clock.step(1)
            }
            
        }
    }
}

class Convolutional_Layer_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    def convolution_one_out(data: Array[Array[Array[Int]]], weights: Array[Array[Array[Int]]], bias: Int, ic: Int, oc: Int, kr: Int, kc: Int): Array[Int] ={
        
        var ps = Array.ofDim[Int](ic / oc + 2)
        ps(0) = 0
        for(t <- 0 until ic / oc){
            var sum = 0
            for(i <- 0 until oc){
                for(j <- 0 until kr){
                    for(k <- 0 until kc){
                        sum = sum + data(t * oc + i)(j)(k) * weights(t * oc + i)(j)(k)
                    }
                }
            }
            ps(t + 1) = ps(t) + sum
        }
        ps(ic / oc + 1) = ps(ic / oc) + bias
        return ps
    }

    behavior of "test"
    it should "test conv" in {
        test(new Partial_Convolutional_Layer(16, 4, 3, 3, false)){ c=> 

            val input_data = Array.ofDim[Int](16, 3, 3)
            for(i <- 0 until 16){
                for(j <- 0 until 3){
                    for(k <- 0 until 3){
                        input_data(i)(j)(k) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                    }
                }
            }
            
            val input_weight = Array.ofDim[Int](16, 3, 3)
            for(i <- 0 until 16){
                for(j <- 0 until 3){
                    for(k <- 0 until 3){
                        input_weight(i)(j)(k) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                    }
                }
            }

            val input_bias = 0

            val result = convolution_one_out(input_data, input_weight, input_bias, 16, 4, 3, 3)
            

            c.io.data_zero.poke(0.S)
            c.io.weight_zero.poke(0.S)
            for(t <- 0 until 4){
                for(i <- 0 until 4){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            c.io.data(i)(j)(k).poke(input_data(t * 4 + i)(j)(k).S)
                            c.io.weights(i)(j)(k).poke(input_weight(t * 4 + i)(j)(k).S)
                        }
                    }
                }
                print(c.io.result.peek())
                c.clock.step(1)
            }
            
            println(result(5))
        }
    }
}


class Reg_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    behavior of "test"
    it should "test shift reg" in {
        test(new Partial_Shift_Reg_Array(8, 8, 9, 3, 2, 3, 2)){ c=> 
            var ps = Array.ofDim[Int](8, 9, 3)

            for(i <- 0 until 8){
                for(j <- 0 until 9){
                    for(k <- 0 until 3){
                        ps(i)(j)(k) = Random.nextInt(15)
                    }
                }
            }

            // print
            for(i <- 0 until 8){
                for(j <- 0 until 9){
                    for(k <- 0 until 3){
                        print(ps(i)(j)(k))
                        print(" ")
                    }
                    println()
                }
                print("----------------")
                println()
            }
            
            for(i <- 0 until 8){
                for(j <- 0 until 9){
                    for(k <- 0 until 3){
                        c.io.in_data(i)(j)(k).poke(ps(i)(j)(k).S)
                    }
                }
            }
            c.io.signal.poke(1.U)
            for(t <- 0 until 25){
                print("vvvvvvvvvvvvvvvvvvvvvvvv")
                println()
                for(i <- 0 until 2){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            print(c.io.out_data(i)(j)(k).peek())
                            print(" ")
                        }
                        println()
                    }
                    print("-----")
                    println()
                }
                print("AAAAAAAAAAAAAAAAAAAAAAAA")
                println()
                c.clock.step(1)
                c.io.signal.poke(0.U)
            }
        }
    }
}

class Reg_Convolutional_Layer_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    def convolution_one_out(data: Array[Array[Array[Int]]], weights: Array[Array[Array[Array[Int]]]], bias: Array[Int], stride: Int): Array[Array[Int]] ={
        val width = data(0)(0).length
        val height = data(0).length
        val in_channel = data.length
        val out_channel = weights.length

        val kernel_size_row = weights(0)(0).length
        val kernel_size_col = weights(0)(0)(0).length

        var out_height = ((height - kernel_size_row) / stride).toInt + 1

        val result = Array.ofDim[Int](out_channel, out_height)
        for(h <- 0 until out_height){
            for(oc <- 0 until out_channel){
                var sum = 0
                for(ic <- 0 until in_channel){
                    for(kr <- 0 until kernel_size_row){
                        for(kc <- 0 until kernel_size_col){
                            sum = sum + data(ic)(kr + h * stride)(kc) * weights(oc)(ic)(kr)(kc)
                        }
                    }
                }
                result(oc)(h) = sum + bias(oc)
            }
        }
        return result
    }

    behavior of "test"
    it should "test conv Reg" in {
        test(new Reg_Convolutional_Layer_Reg(9, 4, 8, 2, 3, 3, 2, 70)){ c=> 

            val input_data = Array.ofDim[Int](8, 9, 3)
            for(i <- 0 until 8){
                for(j <- 0 until 9){
                    for(k <- 0 until 3){
                        if(k == 2){
                            input_data(i)(j)(k) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                        }else{
                            input_data(i)(j)(k) = 1
                        }
                        
                    }
                }
            }
            
            val input_weight = Array.ofDim[Int](4, 8, 3, 3)
            for(l <- 0 until 4){
                for(i <- 0 until 8){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            input_weight(l)(i)(j)(k) = 1
                        }
                    }
                }
            }
                
            val input_bias = Array.ofDim[Int](4)
            for(l <- 0 until 4){
                input_bias(l) = 1
            }

            val input_data2 = Array.ofDim[Int](8, 9, 3)
            for(i <- 0 until 8){
                for(j <- 0 until 9){
                    for(k <- 0 until 3){
                        if(k == 2){
                            input_data2(i)(j)(k) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                        }else{
                            input_data2(i)(j)(k) = 1
                        }
                        
                    }
                }
            }
            
            val input_weight2 = Array.ofDim[Int](4, 8, 3, 3)
            for(l <- 0 until 4){
                for(i <- 0 until 8){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            input_weight2(l)(i)(j)(k) = 1
                        }
                    }
                }
            }
                
            val input_bias2 = Array.ofDim[Int](4)
            for(l <- 0 until 4){
                input_bias2(l) = 1
            }


            val result = convolution_one_out(input_data, input_weight, input_bias, 2)
            val result2 = convolution_one_out(input_data2, input_weight2, input_bias2, 2)
            c.io.bias.poke(1.S)
            c.io.data_zero.poke(0.S)
            c.io.weight_zero.poke(0.S)
            c.io.result_zreo.poke(0.S)
            c.io.m_of_scale.poke(FixedPoint.fromDouble((1.0).toDouble, 64.W, 32.BP))

            for(l <- 0 until 8){
                for(r <- 0 until 9){
                    c.io.reg_in(l)(r)(0).poke(input_data(l)(r)(2).S)
                }
            }

            for(i <- 0 until 2){
                for(j <- 0 until 3){
                    for(k <- 0 until 2){
                        c.io.data(i)(j)(k).poke(input_data(i)(j)(k).S)
                    }
                }
            }

            for(i <- 0 until 2){
                for(j <- 0 until 3){
                    for(k <- 0 until 3){
                        c.io.weights(i)(j)(k).poke(input_weight(0)(i)(j)(k).S)
                    }
                }
            }
            
            for(oh <- 0 until 4){
                for(oc <- 0 until 4){
                    print(result(oc)(oh))
                    print(" ")
                }
                println()
            }
            println("_-_-_-_-_")
            for(oh <- 0 until 4){
                for(oc <- 0 until 4){
                    print(result2(oc)(oh))
                    print(" ")
                }
                println()
            }

            for(loop <- 0 until 150){
                if(loop > 48){
                    for(l <- 0 until 8){
                        for(r <- 0 until 9){
                            c.io.reg_in(l)(r)(0).poke(input_data2(l)(r)(2).S)
                        }
                    }

                    for(i <- 0 until 2){
                        for(j <- 0 until 3){
                            for(k <- 0 until 2){
                                c.io.data(i)(j)(k).poke(input_data2(i)(j)(k).S)
                            }
                        }
                    }

                    for(i <- 0 until 2){
                        for(j <- 0 until 3){
                            for(k <- 0 until 3){
                                c.io.weights(i)(j)(k).poke(input_weight2(0)(i)(j)(k).S)
                            }
                        }
                    }
                }
                if((loop < 81 && loop > 60) || (loop < 140 && loop > 130)){
                    // result print
                    for(oh <- 0 until 4){
                        for(oc <- 0 until 4){
                            print(c.io.reg_out(oc)(oh)(0).peek())
                            print(" ")
                        }
                        println()
                    }
                    print("-------")
                    print(loop)
                    println("-------")
                }
                
                

                c.clock.step(1)
            }
            
        }
    }
}

class Reg_Convolutional_Pooling_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    def convolution_one_out(data: Array[Array[Array[Int]]], weights: Array[Array[Array[Array[Int]]]], bias: Array[Int], stride: Int): Array[Array[Int]] ={
        val width = data(0)(0).length
        val height = data(0).length
        val in_channel = data.length
        val out_channel = weights.length

        val kernel_size_row = weights(0)(0).length
        val kernel_size_col = weights(0)(0)(0).length

        var out_height = ((height - kernel_size_row) / stride).toInt + 1

        val result = Array.ofDim[Int](out_channel, out_height)
        for(h <- 0 until out_height){
            for(oc <- 0 until out_channel){
                var sum = 0
                for(ic <- 0 until in_channel){
                    for(kr <- 0 until kernel_size_row){
                        for(kc <- 0 until kernel_size_col){
                            sum = sum + data(ic)(kr + h * stride)(kc) * weights(oc)(ic)(kr)(kc)
                        }
                    }
                }
                result(oc)(h) = sum + bias(oc)
            }
        }
        return result
    }

    behavior of "test"
    it should "test conv Reg" in {
        test(new Reg_Convolutional_Layer_Reg_Pooling_layer_Reg(9, 4, 8, 2, 3, 3, 2, 2, 2, 2, true, 70)){ c=> 

            val input_data = Array.ofDim[Int](8, 9, 3)
            for(i <- 0 until 8){
                for(j <- 0 until 9){
                    for(k <- 0 until 3){
                        if(k == 2){
                            input_data(i)(j)(k) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                        }else{
                            input_data(i)(j)(k) = 1
                        }
                        
                    }
                }
            }
            
            val input_weight = Array.ofDim[Int](4, 8, 3, 3)
            for(l <- 0 until 4){
                for(i <- 0 until 8){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            input_weight(l)(i)(j)(k) = 1
                        }
                    }
                }
            }
                
            val input_bias = Array.ofDim[Int](4)
            for(l <- 0 until 4){
                input_bias(l) = 1
            }

            val input_data2 = Array.ofDim[Int](8, 9, 3)
            for(i <- 0 until 8){
                for(j <- 0 until 9){
                    for(k <- 0 until 3){
                        if(k == 2){
                            input_data2(i)(j)(k) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                        }else{
                            input_data2(i)(j)(k) = 1
                        }
                        
                    }
                }
            }
            
            val input_weight2 = Array.ofDim[Int](4, 8, 3, 3)
            for(l <- 0 until 4){
                for(i <- 0 until 8){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            input_weight2(l)(i)(j)(k) = 1
                        }
                    }
                }
            }
                
            val input_bias2 = Array.ofDim[Int](4)
            for(l <- 0 until 4){
                input_bias2(l) = 1
            }


            val result = convolution_one_out(input_data, input_weight, input_bias, 2)
            val result2 = convolution_one_out(input_data2, input_weight2, input_bias2, 2)
            c.io.bias.poke(1.S)
            c.io.data_zero.poke(0.S)
            c.io.weight_zero.poke(0.S)
            c.io.result_zreo.poke(0.S)
            c.io.m_of_scale.poke(FixedPoint.fromDouble((1.0).toDouble, 64.W, 32.BP))
            c.io.pool_data(0)(0)(0).poke(0.S)
            c.io.pool_data(0)(1)(0).poke(0.S)

            for(l <- 0 until 8){
                for(r <- 0 until 9){
                    c.io.reg_in(l)(r)(0).poke(input_data(l)(r)(2).S)
                }
            }

            for(i <- 0 until 2){
                for(j <- 0 until 3){
                    for(k <- 0 until 2){
                        c.io.data(i)(j)(k).poke(input_data(i)(j)(k).S)
                    }
                }
            }

            for(i <- 0 until 2){
                for(j <- 0 until 3){
                    for(k <- 0 until 3){
                        c.io.weights(i)(j)(k).poke(input_weight(0)(i)(j)(k).S)
                    }
                }
            }
            
            for(oh <- 0 until 4){
                for(oc <- 0 until 4){
                    print(result(oc)(oh))
                    print(" ")
                }
                println()
            }
            println("_-_-_-_-_")
            for(oh <- 0 until 4){
                for(oc <- 0 until 4){
                    print(result2(oc)(oh))
                    print(" ")
                }
                println()
            }

            for(loop <- 0 until 150){
                if(loop > 48){
                    for(l <- 0 until 8){
                        for(r <- 0 until 9){
                            c.io.reg_in(l)(r)(0).poke(input_data2(l)(r)(2).S)
                        }
                    }

                    for(i <- 0 until 2){
                        for(j <- 0 until 3){
                            for(k <- 0 until 2){
                                c.io.data(i)(j)(k).poke(input_data2(i)(j)(k).S)
                            }
                        }
                    }

                    for(i <- 0 until 2){
                        for(j <- 0 until 3){
                            for(k <- 0 until 3){
                                c.io.weights(i)(j)(k).poke(input_weight2(0)(i)(j)(k).S)
                            }
                        }
                    }
                }
                if((loop < 81 && loop > 60) || (loop < 140 && loop > 130)){
                    // result print
                    for(oh <- 0 until 2){
                        for(oc <- 0 until 4){
                            print(c.io.reg_out(oc)(oh)(0).peek())
                            print(" ")
                        }
                        println()
                    }
                    print(c.io.c4.peek())
                    println("  ")
                    print("-------")
                    print(loop)
                    println("-------")
                }
                
                

                c.clock.step(1)
            }
            
        }
    }
}

class FC_Layer_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    def fc_one_compute(data: Array[Int], weights: Array[Array[Int]], bias: Array[Int]): Array[Int] ={
        val height = data.length
        val classes = weights.length

        val result = Array.ofDim[Int](classes)
        for(h <- 0 until classes){
            var sum = 0
            for(oc <- 0 until height){
                sum = sum + data(oc) * weights(h)(oc)
            }
            result(h) = sum + bias(h)
        }
        return result
    }

    behavior of "test"
    it should "test conv Reg" in {
        test(new FC_Layer_Reg(5, 15, 5, 30)){ c=> 
            val mini_length = 5
            val input_data = Array.ofDim[Int](15)
            for(i <- 0 until 15){
                input_data(i) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
            }
            
            val input_weight = Array.ofDim[Int](5, 15)
            for(l <- 0 until 5){
                for(i <- 0 until 15){
                        input_weight(l)(i) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                    
                }
            }
                
            val input_bias = Array.ofDim[Int](5)
            for(l <- 0 until 4){
                input_bias(l) = 0
            }

            val result = fc_one_compute(input_data, input_weight, input_bias)
            c.io.bias.poke(0.S)
            c.io.data_zero.poke(0.S)
            c.io.weight_zero.poke(0.S)
            c.io.result_zreo.poke(0.S)
            c.io.m_of_scale.poke(FixedPoint.fromDouble((1.0).toDouble, 64.W, 32.BP))
            
            for(oh <- 0 until 5){
                print(result(oh))
                print(" ")
                
            }
            println()
            val len_times = 15 / mini_length

            for(loop <- 0 until 50){
                print("-----loop")
                print(loop)
                println("-----")
                if(loop >= 1){
                    val len_index = (loop-1) % len_times
                    val weight_index = ((loop-1) / len_times).toInt % 5
                    for(i <- 0 until mini_length){
                        c.io.data(i).poke(input_data(len_index * mini_length + i).S)
                    }
                    for(i <- 0 until mini_length){
                        c.io.weights(i).poke(input_weight(weight_index)(len_index * mini_length + i).S)
                    }

                    for(i <- 0 until 5){
                        print(c.io.reg_out(i).peek())
                        print(" ")
                    }
                }
                println()
                println("-----------------")
                c.clock.step(1)

            }
            
        }
    }
}

class Onc_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    def fc_one_compute(data: Array[Int], weights: Array[Array[Int]], bias: Array[Int]): Array[Int] ={
        val height = data.length
        val classes = weights.length

        val result = Array.ofDim[Int](classes)
        for(h <- 0 until classes){
            var sum = 0
            for(oc <- 0 until height){
                sum = sum + data(oc) * weights(h)(oc)
            }
            result(h) = sum + bias(h)
        }
        return result
    }

    behavior of "test"
    it should "test conv Reg" in {
        test(new One_Class_Partial_Linear(5)){ c=> 
            val mini_length = 5
            val input_data = Array.ofDim[Int](5)
            for(i <- 0 until 5){
                input_data(i) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
            }
            
            val input_weight = Array.ofDim[Int](1, 5)
            for(l <- 0 until 1){
                for(i <- 0 until 5){
                        input_weight(l)(i) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                    
                }
            }
                
            val input_bias = Array.ofDim[Int](1)
            for(l <- 0 until 1){
                input_bias(l) = 0
            }

            val result = fc_one_compute(input_data, input_weight, input_bias)
            c.io.data_zero.poke(0.S)
            c.io.weight_zero.poke(0.S)
            
            for(oh <- 0 until 1){
                print(result(oh))
                print(" ")
                
            }
            println()

            for(loop <- 0 until 3){
                for(i <- 0 until 5){
                    c.io.data(i).poke(input_data(i).S)
                }
                for(i <- 0 until 5){
                    c.io.weights(i).poke(input_weight(0)(i).S)
                }
                print(c.io.result.peek())
                print(" ")

                println()
                println("-----------------")
                c.clock.step(1)

            }
            
        }
    }
}

class FFF_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    def fc_one_compute(data: Array[Int], weights: Array[Array[Int]], bias: Array[Int]): Array[Int] ={
        val height = data.length
        val classes = weights.length

        val result = Array.ofDim[Int](classes)
        for(h <- 0 until classes){
            var sum = 0
            for(oc <- 0 until height){
                sum = sum + data(oc) * weights(h)(oc)
            }
            result(h) = sum + bias(h)
        }
        return result
    }

    behavior of "test"
    it should "test conv Reg" in {
        test(new Fully_Connected_Layer(15, 5)){ c=> 
            val mini_length = 5
            val input_data = Array.ofDim[Int](15)
            for(i <- 0 until 15){
                input_data(i) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
            }
            
            val input_weight = Array.ofDim[Int](1, 15)
            for(l <- 0 until 1){
                for(i <- 0 until 15){
                        input_weight(l)(i) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                    
                }
            }
                
            val input_bias = Array.ofDim[Int](1)
            for(l <- 0 until 1){
                input_bias(l) = 0
            }

            val result = fc_one_compute(input_data, input_weight, input_bias)
            c.io.data_zero.poke(0.S)
            c.io.weight_zero.poke(0.S)
            c.io.result_zreo.poke(0.S)
            c.io.m_of_scale.poke(FixedPoint.fromDouble((1.0).toDouble, 64.W, 32.BP))
            for(oh <- 0 until 1){
                print(result(oh))
                print(" ")
                
            }
            println()
            val len_times = 15 / mini_length
            for(loop <- 0 until 8){
                print("-----loop")
                print(loop)
                println("-----")
                val len_index = loop % len_times
                for(i <- 0 until mini_length){
                    c.io.data(i).poke(input_data(len_index * mini_length + i).S)
                }
                for(i <- 0 until mini_length){
                    c.io.weights(i).poke(input_weight(0)(len_index * mini_length + i).S)
                }
                print(c.io.result.peek())
                print(" ")

                println()
                println("-----------------")
                c.clock.step(1)

            }
            
        }
    }
}


class Multy_Convolutional_Layer_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    def convolution_one_out(data: Array[Array[Array[Int]]], weights: Array[Array[Array[Array[Int]]]], bias: Array[Int], stride: Int): Array[Array[Int]] ={
        val width = data(0)(0).length
        val height = data(0).length
        val in_channel = data.length
        val out_channel = weights.length

        val kernel_size_row = weights(0)(0).length
        val kernel_size_col = weights(0)(0)(0).length

        var out_height = ((height - kernel_size_row) / stride).toInt + 1

        val result = Array.ofDim[Int](out_channel, out_height)
        for(h <- 0 until out_height){
            for(oc <- 0 until out_channel){
                var sum = 0
                for(ic <- 0 until in_channel){
                    for(kr <- 0 until kernel_size_row){
                        for(kc <- 0 until kernel_size_col){
                            sum = sum + data(ic)(kr + h * stride)(kc) * weights(oc)(ic)(kr)(kc)
                        }
                    }
                }
                result(oc)(h) = sum + bias(oc)
            }
        }
        return result
    }

    behavior of "test"
    it should "test conv Reg" in {
        test(new Multy_Columns_Convolutional_Layer(9, 4, 4, 2, 1, 3, 3, 1, 14, 2)){ c=> 

            

            val input_data = Array.ofDim[Int](4, 9, 1)
            for(i <- 0 until 4){
                for(j <- 0 until 9){
                    for(k <- 0 until 1){
                        input_data(i)(j)(k) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                        // input_data(i)(j)(k) = -1
                        // if(j > 2){
                        //     input_data(i)(j)(k) = 0
                        // }
                    }
                }
            }
            
            val input_weight = Array.ofDim[Int](4, 4, 3, 3)
            for(l <- 0 until 4){
                for(i <- 0 until 4){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            input_weight(l)(i)(j)(k) = 1
                        }
                    }
                }
            }
                
            val input_bias = Array.ofDim[Int](4)
            for(l <- 0 until 4){
                input_bias(l) = 1
            }


            c.io.bias_last_column.poke(1.S)
            c.io.data_zero.poke(0.S)
            c.io.weight_zero.poke(0.S)
            c.io.result_zreo.poke(0.S)
            c.io.m_of_scale.poke(FixedPoint.fromDouble((1.0).toDouble, 64.W, 32.BP))

            for(i <- 0 until 1){
                for(j <- 0 until 9){
                    for(k <- 0 until 4){
                        print(input_data(k)(j)(i))
                        print(" | ")
                    }
                    println()
                }
            }
            println("---------------------")

            var ls_count = 0
            var ps_count = 0
            for(loop <- 0 until 150){
                // last col
                val lll = (ls_count % 2) * 2
                val llz = ls_count / 2
                if(llz <= 9 - 3){
                    for(i <- 0 until 2){
                        for(j <- 0 until 3){
                            c.io.in_last_column(i)(j)(0).poke(input_data(lll + i)(llz + j)(0).S)
                        }
                    }
                    ls_count = ls_count + 1
                }
                else if(llz == 9 - 2){
                    ls_count = 0
                    val lll = (ls_count % 2) * 2
                    val llz = ls_count / 2
                    for(i <- 0 until 2){
                        for(j <- 0 until 3){
                            c.io.in_last_column(i)(j)(0).poke(input_data(lll + i)(llz + j)(0).S)
                        }
                    }
                    ls_count = ls_count + 1
                }
                // weight
                for(i <- 0 until 2){
                    for(j <- 0 until 3){
                        c.io.weights_last_column(i)(j)(0).poke(1.S)
                    }
                }
                c.io.partialsum_last_column.poke(0.S)

                // previous col
                val ppp = ps_count % 4
                val ppz = ps_count / 4
                if(ppz <= 9 - 3){
                    for(i <- 0 until 1){
                        for(j <- 0 until 3){
                            c.io.in_previous_columns(i)(j)(0).poke(input_data(ppp + i)(ppz + j)(0).S)
                        }
                    }
                    ps_count = ps_count + 1
                }
                else if(ppz == 9 - 2){
                    ps_count = 0
                    val ppp = ps_count % 4
                    val ppz = ps_count / 4
                    for(i <- 0 until 1){
                        for(j <- 0 until 3){
                            c.io.in_previous_columns(i)(j)(0).poke(input_data(ppp + i)(ppz + j)(0).S)
                        }
                    }
                    ps_count = ps_count + 1
                }
                
                // weight
                for(cc <- 0 until 2){
                    for(i <- 0 until 1){
                        for(j <- 0 until 3){
                            c.io.weights_previous_columns(cc)(i)(j)(0).poke(1.S)
                        }
                    }
                }
                for(i <- 0 until 2){
                    c.io.partialsum_previous_columns(i).poke(0.S)
                }
                
                println("clock time:", loop)
                print(c.io.out_last_column(0)(0)(0).peek())
                print(" | ")
                for(i <- 0 until 2){
                    print(c.io.out_previous_columns(i)(0)(0)(0).peek())
                    print(" <> ")
                }
                println()
                

                c.clock.step(1)
            }
            
        }
    }
}

class Base_Convolutional_Layer_Test extends FlatSpec with ChiselScalatestTester with Matchers{
    def convolution_one_out(data: Array[Array[Array[Int]]], weights: Array[Array[Array[Array[Int]]]], bias: Array[Int], stride: Int): Array[Array[Int]] ={
        val width = data(0)(0).length
        val height = data(0).length
        val in_channel = data.length
        val out_channel = weights.length

        val kernel_size_row = weights(0)(0).length
        val kernel_size_col = weights(0)(0)(0).length

        var out_height = ((height - kernel_size_row) / stride).toInt + 1

        val result = Array.ofDim[Int](out_channel, out_height)
        for(h <- 0 until out_height){
            for(oc <- 0 until out_channel){
                var sum = 0
                for(ic <- 0 until in_channel){
                    for(kr <- 0 until kernel_size_row){
                        for(kc <- 0 until kernel_size_col){
                            sum = sum + data(ic)(kr + h * stride)(kc) * weights(oc)(ic)(kr)(kc)
                        }
                    }
                }
                result(oc)(h) = sum + bias(oc)
            }
        }
        return result
    }

    behavior of "test"
    it should "test conv Reg" in {
        test(new Base_Convolutional_Layer(9, 4, 8, 2, 3, 3, 1, 70)){ c=> 

            // data
            val input_data = Array.ofDim[Int](8, 9, 3)
            for(i <- 0 until 8){
                for(j <- 0 until 9){
                    for(k <- 0 until 3){
                        input_data(i)(j)(k) = Random.nextInt(3) * pow((-1), Random.nextInt(20)).toInt
                    }
                }
            }
            // weights
            val input_weight = Array.ofDim[Int](4, 8, 3, 3)
            for(l <- 0 until 4){
                for(i <- 0 until 8){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            input_weight(l)(i)(j)(k) = 1
                        }
                    }
                }
            }
            // bias
            val input_bias = Array.ofDim[Int](4)
            for(l <- 0 until 4){
                input_bias(l) = 1
            }

            val result = convolution_one_out(input_data, input_weight, input_bias, 1)
            c.io.bias.poke(1.S)
            c.io.data_zero.poke(0.S)
            c.io.weight_zero.poke(0.S)
            c.io.result_zreo.poke(0.S)
            c.io.m_of_scale.poke(FixedPoint.fromDouble((1.0).toDouble, 64.W, 32.BP))
            
            for(oh <- 0 until 7){
                for(oc <- 0 until 4){
                    print(result(oc)(oh))
                    print(" ")
                }
                println()
            }
            println("_-_-_-_-_")

            var ls_count = 0
            for(loop <- 0 until 150){
                 // last col
                val lll = (ls_count % 4) * 2
                val llz = ls_count / 4
                if(llz <= 9 - 3){
                    for(i <- 0 until 2){
                        for(j <- 0 until 3){
                            for(k <- 0 until 3){
                                c.io.in_data(i)(j)(k).poke(input_data(lll + i)(llz + j)(k).S)
                            }
                        }
                    }
                    ls_count = ls_count + 1
                }
                else if(llz == 9 - 2){
                    ls_count = 0
                    val lll = (ls_count % 4) * 2
                    val llz = ls_count / 4
                    for(i <- 0 until 2){
                        for(j <- 0 until 3){
                            for(k <- 0 until 3){
                                c.io.in_data(i)(j)(k).poke(input_data(lll + i)(llz + j)(k).S)
                            }
                        }
                    }
                    ls_count = ls_count + 1
                }
                // weight
                for(i <- 0 until 2){
                    for(j <- 0 until 3){
                        for(k <- 0 until 3){
                            c.io.weights(i)(j)(k).poke(1.S)
                        }
                    }
                }
                
                println("clock time:", loop)
                print(c.io.out_data(0)(0)(0).peek())
                println()
                
                c.clock.step(1)
            }
            
        }
    }
}

