package CNN

import chisel3._
import chisel3.util._
import chisel3.experimental._
import scala.math.floor

class Pooling_layer(is_max: Boolean, one_in_channel_num: Int, pooling_kernel_row: Int, pooling_kernel_col: Int) extends Module with DataConfig{

    val io = IO(new Bundle{
        val data = Input(Vec(one_in_channel_num,  Vec(pooling_kernel_row, Vec(pooling_kernel_col, SInt(QuantizationWidth.W)))))
        val result = Output(Vec(one_in_channel_num, SInt(QuantizationWidth.W)))
    })

    if(is_max){
        val pooling_computers = VecInit(Seq.fill(one_in_channel_num)(Module(new Max_Pooling(pooling_kernel_row, pooling_kernel_col)).io))
        for(i <- 0 until one_in_channel_num){
            pooling_computers(i).data := io.data(i)
            io.result(i) := pooling_computers(i).result
        }
    }else{
        val pooling_computers = VecInit(Seq.fill(one_in_channel_num)(Module(new Average_Pooling(pooling_kernel_row, pooling_kernel_col)).io))
        for(i <- 0 until one_in_channel_num){
            pooling_computers(i).data := io.data(i)
            io.result(i) := pooling_computers(i).result
        }
    }
}

