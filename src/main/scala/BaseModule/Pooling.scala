package CNN

import chisel3._
import chisel3.experimental._

// 0 cycle
// MaxPooling



class Max_Pooling(pooling_kernel_row: Int, pooling_kernel_col: Int) extends Module with DataConfig{
    val io = IO(new Bundle{
        val data = Input(Vec(pooling_kernel_row, Vec(pooling_kernel_col, SInt(QuantizationWidth.W))))
        val result = Output(SInt(QuantizationWidth.W))
    })

    var max = io.data(0)(0)
    for(i <- 0 until pooling_kernel_row){
        for(j <- 0 until pooling_kernel_col){
            max = Mux(max > io.data(i)(j), max, io.data(i)(j))
        }
    }

    io.result := max
}

// 0 cycle
// AveragePooling
class Average_Pooling(pooling_kernel_row: Int, pooling_kernel_col: Int) extends Module with DataConfig{
    val io = IO(new Bundle{
        val data = Input(Vec(pooling_kernel_row, Vec(pooling_kernel_col, SInt(QuantizationWidth.W))))
        val result = Output(SInt(QuantizationWidth.W))
    })

    val partial_sum_wires = Wire(Vec(pooling_kernel_row *  pooling_kernel_col + 1, SInt(DataWidth.W)))
    partial_sum_wires(0) := 0.S
    for(i <- 0 until pooling_kernel_row){
        for(j <- 0 until pooling_kernel_col){
            partial_sum_wires(i * pooling_kernel_col + j + 1) := partial_sum_wires(i * pooling_kernel_col + j) + io.data(i)(j)
        }
    }
    io.result := partial_sum_wires(pooling_kernel_row * pooling_kernel_col)
}







