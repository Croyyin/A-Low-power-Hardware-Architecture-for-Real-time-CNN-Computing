package CNN

import chisel3._
import chisel3.experimental._
import Array._

// 0 cycle
class One_Class_Partial_Linear(length: Int) extends Module with DataConfig{
    val io = IO(new Bundle{
        val data = Input(Vec(length, SInt(QuantizationWidth.W)))
        val weights = Input(Vec(length, SInt(QuantizationWidth.W)))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result = Output(SInt(DataWidth.W))
    })

    var i = 0
    var sum = 0.S
    for(i <- 0 until length){
        sum = sum + PartialQMultiplier(io.data(i), io.data_zero, io.weights(i), io.weight_zero)
    }
    io.result := sum
}
