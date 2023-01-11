package CNN

import chisel3._
import chisel3.experimental._

// 0 cycle
class Kernel_Compute_Unit(kernel_size_row: Int, kernel_size_col: Int) extends Module with DataConfig{
    val io = IO(new Bundle{
        val data = Input(Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W))))
        val weights = Input(Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W))))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result = Output(SInt(DataWidth.W))
    })

    // 乘法器与部分和阵列
    val partial_sum_wires = Wire(Vec(kernel_size_row * kernel_size_col + 1, SInt(DataWidth.W)))
    partial_sum_wires(0) := 0.S

    var i = 0
    for(i <- 0 until kernel_size_row){
        var j = 0
        for(j <- 0 until kernel_size_col){
            partial_sum_wires(i * kernel_size_col + j + 1) := partial_sum_wires(i * kernel_size_col + j) + PartialQMultiplier(io.data(i)(j), io.data_zero, io.weights(i)(j), io.weight_zero)
        }
    }
    io.result := partial_sum_wires(kernel_size_row * kernel_size_col)
}

// 0 cycle
class Mini_Convolution_Kernel(in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int) extends Module with DataConfig{
    val io = IO(new Bundle{
        val data = Input(Vec(in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result = Output(SInt(DataWidth.W))
    })

    // 卷积核与部分和阵列
    val kernel_computers = VecInit(Seq.fill(in_channel_num)(Module(new Kernel_Compute_Unit(kernel_size_row, kernel_size_col)).io))
    val partial_sum_wires = Wire(Vec(in_channel_num + 1, SInt(DataWidth.W)))
    partial_sum_wires(0) := 0.S

    var i = 0
    for(i <- 0 until in_channel_num){
        var j = 0
        for(j <- 0 until kernel_size_row){
            var k = 0
            for(k <- 0 until kernel_size_col){
                kernel_computers(i).weights(j)(k) := io.weights(i)(j)(k)
                kernel_computers(i).data(j)(k) := io.data(i)(j)(k)
                kernel_computers(i).data_zero := io.data_zero
                kernel_computers(i).weight_zero := io.weight_zero
            }
        } 
        partial_sum_wires(i + 1) := partial_sum_wires(i) + kernel_computers(i).result
    }
    io.result := partial_sum_wires(in_channel_num)

}

class Batch_Adder(m_c_k_num: Int, delay: Boolean=true) extends Module with DataConfig{
    val io =IO(new Bundle{
        val current_input = Input(SInt(DataWidth.W))
        val previous_input = Input(SInt(DataWidth.W))
        val be_zero = Input(Bool())
        val result = Output(SInt(DataWidth.W))

    })
    var defulat = m_c_k_num
    if(!delay){
        defulat = defulat - 1
    }
    val adder_result = Wire(SInt(DataWidth.W))
    adder_result := io.current_input
    
    val counter = RegInit(defulat.U(DataWidth.W))

    
    when(counter =/= (m_c_k_num - 1).U){
        adder_result := io.current_input + io.previous_input
    }otherwise{
        adder_result := io.current_input
    }
    when(counter =/= 0.U){
        counter := counter - 1.U
    }otherwise{
        counter := (m_c_k_num - 1).U
    }
    
    when(io.be_zero === true.B){
        counter := (m_c_k_num - 1).U
    }
    
    io.result := adder_result
}

class Bias_Adder extends Module with DataConfig{
    val io =IO(new Bundle{
        val bias = Input(SInt(DataWidth.W))
        val data = Input(SInt(DataWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val result = Output(SInt(QuantizationWidth.W))
    })

    io.result := PartialQResult(io.data, io.bias, io.m_of_scale, io.result_zreo)
}











