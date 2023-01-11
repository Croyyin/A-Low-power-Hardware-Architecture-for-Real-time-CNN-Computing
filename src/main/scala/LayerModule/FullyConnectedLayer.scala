package CNN

import chisel3._
import chisel3.util._
import chisel3.experimental._
import scala.math.floor

class Fully_Connected_Layer(length: Int, mini_lenght: Int) extends Module with DataConfig{

    val io = IO(new Bundle{
        val data = Input(Vec(mini_lenght, SInt(QuantizationWidth.W)))
        val weights = Input(Vec(mini_lenght, SInt(QuantizationWidth.W)))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val be_zero = Input(Bool())
        val result = Output(SInt(QuantizationWidth.W))
    })
    val m_c_k_num = length / mini_lenght

    // convolution
    val mini_fc = Module(new One_Class_Partial_Linear(mini_lenght))
    // adder
    val adder = Module(new Batch_Adder(m_c_k_num))
    // bias adder
    val bias_adder = Module(new Bias_Adder)
    // register for adder
    val middle_reg = RegInit(0.S(DataWidth.W))


    mini_fc.io.data := io.data
    mini_fc.io.weights := io.weights
    mini_fc.io.data_zero := io.data_zero
    mini_fc.io.weight_zero := io.weight_zero

    adder.io.current_input := mini_fc.io.result
    adder.io.previous_input := middle_reg
    adder.io.be_zero := io.be_zero
    middle_reg := adder.io.result

    bias_adder.io.bias := io.bias
    bias_adder.io.data := adder.io.result
    bias_adder.io.result_zreo := io.result_zreo
    bias_adder.io.m_of_scale := io.m_of_scale

    io.result := bias_adder.io.result
}

class FC_Layer_Reg(classes: Int, length: Int, mini_lenght: Int, unified_cycle: Int) extends Module with DataConfig{

    val len_times = (length / mini_lenght).toInt - 1
    val current_cycle = (len_times + 1) * classes 
    val io = IO(new Bundle{
        val data = Input(Vec(mini_lenght, SInt(QuantizationWidth.W)))
        val weights = Input(Vec(mini_lenght, SInt(QuantizationWidth.W)))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val reg_out = Output(Vec(classes, SInt(QuantizationWidth.W)))
    })
    val counter_1 = Module(new Diy_Counter(len_times))
    val clc_counter = Module(new Diy_Counter(unified_cycle, 1))
    
    val reg_after = Module(new One_Shift_Reg_Array(QuantizationWidth, 1, classes))

    val fc_l = Module(new Fully_Connected_Layer(length, mini_lenght))

    clc_counter.io.en := true.B
    for(i <- 0 until classes){
        io.reg_out(i) := reg_after.io.out_data(0)(i)(0)
    }
    
    when(clc_counter.io.number === 0.U){
        // init
        counter_1.io.en := true.B
        fc_l.io.be_zero := false.B
        reg_after.io.signal := 0.U
    }.elsewhen(clc_counter.io.number <= current_cycle.U){
        fc_l.io.be_zero := false.B
        counter_1.io.en := true.B
        when(counter_1.io.number === 0.U){
            reg_after.io.signal := 1.U
        }otherwise{
            reg_after.io.signal := 0.U
        }
    }.elsewhen(clc_counter.io.number > current_cycle.U && clc_counter.io.number < unified_cycle.U){
        fc_l.io.be_zero := false.B
        reg_after.io.signal := 0.U
        counter_1.io.en := false.B
    }otherwise{
        fc_l.io.be_zero := true.B
        counter_1.io.en := true.B
        reg_after.io.signal := 0.U
    }

    
    fc_l.io.data := io.data
    fc_l.io.weights := io.weights
    fc_l.io.bias := io.bias
    fc_l.io.data_zero := io.data_zero
    fc_l.io.weight_zero := io.weight_zero
    fc_l.io.result_zreo := io.result_zreo
    fc_l.io.m_of_scale := io.m_of_scale

    reg_after.io.in_data := fc_l.io.result
}

class Mini_FC_Layer(classes: Int, length: Int, mini_lenght: Int, unified_cycle: Int) extends Module with DataConfig{

    val len_times = (length / mini_lenght).toInt - 1
    val current_cycle = (len_times + 1) * classes 
    val io = IO(new Bundle{
        val data = Input(Vec(mini_lenght, SInt(QuantizationWidth.W)))
        val weights = Input(Vec(mini_lenght, SInt(QuantizationWidth.W)))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val reg_out = Output(Vec(1, SInt(QuantizationWidth.W)))
    })
    val counter_1 = Module(new Diy_Counter(len_times))
    val clc_counter = Module(new Diy_Counter(unified_cycle, 1))
    
    val reg_before = Module(new Partial_Shift_Reg_Array(QuantizationWidth, 1, 1, mini_lenght, 1, 1, 1))

    val fc_l = Module(new Fully_Connected_Layer(length, mini_lenght))
    reg_before.io.signal := 0.U
    reg_before.io.in_data(0)(0) := io.data


    clc_counter.io.en := true.B

    
    when(clc_counter.io.number === 0.U){
        // init
        counter_1.io.en := true.B
        fc_l.io.be_zero := false.B
    }.elsewhen(clc_counter.io.number <= current_cycle.U){
        fc_l.io.be_zero := false.B
        counter_1.io.en := true.B
    }.elsewhen(clc_counter.io.number > current_cycle.U && clc_counter.io.number < unified_cycle.U){
        fc_l.io.be_zero := false.B
        counter_1.io.en := false.B
    }otherwise{
        fc_l.io.be_zero := true.B
        counter_1.io.en := true.B
    }

    
    fc_l.io.data := reg_before.io.out_data(0)(0)
    fc_l.io.weights := io.weights
    fc_l.io.bias := io.bias
    fc_l.io.data_zero := io.data_zero
    fc_l.io.weight_zero := io.weight_zero
    fc_l.io.result_zreo := io.result_zreo
    fc_l.io.m_of_scale := io.m_of_scale
    io.reg_out(0) := fc_l.io.result
}

class Mini_FC_Layer_ReLU(classes: Int, length: Int, mini_lenght: Int, unified_cycle: Int) extends Module with DataConfig{

    val len_times = (length / mini_lenght).toInt - 1
    val current_cycle = (len_times + 1) * classes 
    val io = IO(new Bundle{
        val data = Input(Vec(mini_lenght, SInt(QuantizationWidth.W)))
        val weights = Input(Vec(mini_lenght, SInt(QuantizationWidth.W)))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val reg_out = Output(Vec(1, SInt(QuantizationWidth.W)))
    })
    val counter_1 = Module(new Diy_Counter(len_times))
    val clc_counter = Module(new Diy_Counter(unified_cycle, 1))
    val relu = Module(new ReLU)
    val reg_before = Module(new Partial_Shift_Reg_Array(QuantizationWidth, 1, 1, mini_lenght, 1, 1, 1))

    val fc_l = Module(new Fully_Connected_Layer(length, mini_lenght))
    reg_before.io.signal := 0.U
    reg_before.io.in_data(0)(0) := io.data


    clc_counter.io.en := true.B

    
    when(clc_counter.io.number === 0.U){
        // init
        counter_1.io.en := true.B
        fc_l.io.be_zero := false.B
    }.elsewhen(clc_counter.io.number <= current_cycle.U){
        fc_l.io.be_zero := false.B
        counter_1.io.en := true.B
    }.elsewhen(clc_counter.io.number > current_cycle.U && clc_counter.io.number < unified_cycle.U){
        fc_l.io.be_zero := false.B
        counter_1.io.en := false.B
    }otherwise{
        fc_l.io.be_zero := true.B
        counter_1.io.en := true.B
    }

    
    fc_l.io.data := reg_before.io.out_data(0)(0)
    fc_l.io.weights := io.weights
    fc_l.io.bias := io.bias
    fc_l.io.data_zero := io.data_zero
    fc_l.io.weight_zero := io.weight_zero
    fc_l.io.result_zreo := io.result_zreo
    fc_l.io.m_of_scale := io.m_of_scale

    relu.io.data := fc_l.io.result
    relu.io.zero_point := io.result_zreo
    io.reg_out(0) := relu.io.result

}