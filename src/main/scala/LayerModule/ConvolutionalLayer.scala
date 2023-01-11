package CNN

import chisel3._
import chisel3.util._
import chisel3.experimental._
import scala.math.floor
import scala.math.ceil

class Convolutional_Layer(all_in_channel_num: Int, one_in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int, delay: Boolean=true) extends Module with DataConfig{

    val io = IO(new Bundle{
        val data = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val be_zero = Input(Bool())
        val result = Output(SInt(QuantizationWidth.W))
    })
    val m_c_k_num = all_in_channel_num / one_in_channel_num

    // convolution
    val mini_convolution = Module(new Mini_Convolution_Kernel(one_in_channel_num, kernel_size_row, kernel_size_col))
    // adder
    val adder = Module(new Batch_Adder(m_c_k_num, delay))
    // bias adder
    val bias_adder = Module(new Bias_Adder)
    // relu
    val relu = Module(new ReLU)
    // register for adder
    val middle_reg = RegInit(0.S(DataWidth.W))


    mini_convolution.io.data := io.data
    mini_convolution.io.weights := io.weights
    mini_convolution.io.data_zero := io.data_zero
    mini_convolution.io.weight_zero := io.weight_zero

    adder.io.current_input := mini_convolution.io.result
    adder.io.previous_input := middle_reg
    adder.io.be_zero := io.be_zero

    middle_reg := adder.io.result

    bias_adder.io.bias := io.bias
    bias_adder.io.data := adder.io.result
    bias_adder.io.result_zreo := io.result_zreo
    bias_adder.io.m_of_scale := io.m_of_scale
    

    relu.io.data := bias_adder.io.result
    relu.io.zero_point := io.result_zreo
    io.result := relu.io.result
}

class Partial_Convolutional_Layer(all_in_channel_num: Int, one_in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int, delay: Boolean=true) extends Module with DataConfig{

    val io = IO(new Bundle{
        val data = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val be_zero = Input(Bool())
        val result = Output(SInt(QuantizationWidth.W))
    })
    val m_c_k_num = ceil(all_in_channel_num / one_in_channel_num).toInt

    // convolution
    val mini_convolution = Module(new Mini_Convolution_Kernel(one_in_channel_num, kernel_size_row, kernel_size_col))
    // adder
    val adder = Module(new Batch_Adder(m_c_k_num, delay))
    // register for adder
    val middle_reg = RegInit(0.S(DataWidth.W))


    mini_convolution.io.data := io.data
    mini_convolution.io.weights := io.weights
    mini_convolution.io.data_zero := io.data_zero
    mini_convolution.io.weight_zero := io.weight_zero

    adder.io.current_input := mini_convolution.io.result
    adder.io.previous_input := middle_reg
    adder.io.be_zero := io.be_zero

    middle_reg := adder.io.result
    io.result := adder.io.result
}

class Reg_Convolutional_Layer_Reg(in_height: Int, out_channel_num: Int, all_in_channel_num: Int, one_in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int, stride_row: Int, unified_cycle: Int) extends Module with DataConfig{

    val out_height = ((in_height - kernel_size_row) / stride_row).toInt + 1
    val channel_times = (all_in_channel_num / one_in_channel_num).toInt * out_channel_num - 1
    val row_times = (in_height - kernel_size_row) / stride_row + 1 - 1

    val current_cycle = (channel_times + 1) * (row_times + 1)
    val io = IO(new Bundle{
        val reg_in = Input(Vec(all_in_channel_num, Vec(in_height, Vec(1, SInt(QuantizationWidth.W)))))
        val data = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col - 1, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val reg_out = Output(Vec(out_channel_num, Vec(out_height, Vec(1, SInt(QuantizationWidth.W)))))
    })

    // counter registers definition
    val counter_1  = Module(new Diy_Counter(channel_times))
    val counter_2  = Module(new Diy_Counter(row_times))
    val counter_3 = Module(new Diy_Counter((all_in_channel_num / one_in_channel_num).toInt - 1))
    val clc_counter = Module(new Diy_Counter(unified_cycle, 1))

    val reg_before = Module(new Partial_Shift_Reg_Array(QuantizationWidth, all_in_channel_num, in_height, 1, one_in_channel_num, kernel_size_row, stride_row))
    val reg_after = Module(new One_Shift_Reg_Array(QuantizationWidth, out_channel_num, out_height))
    io.reg_out := reg_after.io.out_data

    val c_l = Module(new Convolutional_Layer(all_in_channel_num, one_in_channel_num, kernel_size_row, kernel_size_col))

    clc_counter.io.en := true.B
    
    when(clc_counter.io.number === 0.U){
        // counter init
        // counter_1 is a state counter for channel
        counter_1.io.en := true.B
        // counter_2 is a state counter for out-row
        counter_2.io.en := false.B
        // counter_3 is a state counter for result shift
        counter_3.io.en := true.B
        // registers init
        reg_before.io.signal := 0.U
        reg_after.io.signal := 0.U

        c_l.io.be_zero := false.B
    }.elsewhen(clc_counter.io.number <= current_cycle.U){
        c_l.io.be_zero := false.B
        when(counter_1.io.number =/= 0.U && counter_1.io.number =/= channel_times.U){
            reg_before.io.signal := 1.U
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := false.B
        }.elsewhen(counter_1.io.number === channel_times.U){
            reg_before.io.signal := 1.U
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := true.B
        }.elsewhen(counter_1.io.number === 0.U && counter_2.io.number =/= 0.U){
            reg_before.io.signal := 2.U
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := false.B
        }otherwise{
            reg_before.io.signal := 1.U
            counter_1.io.en := false.B
            counter_2.io.en := false.B
            counter_3.io.en := false.B
        }

        when(counter_3.io.number === 0.U){
            reg_after.io.signal := 1.U
        }otherwise{
            reg_after.io.signal := 0.U
        }
    }.elsewhen(clc_counter.io.number > current_cycle.U && clc_counter.io.number < unified_cycle.U){
        c_l.io.be_zero := false.B
        reg_before.io.signal := 1.U
        reg_after.io.signal := 0.U
        counter_1.io.en := false.B
        counter_2.io.en := false.B
        counter_3.io.en := false.B
    }otherwise{
        c_l.io.be_zero := true.B
        reg_before.io.signal := 0.U
        reg_after.io.signal := 0.U
        counter_1.io.en := true.B
        counter_3.io.en := true.B
        counter_2.io.en := false.B
    }


    

    for(i <- 0 until one_in_channel_num){
        for(j <- 0 until kernel_size_row){
            for(k <- 0 until kernel_size_col - 1){
                c_l.io.data(i)(j)(k) := io.data(i)(j)(k)
                c_l.io.data(i)(j)(kernel_size_col - 1) := reg_before.io.out_data(i)(j)(0)
            }
        }
    }

    c_l.io.weights := io.weights
    c_l.io.bias := io.bias
    c_l.io.data_zero := io.data_zero
    c_l.io.weight_zero := io.weight_zero
    c_l.io.result_zreo := io.result_zreo
    c_l.io.m_of_scale := io.m_of_scale

    reg_before.io.in_data := io.reg_in
    reg_after.io.in_data := c_l.io.result
}


class Reg_Convolutional_Layer_Reg_Pooling_layer_Reg(in_height: Int, out_channel_num: Int, all_in_channel_num: Int, one_in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int, stride_row: Int, pooling_kernel_row: Int, pooling_kernel_col: Int, pool_stride_row: Int, is_max: Boolean, unified_cycle: Int) extends Module with DataConfig{

    val out_height = ((in_height - kernel_size_row) / stride_row).toInt + 1
    val channel_times = (all_in_channel_num / one_in_channel_num).toInt * out_channel_num - 1
    val row_times = (in_height - kernel_size_row) / stride_row + 1 - 1
    val out_pool = ((out_height - pooling_kernel_row) / pool_stride_row).toInt + 1
    val pool_wait = pool_stride_row - 1
    val pool_cycle = (channel_times + 1) * (((in_height - kernel_size_row) / stride_row).toInt * stride_row + kernel_size_row)

    val current_cycle = (channel_times + 1) * (row_times + 1)
    val io = IO(new Bundle{
        val reg_in = Input(Vec(all_in_channel_num, Vec(in_height, Vec(1, SInt(QuantizationWidth.W)))))
        val data = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col - 1, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))

        val pool_data = Input(Vec(1,  Vec(pooling_kernel_row, Vec(pooling_kernel_col - 1, SInt(QuantizationWidth.W)))))
        val reg_out = Output(Vec(out_channel_num, Vec(out_pool, Vec(1, SInt(QuantizationWidth.W)))))
        val c4 = Output(UInt(DataWidth.W))
    })
    val counter_1  = Module(new Diy_Counter(channel_times))
    val counter_2  = Module(new Diy_Counter(row_times))
    val counter_3 = Module(new Diy_Counter((all_in_channel_num / one_in_channel_num).toInt - 1))
    val counter_4 = Module(new Diy_Counter(pool_wait))
    val is_run = RegInit(false.B)
    val clc_counter = Module(new Diy_Counter(unified_cycle, 1))
    
    val reg_before = Module(new Partial_Shift_Reg_Array(QuantizationWidth, all_in_channel_num, in_height, 1, one_in_channel_num, kernel_size_row, stride_row))
    val reg_after = Module(new One_Shift_Reg_Array(QuantizationWidth, out_channel_num, pooling_kernel_row))
    val reg_pool = Module(new One_Shift_Reg_Array(QuantizationWidth, out_channel_num, out_pool))

    val c_l = Module(new Convolutional_Layer(all_in_channel_num, one_in_channel_num, kernel_size_row, kernel_size_col))
    val p_l = Module(new Pooling_layer(is_max, 1, pooling_kernel_row, pooling_kernel_col))

    io.c4 := counter_4.io.number

    clc_counter.io.en := true.B

    when(clc_counter.io.number === 0.U){
        // counter init
        // counter_1 is a state counter for channel
        counter_1.io.en := true.B
        // counter_2 is a state counter for out-row
        counter_2.io.en := false.B
        // counter_3 is a state counter for result shift
        counter_3.io.en := true.B
        //
        counter_4.io.en := false.B
        // registers init
        reg_before.io.signal := 0.U
        reg_after.io.signal := 0.U
        reg_pool.io.signal := 0.U
        c_l.io.be_zero := false.B
    }.elsewhen(clc_counter.io.number <= current_cycle.U){
        c_l.io.be_zero := false.B
        when(counter_1.io.number =/= 0.U && counter_1.io.number =/= channel_times.U){
            reg_before.io.signal := 1.U
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := false.B
            counter_4.io.en := false.B
        }.elsewhen(counter_1.io.number === channel_times.U){
            reg_before.io.signal := 1.U
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := true.B
            counter_4.io.en := false.B
        }.elsewhen(counter_1.io.number === 0.U && counter_2.io.number =/= 0.U){
            reg_before.io.signal := 2.U
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := false.B
            when(clc_counter.io.number <= pool_cycle.U){
                counter_4.io.en := true.B
            }otherwise{
                counter_4.io.en := false.B
            }
            
        }otherwise{
            reg_before.io.signal := 1.U
            counter_1.io.en := false.B
            counter_2.io.en := false.B
            counter_3.io.en := false.B
            counter_4.io.en := false.B
        }

        when(counter_3.io.number === 0.U){
            reg_after.io.signal := 1.U
            when(counter_4.io.number === pool_wait.U){
                is_run := true.B
            }otherwise{
                is_run := false.B
            }
        }otherwise{
            reg_after.io.signal := 0.U
        }

        when(is_run === true.B){
            reg_pool.io.signal := 1.U
            is_run := false.B
        }otherwise{
            reg_pool.io.signal := 0.U
        }

    }.elsewhen(clc_counter.io.number > current_cycle.U && clc_counter.io.number < unified_cycle.U){
        c_l.io.be_zero := false.B
        reg_before.io.signal := 1.U
        reg_after.io.signal := 0.U
        counter_1.io.en := false.B
        counter_2.io.en := false.B
        counter_3.io.en := false.B
        counter_4.io.en := false.B

        when(is_run === true.B){
            reg_pool.io.signal := 1.U
            is_run := false.B
        }otherwise{
            reg_pool.io.signal := 0.U
        }
    }otherwise{
        c_l.io.be_zero := true.B
        reg_before.io.signal := 0.U
        reg_after.io.signal := 0.U
        reg_pool.io.signal := 0.U
        counter_1.io.en := true.B
        counter_3.io.en := true.B
        counter_2.io.en := false.B
        counter_4.io.en := true.B
    }
    

    for(i <- 0 until one_in_channel_num){
        for(j <- 0 until kernel_size_row){
            for(k <- 0 until kernel_size_col - 1){
                c_l.io.data(i)(j)(k) := io.data(i)(j)(k)
                c_l.io.data(i)(j)(kernel_size_col - 1) := reg_before.io.out_data(i)(j)(0)
            }
        }
    }

    
    for(j <- 0 until pooling_kernel_row){
        for(k <- 0 until pooling_kernel_col - 1){
            p_l.io.data(0)(j)(k) := io.pool_data(0)(j)(k)
            p_l.io.data(0)(j)(pooling_kernel_col - 1) := reg_after.io.out_data(out_channel_num - 1)(j)(0)
        }
    }



    c_l.io.weights := io.weights
    c_l.io.bias := io.bias
    c_l.io.data_zero := io.data_zero
    c_l.io.weight_zero := io.weight_zero
    c_l.io.result_zreo := io.result_zreo
    c_l.io.m_of_scale := io.m_of_scale

    reg_before.io.in_data := io.reg_in
    reg_after.io.in_data := c_l.io.result
    reg_pool.io.in_data := p_l.io.result(0)
    io.reg_out := reg_pool.io.out_data
}

class Mini_Convolutional_Layer(in_height: Int, out_channel_num: Int, all_in_channel_num: Int, one_in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int, stride_row: Int, unified_cycle: Int) extends Module with DataConfig{

    val out_height = ((in_height - kernel_size_row) / stride_row).toInt + 1
    val channel_times = (all_in_channel_num / one_in_channel_num).toInt * out_channel_num - 1
    val row_times = (in_height - kernel_size_row) / stride_row + 1 - 1

    val current_cycle = (channel_times + 1) * (row_times + 1)
    val io = IO(new Bundle{
        val reg_in = Input(Vec(one_in_channel_num, Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val reg_out = Output(Vec(1, Vec(1, Vec(1, SInt(QuantizationWidth.W)))))
    })

    // counter registers definition
    val counter_1  = Module(new Diy_Counter(channel_times))
    val counter_2  = Module(new Diy_Counter(row_times))
    val counter_3 = Module(new Diy_Counter((all_in_channel_num / one_in_channel_num).toInt - 1))
    val clc_counter = Module(new Diy_Counter(unified_cycle, 1))

    val reg_before = Module(new Partial_Shift_Reg_Array(QuantizationWidth, one_in_channel_num, kernel_size_row, kernel_size_col, one_in_channel_num, kernel_size_row, stride_row))
    val reg_after = Module(new One_Shift_Reg_Array(QuantizationWidth, 1, 1))
    io.reg_out := reg_after.io.out_data

    val c_l = Module(new Convolutional_Layer(all_in_channel_num, one_in_channel_num, kernel_size_row, kernel_size_col))

    // cycle control
    clc_counter.io.en := true.B
    reg_before.io.signal := 0.U
    when(clc_counter.io.number === 0.U){
        // counter init
        // counter_1 is a state counter for channel
        counter_1.io.en := true.B
        // counter_2 is a state counter for out-row
        counter_2.io.en := false.B
        // counter_3 is a state counter for result shift
        counter_3.io.en := true.B
        // registers init
        reg_after.io.signal := 0.U

        c_l.io.be_zero := false.B
    }.elsewhen(clc_counter.io.number <= current_cycle.U){
        c_l.io.be_zero := false.B
        when(counter_1.io.number =/= 0.U && counter_1.io.number =/= channel_times.U){
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := false.B
        }.elsewhen(counter_1.io.number === channel_times.U){
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := true.B
        }.elsewhen(counter_1.io.number === 0.U && counter_2.io.number =/= 0.U){
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := false.B
        }otherwise{
            counter_1.io.en := false.B
            counter_2.io.en := false.B
            counter_3.io.en := false.B
        }

        when(counter_3.io.number === 0.U){
            reg_after.io.signal := 1.U
        }otherwise{
            reg_after.io.signal := 0.U
        }
    }.elsewhen(clc_counter.io.number > current_cycle.U && clc_counter.io.number < unified_cycle.U){
        c_l.io.be_zero := false.B
        reg_after.io.signal := 0.U
        counter_1.io.en := false.B
        counter_2.io.en := false.B
        counter_3.io.en := false.B
    }otherwise{
        c_l.io.be_zero := true.B
        reg_after.io.signal := 0.U
        counter_1.io.en := true.B
        counter_3.io.en := true.B
        counter_2.io.en := false.B
    }
    // port link
    c_l.io.data := reg_before.io.out_data
    c_l.io.weights := io.weights
    c_l.io.bias := io.bias
    c_l.io.data_zero := io.data_zero
    c_l.io.weight_zero := io.weight_zero
    c_l.io.result_zreo := io.result_zreo
    c_l.io.m_of_scale := io.m_of_scale

    reg_before.io.in_data := io.reg_in
    reg_after.io.in_data := c_l.io.result
}

class Mini_Convolutional_Pooling_layer(in_height: Int, out_channel_num: Int, all_in_channel_num: Int, one_in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int, stride_row: Int, pooling_kernel_row: Int, pooling_kernel_col: Int, pool_stride_row: Int, is_max: Boolean, unified_cycle: Int) extends Module with DataConfig{

    val out_height = ((in_height - kernel_size_row) / stride_row).toInt + 1
    val channel_times = (all_in_channel_num / one_in_channel_num).toInt * out_channel_num - 1
    val row_times = (in_height - kernel_size_row) / stride_row + 1 - 1
    val out_pool = ((out_height - pooling_kernel_row) / pool_stride_row).toInt + 1
    val pool_wait = pool_stride_row - 1
    val pool_cycle = (channel_times + 1) * (((in_height - kernel_size_row) / stride_row).toInt * stride_row + kernel_size_row)

    val current_cycle = (channel_times + 1) * (row_times + 1)
    val io = IO(new Bundle{
        val reg_in = Input(Vec(one_in_channel_num, Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val bias = Input(SInt(DataWidth.W))
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))

        val pool_data = Input(Vec(pooling_kernel_row * pooling_kernel_col - 1,  SInt(QuantizationWidth.W)))
        val reg_out = Output(Vec(1, Vec(1, Vec(1, SInt(QuantizationWidth.W)))))
    })
    val counter_1  = Module(new Diy_Counter(channel_times))
    val counter_2  = Module(new Diy_Counter(row_times))
    val counter_3 = Module(new Diy_Counter((all_in_channel_num / one_in_channel_num).toInt - 1))
    val counter_4 = Module(new Diy_Counter(pool_wait))
    val is_run = RegInit(false.B)
    val clc_counter = Module(new Diy_Counter(unified_cycle, 1))
    
    val reg_before = Module(new Partial_Shift_Reg_Array(QuantizationWidth, one_in_channel_num, kernel_size_row, kernel_size_col, one_in_channel_num, kernel_size_row, stride_row))
    val reg_after = Module(new One_Shift_Reg_Array(QuantizationWidth, 1, 1))
    val reg_pool = Module(new One_Shift_Reg_Array(QuantizationWidth, 1, 1))

    val c_l = Module(new Convolutional_Layer(all_in_channel_num, one_in_channel_num, kernel_size_row, kernel_size_col))
    val p_l = Module(new Pooling_layer(is_max, 1, pooling_kernel_row, pooling_kernel_col))
    
    reg_before.io.signal := 0.U
    clc_counter.io.en := true.B

    when(clc_counter.io.number === 0.U){
        // counter init
        // counter_1 is a state counter for channel
        counter_1.io.en := true.B
        // counter_2 is a state counter for out-row
        counter_2.io.en := false.B
        // counter_3 is a state counter for result shift
        counter_3.io.en := true.B
        //
        counter_4.io.en := false.B
        // registers init
        
        reg_after.io.signal := 0.U
        reg_pool.io.signal := 0.U
        c_l.io.be_zero := false.B
    }.elsewhen(clc_counter.io.number <= current_cycle.U){
        c_l.io.be_zero := false.B
        when(counter_1.io.number =/= 0.U && counter_1.io.number =/= channel_times.U){
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := false.B
            counter_4.io.en := false.B
        }.elsewhen(counter_1.io.number === channel_times.U){
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := true.B
            counter_4.io.en := false.B
        }.elsewhen(counter_1.io.number === 0.U && counter_2.io.number =/= 0.U){
            counter_1.io.en := true.B
            counter_3.io.en := true.B
            counter_2.io.en := false.B
            when(clc_counter.io.number <= pool_cycle.U){
                counter_4.io.en := true.B
            }otherwise{
                counter_4.io.en := false.B
            }
            
        }otherwise{
            counter_1.io.en := false.B
            counter_2.io.en := false.B
            counter_3.io.en := false.B
            counter_4.io.en := false.B
        }

        when(counter_3.io.number === 0.U){
            reg_after.io.signal := 1.U
            when(counter_4.io.number === pool_wait.U){
                is_run := true.B
            }otherwise{
                is_run := false.B
            }
        }otherwise{
            reg_after.io.signal := 0.U
        }

        when(is_run === true.B){
            reg_pool.io.signal := 1.U
            is_run := false.B
        }otherwise{
            reg_pool.io.signal := 0.U
        }

    }.elsewhen(clc_counter.io.number > current_cycle.U && clc_counter.io.number < unified_cycle.U){
        c_l.io.be_zero := false.B
        reg_after.io.signal := 0.U
        counter_1.io.en := false.B
        counter_2.io.en := false.B
        counter_3.io.en := false.B
        counter_4.io.en := false.B

        when(is_run === true.B){
            reg_pool.io.signal := 1.U
            is_run := false.B
        }otherwise{
            reg_pool.io.signal := 0.U
        }
    }otherwise{
        c_l.io.be_zero := true.B
        reg_after.io.signal := 0.U
        reg_pool.io.signal := 0.U
        counter_1.io.en := true.B
        counter_3.io.en := true.B
        counter_2.io.en := false.B
        counter_4.io.en := true.B
    }
    
    c_l.io.data := reg_before.io.out_data

    for(j <- 0 until pooling_kernel_row){
        for(k <- 0 until pooling_kernel_col){
            if(k == pooling_kernel_col - 1 && j == pooling_kernel_row - 1)
                p_l.io.data(0)(j)(k) := reg_after.io.out_data(0)(0)(0)
            else
                p_l.io.data(0)(j)(k) := io.pool_data(j * pooling_kernel_col + k)
        }
    }
    


    c_l.io.weights := io.weights
    c_l.io.bias := io.bias
    c_l.io.data_zero := io.data_zero
    c_l.io.weight_zero := io.weight_zero
    c_l.io.result_zreo := io.result_zreo
    c_l.io.m_of_scale := io.m_of_scale

    reg_before.io.in_data := io.reg_in
    reg_after.io.in_data := c_l.io.result
    reg_pool.io.in_data := p_l.io.result(0)
    io.reg_out := reg_pool.io.out_data
}

class Multy_Columns_Convolutional_Layer(in_height: Int, out_channel_num: Int, all_in_channel_num: Int, one_in_channel_num_last_column: Int, one_in_channel_num_previous_columns: Int, kernel_size_row: Int, kernel_size_col: Int, stride_row: Int, unified_cycle: Int, multiple: Int) extends Module with DataConfig{

    // val out_height = ((in_height - kernel_size_row) / stride_row).toInt + 1
    // val channel_times_last_column = (all_in_channel_num / one_in_channel_num_last_column).toInt * out_channel_num - 1
    // val channel_times_previous_columns = ceil(all_in_channel_num / one_in_channel_num_previous_columns).toInt * out_channel_num - 1
    // val row_times = (in_height - kernel_size_row) / stride_row + 1 - 1

    // val current_cycle_l = (channel_times_last_column + 1) * (row_times + 1)
    // val current_cycle_p = (channel_times_previous_columns + 1) * (row_times + 1)
    val io = IO(new Bundle{
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        // last column related
        val in_last_column = Input(Vec(one_in_channel_num_last_column, Vec(kernel_size_row, Vec(1, SInt(QuantizationWidth.W)))))
        val weights_last_column = Input(Vec(one_in_channel_num_last_column,  Vec(kernel_size_row, Vec(1, SInt(QuantizationWidth.W)))))
        val bias_last_column = Input(SInt(DataWidth.W))
        val partialsum_last_column = Input(SInt(DataWidth.W))
        // previous columns related
        val in_previous_columns = Input(Vec(one_in_channel_num_previous_columns, Vec(kernel_size_row, Vec(1, SInt(QuantizationWidth.W)))))
        val weights_previous_columns = Input(Vec(kernel_size_col - 1, Vec(one_in_channel_num_previous_columns,  Vec(kernel_size_row, Vec(1, SInt(QuantizationWidth.W))))))
        val partialsum_previous_columns = Input(Vec(kernel_size_col - 1, SInt(DataWidth.W)))
        // out put
        val out_last_column = Output(Vec(1, Vec(1, Vec(1, SInt(QuantizationWidth.W)))))
        val out_previous_columns = Output(Vec(kernel_size_col - 1, Vec(1, Vec(1, Vec(1, SInt(QuantizationWidth.W))))))
    })

    val c_last = Module(new Partial_Convolutional_Layer(all_in_channel_num, one_in_channel_num_last_column, kernel_size_row, 1))
    val c_pre = VecInit(Seq.fill(kernel_size_col - 1)(Module(new Partial_Convolutional_Layer(all_in_channel_num, one_in_channel_num_previous_columns, kernel_size_row, 1)).io))
    // bias adder
    val bias_adder = Module(new Bias_Adder)
    // relu
    val relu = Module(new ReLU)
    val counter  = Module(new Diy_Counter(unified_cycle * multiple, 1))
    counter.io.en := true.B
    
    val reg_before_last = Module(new Partial_Shift_Reg_Array(QuantizationWidth, one_in_channel_num_last_column, kernel_size_row, 1, one_in_channel_num_last_column, kernel_size_row, stride_row))
    val reg_before_pre = VecInit(Seq.fill(kernel_size_col - 1)(Module(new Partial_Shift_Reg_Array(QuantizationWidth, one_in_channel_num_previous_columns, kernel_size_row, 1, one_in_channel_num_previous_columns, kernel_size_row, stride_row)).io))

    reg_before_last.io.signal := 0.U
    for(i <- 0 until kernel_size_col - 1){
        reg_before_pre(i).signal := 0.U
    }

    when(counter.io.number === (unified_cycle * multiple).U){
        c_last.io.be_zero := true.B
        for(i <- 0 until kernel_size_col - 1){
            c_pre(i).be_zero := true.B
        }
    }otherwise{
        c_last.io.be_zero := false.B
        for(i <- 0 until kernel_size_col - 1){
            c_pre(i).be_zero := false.B
        }
    }

    reg_before_last.io.in_data := io.in_last_column

    // port link last c
    c_last.io.be_zero := false.B
    c_last.io.data := reg_before_last.io.out_data
    c_last.io.weights := io.weights_last_column
    c_last.io.data_zero := io.data_zero
    c_last.io.weight_zero := io.weight_zero

    bias_adder.io.bias := io.bias_last_column
    bias_adder.io.data := c_last.io.result + io.partialsum_last_column
    bias_adder.io.result_zreo := io.result_zreo
    bias_adder.io.m_of_scale := io.m_of_scale
    
    relu.io.data := bias_adder.io.result
    relu.io.zero_point := io.result_zreo
    io.out_last_column(0)(0)(0) := relu.io.result

    // port link previous c
    for(i <- 0 until kernel_size_col - 1){
        reg_before_pre(i).in_data := io.in_previous_columns
        c_pre(i).be_zero := false.B
        c_pre(i).data := reg_before_pre(i).out_data
        c_pre(i).weights := io.weights_previous_columns(i)
        c_pre(i).data_zero := io.data_zero
        c_pre(i).weight_zero := io.weight_zero
        io.out_previous_columns(i)(0)(0)(0) := c_pre(i).result + io.partialsum_previous_columns(i)
    }

}

class Multy_Columns_Convolutional_Pooling_Layer(in_height: Int, out_channel_num: Int, all_in_channel_num: Int, one_in_channel_num_last_column: Int, one_in_channel_num_previous_columns: Int, kernel_size_row: Int, kernel_size_col: Int, stride_row: Int, pooling_kernel_row: Int, pooling_kernel_col: Int, pool_stride_row: Int, is_max: Boolean, unified_cycle: Int, multiple: Int) extends Module with DataConfig{

    // val out_height = ((in_height - kernel_size_row) / stride_row).toInt + 1
    // val channel_times_last_column = (all_in_channel_num / one_in_channel_num_last_column).toInt * out_channel_num - 1
    // val channel_times_previous_columns = ceil(all_in_channel_num / one_in_channel_num_previous_columns).toInt * out_channel_num - 1
    // val row_times = (in_height - kernel_size_row) / stride_row + 1 - 1

    // val current_cycle_l = (channel_times_last_column + 1) * (row_times + 1)
    // val current_cycle_p = (channel_times_previous_columns + 1) * (row_times + 1)
    val io = IO(new Bundle{
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        // last column related
        val in_last_column = Input(Vec(one_in_channel_num_last_column, Vec(kernel_size_row, Vec(1, SInt(QuantizationWidth.W)))))
        val weights_last_column = Input(Vec(one_in_channel_num_last_column,  Vec(kernel_size_row, Vec(1, SInt(QuantizationWidth.W)))))
        val bias_last_column = Input(SInt(DataWidth.W))
        val partialsum_last_column = Input(SInt(DataWidth.W))
        // previous columns related
        val in_previous_columns = Input(Vec(one_in_channel_num_previous_columns, Vec(kernel_size_row, Vec(1, SInt(QuantizationWidth.W)))))
        val weights_previous_columns = Input(Vec(kernel_size_col - 1, Vec(one_in_channel_num_previous_columns,  Vec(kernel_size_row, Vec(1, SInt(QuantizationWidth.W))))))
        val partialsum_previous_columns = Input(Vec(kernel_size_col - 1, SInt(DataWidth.W)))
        // pooling
        val pool_data = Input(Vec(pooling_kernel_row * pooling_kernel_col - 1,  SInt(QuantizationWidth.W)))
        // out put
        val out_last_column = Output(Vec(1, Vec(1, Vec(1, SInt(QuantizationWidth.W)))))
        val out_previous_columns = Output(Vec(kernel_size_col - 1, Vec(1, Vec(1, Vec(1, SInt(QuantizationWidth.W))))))
    })

    val c_last = Module(new Partial_Convolutional_Layer(all_in_channel_num, one_in_channel_num_last_column, kernel_size_row, 1))
    val c_pre = VecInit(Seq.fill(kernel_size_col - 1)(Module(new Partial_Convolutional_Layer(all_in_channel_num, one_in_channel_num_previous_columns, kernel_size_row, 1)).io))

    val p_last = Module(new Pooling_layer(is_max, 1, pooling_kernel_row, pooling_kernel_col))
    // bias adder
    val bias_adder = Module(new Bias_Adder)
    // relu
    val relu = Module(new ReLU)
    val counter  = Module(new Diy_Counter(unified_cycle * multiple, 1))
    counter.io.en := true.B

    val reg_before_last = Module(new Partial_Shift_Reg_Array(QuantizationWidth, one_in_channel_num_last_column, kernel_size_row, 1, one_in_channel_num_last_column, kernel_size_row, stride_row))
    val reg_before_pre = VecInit(Seq.fill(kernel_size_col - 1)(Module(new Partial_Shift_Reg_Array(QuantizationWidth, one_in_channel_num_previous_columns, kernel_size_row, 1, one_in_channel_num_previous_columns, kernel_size_row, stride_row)).io))

    reg_before_last.io.signal := 0.U
    for(i <- 0 until kernel_size_col - 1){
        reg_before_pre(i).signal := 0.U
    }

    when(counter.io.number === (unified_cycle * multiple).U){
        c_last.io.be_zero := true.B
        for(i <- 0 until kernel_size_col - 1){
            c_pre(i).be_zero := true.B
        }
    }otherwise{
        c_last.io.be_zero := false.B
        for(i <- 0 until kernel_size_col - 1){
            c_pre(i).be_zero := false.B
        }
    }

    reg_before_last.io.in_data := io.in_last_column
    // port link last c
    c_last.io.be_zero := false.B
    c_last.io.data := reg_before_last.io.out_data
    c_last.io.weights := io.weights_last_column
    c_last.io.data_zero := io.data_zero
    c_last.io.weight_zero := io.weight_zero

    bias_adder.io.bias := io.bias_last_column
    bias_adder.io.data := c_last.io.result + io.partialsum_last_column
    bias_adder.io.result_zreo := io.result_zreo
    bias_adder.io.m_of_scale := io.m_of_scale
    
    relu.io.data := bias_adder.io.result
    relu.io.zero_point := io.result_zreo
    
    for(j <- 0 until pooling_kernel_row){
        for(k <- 0 until pooling_kernel_col){
            if(k == pooling_kernel_col - 1 && j == pooling_kernel_row - 1)
                p_last.io.data(0)(j)(k) := relu.io.result
            else
                p_last.io.data(0)(j)(k) := io.pool_data(j * pooling_kernel_col + k)
        }
    }
    io.out_last_column(0)(0)(0) := p_last.io.result(0)


    // port link previous c
    for(i <- 0 until kernel_size_col - 1){
        reg_before_pre(i).in_data := io.in_previous_columns
        c_pre(i).be_zero := false.B
        c_pre(i).data := reg_before_pre(i).out_data
        c_pre(i).weights := io.weights_previous_columns(i)
        c_pre(i).data_zero := io.data_zero
        c_pre(i).weight_zero := io.weight_zero
        io.out_previous_columns(i)(0)(0)(0) := c_pre(i).result + io.partialsum_previous_columns(i)
    }

}

class Base_Convolutional_Layer(in_height: Int, out_channel_num: Int, all_in_channel_num: Int, one_in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int, stride_row: Int, unified_cycle: Int) extends Module with DataConfig{
    val io = IO(new Bundle{
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        // last column related
        val in_data = Input(Vec(one_in_channel_num, Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val bias = Input(SInt(DataWidth.W))
        // out put
        val out_data = Output(Vec(1, Vec(1, Vec(1, SInt(QuantizationWidth.W)))))
    })

    val c_last = Module(new Partial_Convolutional_Layer(all_in_channel_num, one_in_channel_num, kernel_size_row, kernel_size_col))
    // bias adder
    val bias_adder = Module(new Bias_Adder)
    // relu
    val relu = Module(new ReLU)
    val counter  = Module(new Diy_Counter(unified_cycle, 1))
    counter.io.en := true.B
    
    val reg_before = Module(new Partial_Shift_Reg_Array(QuantizationWidth, one_in_channel_num, kernel_size_row, kernel_size_col, one_in_channel_num, kernel_size_row, stride_row))

    reg_before.io.signal := 0.U

    when(counter.io.number === unified_cycle.U){
        c_last.io.be_zero := true.B
    }otherwise{
        c_last.io.be_zero := false.B
    }

    reg_before.io.in_data := io.in_data

    // port link last c
    c_last.io.be_zero := false.B
    c_last.io.data := reg_before.io.out_data
    c_last.io.weights := io.weights
    c_last.io.data_zero := io.data_zero
    c_last.io.weight_zero := io.weight_zero

    bias_adder.io.bias := io.bias
    bias_adder.io.data := c_last.io.result
    bias_adder.io.result_zreo := io.result_zreo
    bias_adder.io.m_of_scale := io.m_of_scale
    
    relu.io.data := bias_adder.io.result
    relu.io.zero_point := io.result_zreo
    io.out_data(0)(0)(0) := relu.io.result

}

class Base_Convolutional_Pooling_Layer(in_height: Int, out_channel_num: Int, all_in_channel_num: Int, one_in_channel_num: Int, kernel_size_row: Int, kernel_size_col: Int, stride_row: Int, pooling_kernel_row: Int, pooling_kernel_col: Int, pool_stride_row: Int, is_max: Boolean, unified_cycle: Int) extends Module with DataConfig{

    val io = IO(new Bundle{
        val data_zero = Input(SInt(QuantizationWidth.W))
        val weight_zero = Input(SInt(QuantizationWidth.W))
        val result_zreo = Input(SInt(DataWidth.W))
        val m_of_scale = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        // last column related
        val in_data = Input(Vec(one_in_channel_num, Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val weights = Input(Vec(one_in_channel_num,  Vec(kernel_size_row, Vec(kernel_size_col, SInt(QuantizationWidth.W)))))
        val bias = Input(SInt(DataWidth.W))
        // pooling
        val pool_data = Input(Vec(pooling_kernel_row * pooling_kernel_col - 1,  SInt(QuantizationWidth.W)))
        // out put
        val out_data = Output(Vec(1, Vec(1, Vec(1, SInt(QuantizationWidth.W)))))
    })

    val c_last = Module(new Partial_Convolutional_Layer(all_in_channel_num, one_in_channel_num, kernel_size_row, kernel_size_col))

    val p_last = Module(new Pooling_layer(is_max, 1, pooling_kernel_row, pooling_kernel_col))
    // bias adder
    val bias_adder = Module(new Bias_Adder)
    // relu
    val relu = Module(new ReLU)
    val counter  = Module(new Diy_Counter(unified_cycle, 1))
    counter.io.en := true.B

    val reg_before = Module(new Partial_Shift_Reg_Array(QuantizationWidth, one_in_channel_num, kernel_size_row, kernel_size_col, one_in_channel_num, kernel_size_row, stride_row))

    reg_before.io.signal := 0.U

    when(counter.io.number === unified_cycle.U){
        c_last.io.be_zero := true.B
    }otherwise{
        c_last.io.be_zero := false.B
    }

    reg_before.io.in_data := io.in_data
    // port link last c
    c_last.io.be_zero := false.B
    c_last.io.data := reg_before.io.out_data
    c_last.io.weights := io.weights
    c_last.io.data_zero := io.data_zero
    c_last.io.weight_zero := io.weight_zero

    bias_adder.io.bias := io.bias
    bias_adder.io.data := c_last.io.result
    bias_adder.io.result_zreo := io.result_zreo
    bias_adder.io.m_of_scale := io.m_of_scale
    
    relu.io.data := bias_adder.io.result
    relu.io.zero_point := io.result_zreo
    
    for(j <- 0 until pooling_kernel_row){
        for(k <- 0 until pooling_kernel_col){
            if(k == pooling_kernel_col - 1 && j == pooling_kernel_row - 1)
                p_last.io.data(0)(j)(k) := relu.io.result
            else
                p_last.io.data(0)(j)(k) := io.pool_data(j * pooling_kernel_col + k)
        }
    }
    io.out_data(0)(0)(0) := p_last.io.result(0)
}