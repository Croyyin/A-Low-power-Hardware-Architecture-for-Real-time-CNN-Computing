package CNN

import chisel3._
import chisel3.experimental._

class Reg_Array(Datawidth: Int, Channel: Int, Row: Int, Column: Int) extends Module with DataConfig{

    // Index转换
    def InDeX(Ch: Int, Rw: Int, Cl: Int): Int={
        var index = Ch * (Row * Column) + Rw * Column + Cl
        return index
    }

    // 端口
    val io = IO(new Bundle{
        val in_data = Output(Vec(Channel, Vec(Row, Vec(1, SInt(Datawidth.W)))))
        val out_data = Output(Vec(Channel, Vec(Row, Vec(Column, SInt(Datawidth.W)))))
      })


    val regArray = Reg(Vec(Channel * Row * Column, SInt(Datawidth.W)))
    // 第一列赋值
    for(i <- 0 until Channel){
        var j = 0
        for(j <- 0 until Row){
            regArray(InDeX(i, j, Column - 1)) := io.in_data(i)(j)(0)
        }
    }


    // 结果连接
    for(i <- 0 until Channel){
        var j = 0
        for(j <- 0 until Row){
            var k = 0
            for(k <- 0 until Column){
                io.out_data(i)(j)(k) := regArray(InDeX(i, j, k))
            }
        }
    }

}

class Partial_Shift_Reg_Array(data_width: Int, channel: Int, row: Int, column: Int, out_put_channel: Int, out_put_row: Int, shift_step: Int) extends Module with DataConfig{

    // Index转换
    def InDeX(Ch: Int, Rw: Int, Cl: Int): Int={
        var index = Ch * (row * column) + Rw * column + Cl
        return index
    }

    // 端口
    val io = IO(new Bundle{
        val in_data = Input(Vec(channel, Vec(row, Vec(column, SInt(data_width.W)))))
        val signal = Input(UInt(QuantizationWidth.W))
        val out_data = Output(Vec(out_put_channel, Vec(out_put_row, Vec(column, SInt(data_width.W)))))
      })


    val regArray = Reg(Vec(channel * row * column, SInt(data_width.W)))

    
    // 
    for(i <- 0 until channel){
        for(j <- 0 until row){
            for(k <- 0 until column){
                regArray(InDeX(i, j, k)) := io.in_data(i)(j)(k)
            }
        }
    }

    val shift_times = (channel / out_put_channel).toInt  - 1
    val row_shift_times = ((row - out_put_row) / shift_step + 1).toInt  - 1


    // partial shift
    when(io.signal === 1.U){
        for(rs <- 0 until row_shift_times + 1){
            for(i <- 0 until shift_times){
                for(ii <- 0 until out_put_channel){
                    for(j <- 0 until out_put_row){
                        for(k <- 0 until column){
                            regArray(InDeX(i * out_put_channel + ii, j + rs * shift_step, k)) := regArray(InDeX((i + 1) *  out_put_channel + ii, j + rs * shift_step, k))
                        }
                    }
                }
            }
        }
        for(rs <- 0 until row_shift_times + 1){
            for(ii <- 0 until out_put_channel){
                for(j <- 0 until out_put_row){
                    for(k <- 0 until column){
                        regArray(InDeX(ii + shift_times * out_put_channel, j + rs * shift_step, k)) := regArray(InDeX(ii, j +  rs * shift_step, k))
                    }
                }
            }

        }
    }.elsewhen(io.signal === 2.U){
        for(i <- 0 until channel){
            for(j <- 0 until row - shift_step){
                for(k <- 0 until column){
                    regArray(InDeX(i, j, k)) := regArray(InDeX(i, j + shift_step, k))
                }
            }
        }
    }otherwise{
        for(i <- 0 until channel){
            for(j <- 0 until row){
                for(k <- 0 until column){
                    regArray(InDeX(i, j, k)) := io.in_data(i)(j)(k)
                }
            }
        }
    }


    // 结果连接
    for(i <- 0 until out_put_channel){
        for(j <- 0 until out_put_row){
            for(k <- 0 until column){
                io.out_data(i)(j)(k) := regArray(InDeX(i, j, k))
            }
        }
    }

}

class One_Shift_Reg_Array(data_width: Int, channel: Int, row: Int) extends Module with DataConfig{

    // Index转换
    def InDeX(Ch: Int, Rw: Int, Cl: Int): Int={
        var index = Ch * (row * 1) + Rw * 1 + Cl
        return index
    }

    // 端口
    val io = IO(new Bundle{
        val in_data = Input(SInt(data_width.W))
        val signal = Input(UInt(QuantizationWidth.W))
        val out_data = Output(Vec(channel, Vec(row, Vec(1, SInt(data_width.W)))))
      })


    val regArray = Reg(Vec(channel * row * 1, SInt(data_width.W)))

    // 
    regArray(InDeX(channel - 1, row - 1, 0)) := io.in_data
    // row and channel shift when signal is ture
    when(io.signal === 1.U){
        regArray(InDeX(channel - 1, row - 1, 0)) := io.in_data
        for(i <- 0 until channel - 1){
            for(j <- 0 until row){
                regArray(InDeX(i, j, 0)) := regArray(InDeX(i + 1, j, 0))
            }
        }
        for(j <- 0 until row - 1){
            regArray(InDeX(channel - 1, j, 0)) := regArray(InDeX(0, j + 1, 0))
        }
    }otherwise{
        for(i <- 0 until channel){
            for(j <- 0 until row){
                regArray(InDeX(i, j, 0)) := regArray(InDeX(i, j, 0))
            }
        }
    }

    

    // 结果连接
    for(i <- 0 until channel){
        for(j <- 0 until row){
            io.out_data(i)(j)(0) := regArray(InDeX(i, j, 0))
        }
    }

}


