package CNN

import chisel3._
import chisel3.experimental._

// 0 cycle
// MaxPooling
class Diy_Counter(Max: Int, Min: Int=0, Init : Int=0) extends Module with DataConfig{
    val io = IO(new Bundle{
        val en = Input(Bool())
        val number = Output(UInt(DataWidth.W))
    })

    val counter = RegInit(Init.U(DataWidth.W))
    when(io.en){
        when(counter === Max.U){
            counter := Min.U
        }otherwise{
            counter := counter + 1.U
        }
    }

    io.number := counter
}
