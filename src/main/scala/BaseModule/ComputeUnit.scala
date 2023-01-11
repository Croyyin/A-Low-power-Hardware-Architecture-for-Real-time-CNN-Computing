package CNN

import chisel3._
import chisel3.experimental._
import scala.math._

// 量化部分乘法
class PartialQMultiplier extends Module with DataConfig{
    val io =IO(new Bundle{
        val qa = Input(SInt(QuantizationWidth.W))
        val qb = Input(SInt(QuantizationWidth.W))
        val za = Input(SInt(QuantizationWidth.W))
        val zb = Input(SInt(QuantizationWidth.W))
        val r = Output(SInt((QuantizationWidth * 2 + 2).W))
    })

    io.r := (io.qa - io.za) * (io.qb - io.zb)
}

object PartialQMultiplier{
    def apply(qa: SInt, za: SInt, qb: SInt, zb: SInt): SInt = {
        val inst = Module(new PartialQMultiplier)
        inst.io.qa := qa 
        inst.io.qb := qb
        inst.io.za := za 
        inst.io.zb := zb
        
        inst.io.r
    }
}

// 量化结果
class PartialQResult extends Module with DataConfig{
    val io =IO(new Bundle{
        val sum = Input(SInt(DataWidth.W))
        val b = Input(SInt(DataWidth.W))
        val S = Input(FixedPoint(FPWidth.W, BinaryPoint.BP))
        val Zr = Input(SInt(QuantizationWidth.W))
        val r = Output(SInt(QuantizationWidth.W))
    })
    val mlp = SIntToFixedPoint(DataWidth * 2, io.sum + io.b) * io.S
    val result = FixedPointToSInt(DataWidth * 2, mlp) + io.Zr
    val max = (math.pow(2, (QuantizationWidth - 1)).toInt - 1).S
    val min = (0 - math.pow(2, (QuantizationWidth - 1)).toInt).S
    
    when(result > max){
        io.r := max
    }.elsewhen(result < min){
        io.r := min
    }.otherwise{
        io.r := result
    }
}

object PartialQResult{
    def apply(sum: SInt, b: SInt, S: FixedPoint, Zr: SInt): SInt = {
        val inst = Module(new PartialQResult)
        inst.io.sum := sum
        inst.io.b := b 
        inst.io.S := S
        inst.io.Zr := Zr
        inst.io.r
    }
}


class SIntToFixedPoint(SintWidth: Int) extends Module with DataConfig{
    // 端口
	val io = IO(new Bundle{
        val in = Input(SInt(SintWidth.W))
        val out = Output(FixedPoint((BinaryPoint + SintWidth).W, BinaryPoint.BP))
    })

    val new_a = Wire(SInt((BinaryPoint + SintWidth).W))
    new_a := io.in << BinaryPoint
    io.out := new_a.asTypeOf(io.out)
}

object SIntToFixedPoint{
    def apply(SintWidth: Int, a: SInt): FixedPoint = {
        val inst = Module(new SIntToFixedPoint(SintWidth))
        inst.io.in := a
        inst.io.out
    }
}

class FixedPointToSInt(SintWidth: Int) extends Module with DataConfig{
    val io = IO(new Bundle{
        val in = Input(FixedPoint((SintWidth + BinaryPoint).W, BinaryPoint.BP))
        val out = Output(SInt(SintWidth.W))
    })

    io.out := io.in.asSInt >> BinaryPoint
    
}

object FixedPointToSInt{
    def apply(SintWidth: Int, a: FixedPoint): SInt = {
        val inst = Module(new FixedPointToSInt(SintWidth))
        inst.io.in := a 
        inst.io.out
    }
}