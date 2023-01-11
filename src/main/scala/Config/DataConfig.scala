package CNN
import chisel3._
import chisel3.experimental._


trait DataConfig{
    val DataWidth = 32
    val FPWidth = 64
    val BinaryPoint = 32
    val QuantizationWidth = 8
}
