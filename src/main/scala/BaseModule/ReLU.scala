package CNN

import chisel3._
import chisel3.experimental._

// 0 cycle
class ReLU extends Module with DataConfig{
  	val io = IO(new Bundle 		
	{
		val data  = Input(SInt(QuantizationWidth.W))
		val zero_point = Input(SInt(QuantizationWidth.W))
    	val result	= Output(SInt(QuantizationWidth.W))
  	})

	when (io.data < io.zero_point) 
	{
		io.result := io.zero_point
	}
	.otherwise
	{	
		io.result := io.data
	}
	
}

