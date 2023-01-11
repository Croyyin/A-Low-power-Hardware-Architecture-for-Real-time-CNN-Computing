package CNN
import chisel3._
import chisel3.experimental._

trait NetworkConfig{
    val net_structure = "cCcCccCccCccCfff"
    val in_height = Array(224, 224, 112, 112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 14, 14, 14, 7)
    val in_channel = Array(3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 25088, 4096, 4096)
    val out_channel = Array(64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 21)
    val kernel_size = Array(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    val stride = Array(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    val pooling_kernel_size = Array(2, 2, 2, 2, 2)
    val pooling_stride = Array(2, 2, 2, 2, 2)
    val fc_in = Array(25088, 4096, 4096)
    val fc_out = Array(4096, 4096, 21)
    val mini_channel = Array(1, 28, 14, 28, 14, 28, 28, 14, 28, 28, 7, 7, 7)
    val pre_mini_channel = Array(1, 28, 14, 28, 14, 28, 28, 14, 28, 28, 7, 7, 7)
    val min_fc_len = Array(98, 16, 1)
    val perlayer_cycle = Array(43008, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 2688)
    val unified_cycle = 1
    val multiple = Array(1, 28, 14, 28, 14, 28, 28, 14, 28, 28, 7, 7, 7)
}