package CNN
import chisel3.stage.{ChiselStage, ChiselGeneratorAnnotation}
import chisel3.experimental._
import chisel3._
import chisel3.RawModule
import java.io._
import java.io.File
import Array._


object CLGen extends App {
  // chisel3.Driver.execute(args, () => new Pipeline())
  (new chisel3.stage.ChiselStage).emitVerilog(new Convolutional_Layer(256, 4, 11, 11), args)
}
object RCLPGen extends App {
  // chisel3.Driver.execute(args, () => new Pipeline())
  (new chisel3.stage.ChiselStage).emitVerilog(new Reg_Convolutional_Layer_Reg_Pooling_layer_Reg(222, 64, 64, 4, 3, 3, 1, 2, 2, 2, true, 70), args)
}
object FCLGen extends App {
  // chisel3.Driver.execute(args, () => new Pipeline())
  (new chisel3.stage.ChiselStage).emitVerilog(new Mini_FC_Layer(4096, 1024, 16, 20), args)
}

object FGen extends App {
  // chisel3.Driver.execute(args, () => new Pipeline())
  (new chisel3.stage.ChiselStage).emitVerilog(new Fully_Connected_Layer(1024, 64), args)
}

// object NetworkComponentsGen extends App with NetworkConfig with DataConfig{
//     def delFile(file: File){
//         // delete
//         if (!file.exists()) {
//             return false
//         }
//         if (file.isDirectory()) {
//             val files = file.listFiles()
//             var fls = 0
//             for(fls <- 0 until files.length) {
//                 delFile(files(fls))
//             }
//         }
//         return file.delete();

//     }


//     val chiselStage = new chisel3.stage.ChiselStage                                         
//     var save_path = "verilog_file/"
    
//     var ar = 0
//     var td_index = -1
//     for(ar <- 0 until args.length){
//         if(args(ar) == "-td" || args(ar) == "--target-dir"){
//             td_index = ar
//         }
//     }
//     if(td_index != -1 && args.length > td_index + 1){
//         save_path = args(td_index + 1) + "/" + "verilog_file/"
//         args(td_index + 1) = args(td_index + 1) + "/trash"
//     }
//     var new_args = args
//     if(td_index == -1){
//         val ad_args = Array("--target-dir", "./trash")
//         new_args = concat(new_args, ad_args)
//     }
//     for(ar <- 0 until new_args.length){
//         if(new_args(ar) == "-td" || new_args(ar) == "--target-dir"){
//             td_index = ar
//         }
//     }

//     val file = new File(save_path);
// 		if (!file.exists()) {
// 			  file.mkdirs();
// 		}
    
//     // testprint
//     var tps = 0
//     for(tps <- 0 until new_args.length){
//       println(new_args(tps))
//     }


//     // conv
//     var stage = 0
//     var pool_index = 0
//     var fc_index = 0
//     for(stage <- 0 until net_structure.length()){

//         if(net_structure(stage) == 'c'){
//             val conv_string = "Convolutional_Layer_" + (stage + 1).toString
            
//             var conv_mini_module = chiselStage.emitVerilog(new Mini_Convolutional_Layer(in_height(stage), out_channel(stage), in_channel(stage), mini_channel(stage), kernel_size(stage)(0), kernel_size(stage)(1), stride(stage)(0), unified_cycle), new_args)

//             conv_mini_module = conv_mini_module.replace("module Mini_Convolutional_Layer", "module " + conv_string)
//             val writer = new PrintWriter(new File(save_path + conv_string + ".v"))
//             writer.write(conv_mini_module)
//             writer.close()
//         }else if(net_structure(stage) == 'C'){
//             val conv_string = "Convolutional_Pooling_Layer_" + (stage + 1).toString
            
//             var conv_mini_module = chiselStage.emitVerilog(new Mini_Convolutional_Pooling_layer(in_height(stage), out_channel(stage), in_channel(stage), mini_channel(stage), kernel_size(stage)(0), kernel_size(stage)(1), stride(stage)(0), pooling_kernel_size(pool_index)(0), pooling_kernel_size(pool_index)(1), pooling_stride(pool_index)(0), true, unified_cycle), new_args)

//             conv_mini_module = conv_mini_module.replace("module Mini_Convolutional_Pooling_layer", "module " + conv_string)
//             val writer = new PrintWriter(new File(save_path + conv_string + ".v"))
//             writer.write(conv_mini_module)
//             writer.close()

//             pool_index = pool_index + 1
//         }else if(net_structure(stage) == 'f'){
//             var fc_string = "FC_Layer_" + (fc_index + 1).toString

//             var fc = chiselStage.emitVerilog(new Mini_FC_Layer(fc_out(fc_index), fc_in(fc_index), min_fc_len(fc_index), unified_cycle), new_args)

//             fc = fc.replace("module Mini_FC_Layer", "module " + fc_string)
//             val writer = new PrintWriter(new File(save_path + fc_string  + ".v"))
//             writer.write(fc)
//             writer.close()

//             fc_index = fc_index + 1
//         }
//     }
//     // delete
//     val del_file = new File(new_args(td_index + 1))
//     delFile(del_file)
// }

object ConvPowerTest extends App with NetworkConfig with DataConfig{
    def delFile(file: File){
        // delete
        if (!file.exists()) {
            return false
        }
        if (file.isDirectory()) {
            val files = file.listFiles()
            var fls = 0
            for(fls <- 0 until files.length) {
                delFile(files(fls))
            }
        }
        return file.delete();

    }


    val chiselStage = new chisel3.stage.ChiselStage                                         
    var save_path = "syn/"
    
    var ar = 0
    var td_index = -1
    for(ar <- 0 until args.length){
        if(args(ar) == "-td" || args(ar) == "--target-dir"){
            td_index = ar
        }
    }
    if(td_index != -1 && args.length > td_index + 1){
        save_path = args(td_index + 1) + "/" + "syn/"
        args(td_index + 1) = args(td_index + 1) + "/trash"
    }
    var new_args = args
    if(td_index == -1){
        val ad_args = Array("--target-dir", "./trash")
        new_args = concat(new_args, ad_args)
    }
    for(ar <- 0 until new_args.length){
        if(new_args(ar) == "-td" || new_args(ar) == "--target-dir"){
            td_index = ar
        }
    }

    val file = new File(save_path);
		if (!file.exists()) {
			  file.mkdirs();
		}
    
    // testprint
    var tps = 0
    for(tps <- 0 until new_args.length){
      println(new_args(tps))
    }
    val base_in_height = 50
    val base_out_channel = 64
    val base_stride = 1
    var base_all_inchannel = 32
    var base_mini_inchannel = 1
    var base_kernel_size = 3
    val base_cyc = (base_in_height - base_kernel_size + 1) *  base_out_channel * base_all_inchannel
    // test mini channel
    val max_mini_channel = 256
    base_all_inchannel = 256
    base_mini_inchannel = 1
    base_kernel_size = 3
    while(base_mini_inchannel <= max_mini_channel){
        val conv_string = "Mini_Convolutional_Layer" + (base_all_inchannel).toString + "_" + (base_mini_inchannel).toString + "_" + (base_kernel_size).toString + "_" + (base_kernel_size).toString
        
        var conv_mini_module = chiselStage.emitVerilog(new Mini_Convolutional_Layer(base_in_height,base_out_channel, base_all_inchannel, base_mini_inchannel, base_kernel_size, base_kernel_size, base_stride, base_cyc), new_args)
        conv_mini_module = conv_mini_module.replace("module Mini_Convolutional_Layer", "module " +conv_string)
        val writer = new PrintWriter(new File(save_path + conv_string + ".v"))
        writer.write(conv_mini_module)
        writer.close()
        // self up
        base_mini_inchannel = base_mini_inchannel * 2
    }

    // test kernel size
    val max_kernel_size = 11
    base_all_inchannel = 256
    base_mini_inchannel = 4
    base_kernel_size = 3
    while(base_kernel_size <= max_kernel_size){
        val conv_string = "Mini_Convolutional_Layer" + (base_all_inchannel).toString + "_" + (base_mini_inchannel).toString + "_" + (base_kernel_size).toString + "_" + (base_kernel_size).toString
        
        var conv_mini_module = chiselStage.emitVerilog(new Mini_Convolutional_Layer(base_in_height,base_out_channel, base_all_inchannel, base_mini_inchannel, base_kernel_size, base_kernel_size, base_stride, base_cyc), new_args)
        conv_mini_module = conv_mini_module.replace("module Mini_Convolutional_Layer", "module " +conv_string)
        val writer = new PrintWriter(new File(save_path + conv_string + ".v"))
        writer.write(conv_mini_module)
        writer.close()
        // self up
        base_kernel_size = base_kernel_size + 2
    }


    // delete
    val del_file = new File(new_args(td_index + 1))
    delFile(del_file)
}

object FCPowerTest extends App with NetworkConfig with DataConfig{
    def delFile(file: File){
        // delete
        if (!file.exists()) {
            return false
        }
        if (file.isDirectory()) {
            val files = file.listFiles()
            var fls = 0
            for(fls <- 0 until files.length) {
                delFile(files(fls))
            }
        }
        return file.delete();

    }


    val chiselStage = new chisel3.stage.ChiselStage                                         
    var save_path = "syn/"
    
    var ar = 0
    var td_index = -1
    for(ar <- 0 until args.length){
        if(args(ar) == "-td" || args(ar) == "--target-dir"){
            td_index = ar
        }
    }
    if(td_index != -1 && args.length > td_index + 1){
        save_path = args(td_index + 1) + "/" + "syn/"
        args(td_index + 1) = args(td_index + 1) + "/trash"
    }
    var new_args = args
    if(td_index == -1){
        val ad_args = Array("--target-dir", "./trash")
        new_args = concat(new_args, ad_args)
    }
    for(ar <- 0 until new_args.length){
        if(new_args(ar) == "-td" || new_args(ar) == "--target-dir"){
            td_index = ar
        }
    }

    val file = new File(save_path);
		if (!file.exists()) {
			  file.mkdirs();
		}
    
    // testprint
    var tps = 0
    for(tps <- 0 until new_args.length){
      println(new_args(tps))
    }
    val base_out_len = 1000
    val base_in_len = 10000
    var base_mini_len = 1
    val base_cyc = (base_in_len / base_mini_len) * base_out_len
    // len
    val max_mini_len = 1024
    while(base_mini_len <= max_mini_len){
        var fc_string = "Mini_FC_Layer_" + (base_in_len).toString + "_" + (base_mini_len).toString

        var fc = chiselStage.emitVerilog(new Mini_FC_Layer(base_out_len, base_in_len, base_mini_len, base_cyc), new_args)

        fc = fc.replace("module Mini_FC_Layer", "module " + fc_string)
        val writer = new PrintWriter(new File(save_path + fc_string  + ".v"))
        writer.write(fc)
        writer.close()

        base_mini_len = base_mini_len * 2
    }

    // delete
    val del_file = new File(new_args(td_index + 1))
    delFile(del_file)
}

object NetworkComponentsBatchGen extends App with NetworkConfig with DataConfig{
    def delFile(file: File){
        // delete
        if (!file.exists()) {
            return false
        }
        if (file.isDirectory()) {
            val files = file.listFiles()
            var fls = 0
            for(fls <- 0 until files.length) {
                delFile(files(fls))
            }
        }
        return file.delete();

    }


    val chiselStage = new chisel3.stage.ChiselStage      


    var save_path = "verilog_file/"
    var ar = 0
    var td_index = -1
    for(ar <- 0 until args.length){
        if(args(ar) == "-td" || args(ar) == "--target-dir"){
            td_index = ar
        }
    }
    if(td_index != -1 && args.length > td_index + 1){
        save_path = args(td_index + 1) + "/" + "verilog_file/"
        args(td_index + 1) = args(td_index + 1) + "/trash"
    }
    var new_args = args
    if(td_index == -1){
        val ad_args = Array("--target-dir", "./trash")
        new_args = concat(new_args, ad_args)
    }
    for(ar <- 0 until new_args.length){
        if(new_args(ar) == "-td" || new_args(ar) == "--target-dir"){
            td_index = ar
        }
    }

    val file = new File(save_path);
		if (!file.exists()) {
			  file.mkdirs();
		}
    
    // testprint
    var tps = 0
    for(tps <- 0 until new_args.length){
      println(new_args(tps))
    }


    // conv
    var stage = 0
    var pool_index = 0
    var fc_index = 0
    var count = 0
    for(stage <- 0 until net_structure.length()){

        if(net_structure(stage) == 'c'){
            val conv_string = "Convolutional_Layer_" + (stage + 1).toString
            
            var conv_mini_module = chiselStage.emitVerilog(new Mini_Convolutional_Layer(in_height(stage), out_channel(stage), in_channel(stage), mini_channel(stage), kernel_size(stage), kernel_size(stage), stride(stage), perlayer_cycle(count)), new_args)

            conv_mini_module = conv_mini_module.replace("module Mini_Convolutional_Layer", "module " + conv_string)
            val writer = new PrintWriter(new File(save_path + conv_string + ".v"))
            writer.write(conv_mini_module)
            writer.close()
        }else if(net_structure(stage) == 'C'){
            val conv_string = "Convolutional_Pooling_Layer_" + (stage + 1).toString
            
            var conv_mini_module = chiselStage.emitVerilog(new Mini_Convolutional_Pooling_layer(in_height(stage), out_channel(stage), in_channel(stage), mini_channel(stage), kernel_size(stage), kernel_size(stage), stride(stage), pooling_kernel_size(pool_index), pooling_kernel_size(pool_index), pooling_stride(pool_index), true, perlayer_cycle(count)), new_args)

            conv_mini_module = conv_mini_module.replace("module Mini_Convolutional_Pooling_layer", "module " + conv_string)
            val writer = new PrintWriter(new File(save_path + conv_string + ".v"))
            writer.write(conv_mini_module)
            writer.close()

            pool_index = pool_index + 1
        }else if(net_structure(stage) == 'f'){
            var fc_string = "FC_Layer_" + (fc_index + 1).toString

            var fc = chiselStage.emitVerilog(new Mini_FC_Layer(fc_out(fc_index), fc_in(fc_index), min_fc_len(fc_index), perlayer_cycle(count)), new_args)

            fc = fc.replace("module Mini_FC_Layer", "module " + fc_string)
            val writer = new PrintWriter(new File(save_path + fc_string  + ".v"))
            writer.write(fc)
            writer.close()

            fc_index = fc_index + 1
        }
        count += 1
    }
    // delete
    val del_file = new File(new_args(td_index + 1))
    delFile(del_file)
}
