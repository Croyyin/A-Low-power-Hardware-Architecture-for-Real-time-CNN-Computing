package CNN
import chisel3.stage.{ChiselStage, ChiselGeneratorAnnotation}
import chisel3.experimental._
import chisel3._
import chisel3.RawModule
import java.io._
import java.io.File
import Array._
import scala.io.{BufferedSource, Source}
import scala.runtime.ScalaRunTime._


object NetworkComponentsGen extends App with DataConfig{
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
    var save_path = "./"
    // 确定参数位置
    var ar = 0
    var td_index = -1
    var sd_index = -1
    for(ar <- 0 until args.length){
        if(args(ar) == "-td" || args(ar) == "--target-dir"){
            td_index = ar
        }
        if(args(ar) == "-sd" || args(ar) == "--source-dir"){
            sd_index = ar
        }
    }
    if(sd_index == -1){
        println("No source files")
        sys.exit(0)
    }
    // 确定生成文件路径
    if(td_index != -1 && args.length > td_index + 1){
        save_path = args(td_index + 1) + "/" 
        args(td_index + 1) = args(td_index + 1) + "/trash"
    }
    // 若不存在该生成路径则添加生成路径
    var flag = 0
    var b_idx = 0
    var new_args = new Array[String](args.size - 2) 
    for(i <- 0 until args.size){
        if(args(i) == "-sd" || args(i) == "--source-dir"){
            flag = 1
            b_idx = b_idx - 1
        }else if(flag == 1){
            flag = 0
            b_idx = b_idx - 1
        }else{
            new_args(i + b_idx) = args(i)
        }
        
    }

    if(td_index == -1){
        val ad_args = Array("--target-dir", "./trash")
        new_args = concat(new_args, ad_args)
    }
    // 控制trash的删除
    for(ar <- 0 until new_args.length){
        if(new_args(ar) == "-td" || new_args(ar) == "--target-dir"){
            td_index = ar
        }
    }

    
    

    var base_path = args(sd_index + 1)
    var net = new NetModel()
    var source_file = new File(base_path);
    var files:Array[String] = source_file.list()
    for(i <- 0 until files.size){
        net.read_file(base_path + "/" + files(i))
        var save_name = files(i).substring(0, files(i).length() - 4)
        var save_dir = save_path + save_name + "/verilog_file"
        val file = new File(save_dir);
	    if (!file.exists()){
	    	  file.mkdirs()
	    }

        // conv
        var stage = 0
        var pool_index = 0
        var fc_index = 0
        for(stage <- 0 until net.net_structure.length()){
        
            if(net.net_structure(stage) == 'c'){
                val conv_string = "Convolutional_Layer_" + (stage + 1).toString
                var conv_mini_module = chiselStage.emitVerilog(new Multy_Columns_Convolutional_Layer(net.in_height(stage), net.out_channel(stage), net.in_channel(stage), net.last_in_channel(stage), net.pre_in_channel(stage), net.kernel_size_row(stage), net.kernel_size_col(stage), net.kernel_stride_row(stage), net.unified_cycle, net.multiple(stage)), new_args)
    
                conv_mini_module = conv_mini_module.replace("module Multy_Columns_Convolutional_Layer", "module " + conv_string)
                val writer = new PrintWriter(new File(save_dir + "/" + conv_string + ".v"))
                writer.write(conv_mini_module)
                writer.close()
            }else if(net.net_structure(stage) == 'C'){
                val conv_string = "Convolutional_Pooling_Layer_" + (stage + 1).toString
                
                var conv_mini_module = chiselStage.emitVerilog(new Multy_Columns_Convolutional_Pooling_Layer(net.in_height(stage), net.out_channel(stage), net.in_channel(stage), net.last_in_channel(stage), net.pre_in_channel(stage), net.kernel_size_row(stage), net.kernel_size_col(stage), net.kernel_stride_row(stage), net.pooling_kernel_size_row(pool_index), net.pooling_kernel_size_col(pool_index), net.pooling_kernel_stride_row(pool_index), true, net.unified_cycle, net.multiple(stage)), new_args)
    
                conv_mini_module = conv_mini_module.replace("module Multy_Columns_Convolutional_Pooling_Layer", "module " + conv_string)
                val writer = new PrintWriter(new File(save_dir + "/" + conv_string + ".v"))
                writer.write(conv_mini_module)
                writer.close()
    
                pool_index = pool_index + 1
            }else if(net.net_structure(stage) == 'f'){
                var fc_string = "FC_Layer_" + (fc_index + 1).toString
                if(stage == net.net_structure.length() - 1){
                    var fc = chiselStage.emitVerilog(new Mini_FC_Layer(net.fc_out_size(fc_index), net.fc_in_size(fc_index), net.mini_fc_len(fc_index), net.unified_cycle), new_args)
                    fc = fc.replace("module Mini_FC_Layer", "module " + fc_string)
                    val writer = new PrintWriter(new File(save_dir + "/" + fc_string  + ".v"))
                    writer.write(fc)
                    writer.close()
                }else{
                    var fc = chiselStage.emitVerilog(new Mini_FC_Layer_ReLU(net.fc_out_size(fc_index), net.fc_in_size(fc_index), net.mini_fc_len(fc_index), net.unified_cycle), new_args)
                    fc = fc.replace("module Mini_FC_Layer_ReLU", "module " + fc_string)
                    val writer = new PrintWriter(new File(save_dir + "/" + fc_string  + ".v"))
                    writer.write(fc)
                    writer.close()
                }
                
                fc_index = fc_index + 1
            }
        }



    }



        
    // delete
    val del_file = new File(new_args(td_index + 1))
    delFile(del_file)
}

object BaseNetworkComponentsGen extends App with DataConfig{
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
    var save_path = "./"
    // 确定参数位置
    var ar = 0
    var td_index = -1
    var sd_index = -1
    for(ar <- 0 until args.length){
        if(args(ar) == "-td" || args(ar) == "--target-dir"){
            td_index = ar
        }
        if(args(ar) == "-sd" || args(ar) == "--source-dir"){
            sd_index = ar
        }
    }
    if(sd_index == -1){
        println("No source files")
        sys.exit(0)
    }
    // 确定生成文件路径
    if(td_index != -1 && args.length > td_index + 1){
        save_path = args(td_index + 1) + "/" 
        args(td_index + 1) = args(td_index + 1) + "/trash"
    }
    // 若不存在该生成路径则添加生成路径
    var flag = 0
    var b_idx = 0
    var new_args = new Array[String](args.size - 2) 
    for(i <- 0 until args.size){
        if(args(i) == "-sd" || args(i) == "--source-dir"){
            flag = 1
            b_idx = b_idx - 1
        }else if(flag == 1){
            flag = 0
            b_idx = b_idx - 1
        }else{
            new_args(i + b_idx) = args(i)
        }
        
    }

    if(td_index == -1){
        val ad_args = Array("--target-dir", "./trash")
        new_args = concat(new_args, ad_args)
    }
    // 控制trash的删除
    for(ar <- 0 until new_args.length){
        if(new_args(ar) == "-td" || new_args(ar) == "--target-dir"){
            td_index = ar
        }
    }

    
    

    var base_path = args(sd_index + 1)
    var net = new BaseNetModel()
    var source_file = new File(base_path);
    var files:Array[String] = source_file.list()
    for(i <- 0 until files.size){
        net.read_file(base_path + "/" + files(i))
        var save_name = files(i).substring(0, files(i).length() - 4)
        var save_dir = save_path + save_name + "/verilog_file"
        val file = new File(save_dir);
	    if (!file.exists()){
	    	  file.mkdirs()
	    }

        // conv
        var stage = 0
        var pool_index = 0
        var fc_index = 0
        for(stage <- 0 until net.net_structure.length()){
        
            if(net.net_structure(stage) == 'c'){
                val conv_string = "Convolutional_Layer_" + (stage + 1).toString
                var conv_mini_module = chiselStage.emitVerilog(new Base_Convolutional_Layer(net.in_height(stage), net.out_channel(stage), net.in_channel(stage), net.mini_in_channel(stage), net.kernel_size_row(stage), net.kernel_size_col(stage), net.kernel_stride_row(stage), net.unified_cycle), new_args)
    
                conv_mini_module = conv_mini_module.replace("module Base_Convolutional_Layer", "module " + conv_string)
                val writer = new PrintWriter(new File(save_dir + "/" + conv_string + ".v"))
                writer.write(conv_mini_module)
                writer.close()
            }else if(net.net_structure(stage) == 'C'){
                val conv_string = "Convolutional_Pooling_Layer_" + (stage + 1).toString
                
                var conv_mini_module = chiselStage.emitVerilog(new Base_Convolutional_Pooling_Layer(net.in_height(stage), net.out_channel(stage), net.in_channel(stage), net.mini_in_channel(stage), net.kernel_size_row(stage), net.kernel_size_col(stage), net.kernel_stride_row(stage), net.pooling_kernel_size_row(pool_index), net.pooling_kernel_size_col(pool_index), net.pooling_kernel_stride_row(pool_index), true, net.unified_cycle), new_args)
    
                conv_mini_module = conv_mini_module.replace("module Base_Convolutional_Pooling_Layer", "module " + conv_string)
                val writer = new PrintWriter(new File(save_dir + "/" + conv_string + ".v"))
                writer.write(conv_mini_module)
                writer.close()
    
                pool_index = pool_index + 1
            }else if(net.net_structure(stage) == 'f'){
                var fc_string = "FC_Layer_" + (fc_index + 1).toString
                if(stage == net.net_structure.length() - 1){
                    var fc = chiselStage.emitVerilog(new Mini_FC_Layer(net.fc_out_size(fc_index), net.fc_in_size(fc_index), net.mini_fc_len(fc_index), net.unified_cycle), new_args)
                    fc = fc.replace("module Mini_FC_Layer", "module " + fc_string)
                    val writer = new PrintWriter(new File(save_dir + "/" + fc_string  + ".v"))
                    writer.write(fc)
                    writer.close()
                }else{
                    var fc = chiselStage.emitVerilog(new Mini_FC_Layer_ReLU(net.fc_out_size(fc_index), net.fc_in_size(fc_index), net.mini_fc_len(fc_index), net.unified_cycle), new_args)
                    fc = fc.replace("module Mini_FC_Layer_ReLU", "module " + fc_string)
                    val writer = new PrintWriter(new File(save_dir + "/" + fc_string  + ".v"))
                    writer.write(fc)
                    writer.close()
                }
                
                fc_index = fc_index + 1
            }
        }



    }



        
    // delete
    val del_file = new File(new_args(td_index + 1))
    delFile(del_file)
}