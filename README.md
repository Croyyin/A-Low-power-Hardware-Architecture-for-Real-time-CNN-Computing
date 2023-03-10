# README

## Background
This project is the project involved in the paper "A Low-power Hardware Architecture for Real-time CNN
Computing" experiment. 

The following describes the directory structure of the project and a simple usage guide

The main directory structure of the project is as follow:
```
D:.
│  build.sbt
│  LICENSE
│  README.md              // read me file
│  
├─.bsp
│      sbt.json
│      
├─new_assistant
│  │  batch_dc.py         // generate syn file for verilog
│  │  config_save.py
│  │  model_config.py     // network structure
│  │  pb_balance.py       // generate all possible cases config files for Chisel
│  │  power_sta.py
│  │  print_line.py
│  │  test.py
│          
├─project
│  │  build.properties
│  │  
│  └─target
│                              
└─src
    ├─main
    │  └─scala
    │      ├─BaseModule
    │      │      ComputeUnit.scala
    │      │      Convolution.scala
    │      │      Counter.scala
    │      │      Linear.scala
    │      │      Pooling.scala
    │      │      RegArray.scala
    │      │      ReLU.scala
    │      │      
    │      ├─Config
    │      │      DataConfig.scala
    │      │      NetworkConfig.scala
    │      │      
    │      ├─LayerModule
    │      │      AfterTop.scala
    │      │      ConvolutionalLayer.scala
    │      │      FullyConnectedLayer.scala
    │      │      PoolingLayer.scala
    │      │      
    │      └─Net
    │              NetModel.scala
    │              
    └─test
        └─scala
                ModuleTest.scala
                NewGenerator.scala
                VerilogGenerator.scala        // main file to generate verilog from config files
```


## Usage
Here is a brief introduction on how to use the modified project to obtain static analysis results.

The CNN model structure is writed in file ```model_config.py```, here takes LeNet as an example.

First, enter the root directory and generate all possible hardware architecture configuration files of the LeNet structure through the following commands, and save them in directory ```./test_config```.
```
python ./new_assistant/pb_balance.py -m le -f gen -p ./test_config
```
Then enter ```sbt``` to enter the sbt shell, and enter the following command to generate a verilog behavior model based on the content of the configuration file.
```
test:runMain CNN.NetworkComponentsGen -td ./test_verilog/our/le -sd ./test_config/our/le
```
The following command adds the syn file for Design Complier to the generated verilog.
```
python ./new_assistant/batch_dc.py -p test_verilog/our/le
```
Enter the following command to jump to the directory.
```
cd test_verilog/our/le
```
Run the following command to start hardware synthesis.
```
sh ./run.sh
```
