import os
from copy import deepcopy
import argparse

area_report = "report_area >> %/rpt/&_area.rpt\n"
timing_report = "report_timing -path full >> %/rpt/&_timing.rpt\n"
power_report = "report_power >> %/rpt/&_power.rpt\n"
analyze = "analyze -format verilog {%/verilog_file/&.v}\n"
elaborate = "elaborate &\n" 
header = "source library_Setup\n"
wirte_verilog = "write -f verilog -output \"%/src/&.v\"\n"
time_clock = "create_clock clock -name clock -period &\n"
set_area = "set_max_area 0\n"
make_compile = "compile -ungroup_all\n"

def replace_syn(module_name, pre_path, cycle, wirte_v=1):
    an = deepcopy(analyze).replace("&", module_name).replace("%", pre_path)
    el = deepcopy(elaborate).replace("&", module_name)
    ar = deepcopy(area_report).replace("&", module_name).replace("%", pre_path)
    ti = deepcopy(timing_report).replace("&", module_name).replace("%", pre_path)
    po = deepcopy(power_report).replace("&", module_name).replace("%", pre_path)
    ex = "exit"
    tc = deepcopy(time_clock).replace("&", str(cycle))
    hd = deepcopy(header).replace("%", pre_path)
    if wirte_v:
        wv = deepcopy(wirte_verilog).replace("&", module_name).replace("%", pre_path)
        return hd + an + el + tc + set_area + make_compile + ar + ti + po + wv + ex
    else:
        return hd + an + el + tc + set_area + make_compile + ar + ti + po + ex

def files_name_read(path):
    file_list = os.listdir(path)
    result_list = []
    for f in file_list:
        if ".v" in f:
            result_list.append(f[:-2])
    return result_list

def syn_gen(path, pre_path, cycle, wirte_v=1):
    files = files_name_read(path)
    for f in files:
        if ".tcl" in f:
            pass
        else:
            content = replace_syn(f, pre_path, cycle, wirte_v)
            with open(path + "/syn_" + f + ".tcl", "w") as s:
                s.writelines(content)

def sh_gen(path):
    header_str = '#!/bin/sh\n'
    time_str = 'date "+%F %T"\n'
    sh_str = 'folder="./"\nsoftfiles=$(ls */verilog_file/*.tcl)\nfor sfile in ${softfiles}\ndo \n    run=$(dc_shell -f $sfile)\ndone\n'
    with open(path + "/run.sh", 'w') as f:
        f.write(header_str + time_str + sh_str + time_str)
    return

def lib_gen(path):
    lib_str = 'lappend search_path /opt/tsmc65/\nset target_library {/opt/tsmc65/CORE65LPSVT_nom_1.10V_25C.db}\nset link_library {/opt/tsmc65/CORE65LPSVT_nom_1.10V_25C.db}\nset symbol_library {/opt/tsmc65/CORE65LPSVT.sdb}\nset allow_newer_db_files "true"'
    with open(path + "/library_Setup", 'w') as f:
        f.write(lib_str)
    return
if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cycle", type=int, default=20, help='clock cycle')
    parser.add_argument("-p", "--path", type=str, default="./batch_verilog", help='parent dir of verilog_file')
    parser.add_argument("-w", "--write", type=int, default=1, help='clock cycle')
    args = parser.parse_args()

    file_list = os.listdir(args.path)
    for f in file_list:
        if "." in f or "lib" in f:
            pass
        else:
            if os.path.exists(args.path + "/" + f + "/" + "src"):
                pass
            else:
                os.mkdir(args.path + "/" + f + "/" + "src")
            if os.path.exists(args.path + "/" + f + "/" + "rpt"):
                pass
            else:
                os.mkdir(args.path + "/" + f + "/" + "rpt")
            syn_gen(args.path + "/" + f + "/" + "verilog_file", "./" + f, args.cycle, args.write)
    sh_gen(args.path)
    lib_gen(args.path)