"""
Before use this file, install xgboost first
""" 
from __future__ import print_function
import sys
import xgboost as xgb
import os
import struct


def convert(input_model, output_file):
    model = xgb.Booster()
    model.load_model(input_model)
    tmp_file = output_file + ".gbdt_rs.mid"
    # extract base score
    try :
        with open(input_model) as f:
            model_format = struct.unpack('cccc',f.read(4))
            model_format = "".join(model_format)
            if model_format == "bs64":
                print("This model type is not supported")
            elif model_format != "binf":
                f.seek(0)
            base_score = struct.unpack('f',f.read(4))[0]
    except Exception as e:
        print("error: ", e)
        return 1
    
    if os.path.exists(tmp_file):
        print("Intermediate file %s exists. Please remove this file or change your output file path" % tmp_file)
        return 1

    model.dump_model(tmp_file, dump_format="json")
    try:
        with open(output_file, "w") as f:
            f.write(repr(base_score) + "\n")
            with open(tmp_file) as f2:
                for line in f2.xreadlines():
                    f.write(line)
    except Exception as e:
        print("error: ", e)
        return 1
    os.remove(tmp_file)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python script input_model_path output_file_path")
        exit(1)
    convert(sys.argv[1], sys.argv[2])
    

