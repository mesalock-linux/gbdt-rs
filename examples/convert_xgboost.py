"""
Before use this file, install xgboost first
"""
import os
import struct
import sys

import xgboost as xgb


def convert(input_model, output_file):
    model = xgb.Booster()
    model.load_model(input_model)
    tmp_file = output_file + ".gbdt_rs.mid"
    # extract base score
    try :
        with open(input_model, "rb") as f:
            model_format = struct.unpack('cccc',f.read(4))
            model_format = b"".join(model_format)
            if model_format == b"bs64":
                print("This model type is not supported")
            elif model_format != b"binf":
                f.seek(0)
            base_score = struct.unpack('f',f.read(4))[0]
    except Exception as e:
        print("error: ", e)
        return
    
    if os.path.exists(tmp_file):
        print("Intermediate file %s exists. Please remove this file or change your output file path" % tmp_file)
        return

    # dump json
    model.dump_model(tmp_file, dump_format="json")

    # add base score to json file
    try:
        with open(output_file, "w") as f:
            f.write(repr(base_score) + "\n")
            with open(tmp_file) as f2:
                for line in f2.readlines():
                    f.write(line)
    except Exception as e:
        print("error: ", e)
        os.remove(tmp_file)
        return

    os.remove(tmp_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python convert_xgboost.py <input_xgboost_model_path> <output_gbdt_model_path>")
        print("--supported booster: gbtree")
        print("--supported objective: 'reg:linear', 'binary:logistic', 'reg:logistic', "
              "'binary:logitraw', 'multi:softmax', 'multi:softprob', 'rank:pairwise'")
        exit(1)
    convert(sys.argv[1], sys.argv[2])
