import pandas as pd
import glob
import os

def parse(file_path, out_path):
    with open(file_path) as f:
        lines = f.readlines()
        lines = [s.strip() for s in lines]

    df_dict = {
        'ratios': [],
        'sizes' : [],
        'compressing' : [],
        'decompressing' : [],
        'com_n_sync' : [],
        'sync' : [],
        'sync_list' : [],
    }
    for i in range(0, len(lines)):
        tmp = lines[i]
        # print(tmp)
        if "Compression rate" in tmp:
            df_dict['ratios'].append(tmp.split(" = ")[-1].strip())
        if "Message size" in tmp:
            df_dict['sizes'].append(tmp.split(" = ")[-1].strip())
        if " compressing" in tmp:
            df_dict['compressing'].append(tmp.split("|")[2].strip()[:-1])
        if " decompressing" in tmp:
            df_dict['decompressing'].append(tmp.split("|")[2].strip()[:-1])
        if "compress and sync" in tmp:
            df_dict['com_n_sync'].append(tmp.split("|")[2].strip()[:-1])
        if "sync time" in tmp:
            t = tmp.split("|")[2][:-1]
            df_dict['sync'].append(t.split(" ")[-1].strip())
        if "[Synctime]" in tmp:
            df_dict['sync_list'].append(tmp[12:-1].split(", "))
    #     all_logs.append(logs)
    df = pd.DataFrame.from_dict(df_dict, orient='index').T
    df.to_csv(out_path)
if __name__ == "__main__":
    file_paths = glob.glob("/home/aaa10078nj/Structured_Sparsification/DDL-Compression-Benchmark/logs/*/SUMMARY_*.txt")
    out_folder = "./logs/"
    for file_path in file_paths:
        basename = os.path.basename(file_path)[:-4] + "_parsed.csv"
        parse(file_path, os.path.join(out_folder, basename))
    

