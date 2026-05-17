'''
File has fucntion that can be used to clean and get test and train datasets.
Uncomment function name in main to use
'''
import argparse

import pandas as pd 
from datetime import datetime

def convert_parquet_to_logFile(input_file, output_file):
# This function is used to convert parquet file to log file. It also removes the anomaly column if it exists.
    df = pd.read_parquet(input_file) 
    if "anomaly" in df.columns: 
        df = df.drop(columns=["anomaly"]) 
    df.to_csv(output_file, index=False, header=False, sep=" ")


def remove_empty_contents_row(input_file, output_file):
# This function is used to remove rows that have empty content and INFO level.
    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split()

            # Skip malformed lines
            if len(parts) < 6:
                continue

            level = parts[8]                 # level column
            content = " ".join(parts[9:])    # message/content

            # Keep only rows that are NOT (INFO + empty content)
            if not (level == "INFO" and content.strip() == ""):
                fout.write(line)

def separate_normal_abnormal_logs_fromBGL_file(input_file, normal_output_file, abnormal_output_file):
# This function is used to separate normal and abnormal logs from BGL file.
    normal_counter = 0
    abnormal_counter = 0
    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin, \
         open(normal_output_file, "w", encoding="utf-8") as normal_fout, \
         open(abnormal_output_file, "w", encoding="utf-8") as abnormal_fout:    
        for line in fin:
            if line.startswith("-"): 
                normal_counter = normal_counter + 1 
                normal_fout.write(line + "\n") 
            else: 
                abnormal_counter = abnormal_counter + 1 
                abnormal_fout.write(line + "\n") 
    print(normal_counter, abnormal_counter) 

def create_test_datasets(input_file, test_output_file, lines_in_file, seed):
# This function is used to create test datasets from the log file. It randomly samples lines from the log file and writes them to the test output file.
    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin:
        lines = fin.readlines()
    
    # Randomly sample lines for the test dataset
    sampled_lines = pd.Series(lines).sample(n=lines_in_file, random_state=seed).tolist()
    
    # Write the sampled lines to the test output file
    with open(test_output_file, "w", encoding="utf-8") as fout:
        fout.writelines(sampled_lines)



def drop_first_column_label(input_file, output_file):
# This function is used to drop the first column label from the log file. It is used for BGL log file which has a label column at the beginning of each line.
    with open(input_file, "r") as fin, open(output_file, "w") as fout: 
        for line in fin: 
            parts = line.strip().split() 
            if len(parts) > 1: 
                fout.write(" ".join(parts[1:]) + "\n") 



def parse_ts(ts):
# This function is used to parse the timestamp from the log file. It converts the timestamp string to a datetime object for sorting.
    return datetime.strptime(ts, "%Y-%m-%d-%H.%M.%S.%f")

def sort_logs_by_timestamp(input_file, output_file):
#This function is used to sort the log file by timestamp. It reads the log file, sorts the lines based on the timestamp, and writes the sorted lines to the output file.
    with open(input_file, "r") as f:
        lines = f.readlines()

    sorted_lines = sorted(lines, key=lambda x: int(x.split()[0]))

    with open(output_file, "w") as f:
        f.writelines(sorted_lines)

def compare_logs_in_normal(line, normal_file, counter_l0):
# This function is used to compare a log line with the normal log file. It checks if the log line is present in the normal log file and returns True if it is, otherwise False.
    with open(normal_file, "r") as f:
        for normal_line in f:
            if line.strip() == normal_line.strip():
                counter_l0 += 1
                return True
    return False

def compare_logs_in_abnormal(line, abnormal_file, counter_l1):
# This function is used to compare a log line with the abnormal log file. It checks if the log line is present in the abnormal log file and returns True if it is, otherwise False.
    with open(abnormal_file, "r") as f:
        for abnormal_line in f:
            if line.strip() == abnormal_line.strip():
                counter_l1 += 1
                return True
    return False

def cleanup_line(line):
    nl = line[23:].strip()
    if nl.endswith("}"):
        nl = nl[:-1]
    if nl.startswith('"') and nl.endswith('"'):
        nl = nl[1:-1]
    nl = nl.replace('\\"', '"')
    return nl

def read_results_and_evaluate(results_file, normal_file, abnormal_file):
# This function is used to read the results from the results file and evaluate the performance of the model. It compares the predicted labels with the true labels from the normal and abnormal log files and calculates the accuracy, precision, recall, and F1 score.
    counter_l0 = 0
    counter_l1 = 0
    l0 = 0
    l1 = 0
    with open(results_file, "r") as f:
        for line in f:

            if line[11] == "0":
                l0 += 1
                cl = cleanup_line(line)
                if compare_logs_in_normal(cl, normal_file, counter_l0):
                    counter_l0 += 1
            elif line[11] == "1":
                l1 += 1
                cl = cleanup_line(line)
                if compare_logs_in_abnormal(cl, abnormal_file, counter_l1):
                    counter_l1 += 1
            else:
                print("Log line not found in either normal or abnormal log files:", line.strip())
    print("Counter L0 (Normal):", counter_l0)
    print("Counter L1 (Abnormal):", counter_l1)



def main():
    parser = argparse.ArgumentParser(description="For cleaning up and setting up testing datasets")
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--output_log_file", type=str, required=True)
    parser.add_argument("--lines_in_file", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal_output_file", type=str, default="normal_logs_BGL.log")
    parser.add_argument("--abnormal_output_file", type=str, default="abnormal_logs_BGL.log")
    
    args = parser.parse_args()


    convert_parquet_to_logFile(args.log_file, args.output_log_file)
    #remove_empty_contents_row(args.log_file, args.output_log_file)
    #separate_normal_abnormal_logs_fromBGL_file(args.output_log_file, args.normal_output_file, args.abnormal_output_file)
    #create_test_datasets(args.output_log_file, "Test_500.log", args.lines_in_file, args.seed)
    #drop_first_column_label("Test_500.log", "Test_500_no_label.log")
    #sort_logs_by_timestamp("Test_500_no_label.log", "Test_500_no_label_sorted.log")
    #read_results_and_evaluate(args.output_log_file, args.normal_output_file, args.abnormal_output_file)

if __name__ == "__main__":
    main()
