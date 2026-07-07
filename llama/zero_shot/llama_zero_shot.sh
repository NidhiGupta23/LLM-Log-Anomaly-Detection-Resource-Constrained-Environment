set -e

# Run each Python file in sequence
python3 new_Zero_Shot.py --input ../../Test_500_no_label_sorted.log --max_new_tokens 10
python3 new_Zero_Shot.py --input ../../Test_500_no_label_sorted.log --max_new_tokens 25
python3 new_Zero_Shot.py --input ../../Test_500_no_label_sorted.log --max_new_tokens 50
python3 new_Zero_Shot.py --input ../../Test_500_no_label_sorted.log --max_new_tokens 75
python3 new_Zero_Shot.py --input ../../Test_500_no_label_sorted.log --max_new_tokens 100
