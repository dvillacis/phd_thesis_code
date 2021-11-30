
#! /bin/zsh

#conda activate bilevel

join_arr() {
    local IFS="$1"
    shift
    echo "$*"
}

for reg_parameter in 0.00001
do
echo "Executing alpha initialization with alpha=$reg_parameter..."
out_dir="alpha_initialization_reg_parameter/$reg_parameter/"
python ../regularization_learning.py ../datasets/cameraman_128_5/filelist.txt -t scalar -i $reg_parameter -v -o $out_dir

info=($reg_parameter)
input="$out_dir/summary.out"
while read -r line 
do
contents=(${line//:/ })
info+=${contents[1]}
echo $info
done < "$input"
echo "${info[@]}" >> "alpha_initialization_reg_parameter/summary_table.csv"
done
