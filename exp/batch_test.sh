#!/bin/bash
exp_name=$1
problem=$(awk -F _ "{print \$1;}" <<< $exp_name)
feat_type=$(awk -F _ "{print \$2;}" <<< $exp_name)
if [[ $feat_type == "baseline" ]]
then
    use_feats="sssp"
elif [[ $feat_type == "featall" ]]
then
    use_feats="nodeheat od spacesyntax"
else
    echo "Please check your input."
    exit 2
fi
shift;

device="cuda:0"
data_dir="./data"
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      device="$2"
      shift 2
      ;;
    --data_dir)
      data_dir="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

for instance in $(ls "$data_dir"/raw_instance/* |grep val)
do
    instance_problem=$(echo $instance | awk -F / "{print \$NF;}" | awk -F _ "{print \$1}")
    if [[ $instance_problem != ${problem^^} ]]
    then
        continue
    fi
    data_name=$(awk -F _ '{print $6"_"$8"_"$9}' <<< $instance | awk -F . '{print $1}')
    if [[ ! -d "./result/$data_name" ]]
    then
        echo "Please run LKH for $data_name first!"
        exit 1
    fi
    python ./lade_CVRP_test.py --problem ${problem^^} --file_path $instance \
            --model_path ./saved/$exp_name/$data_name/best.pth --device $device \
            --use_feats $use_feats --output_file ./result/$data_name/$exp_name".pkl" \
            --num_trials 35000 || exit $?;
done
