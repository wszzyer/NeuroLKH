#!/bin/bash
data_dir="./data"
if [[ ! -z "$1" ]]
then
  data_dir="$1"
fi

for instance in $(ls "$data_dir"/raw_instance/* |grep val)
do
    instance_name=$(awk -F / "{print \$NF;}" <<< $instance)
    problem=$(awk -F _ "{print \$1}" <<< $instance_name)
    data_name=$(awk -F _ '{print $5"_"$6"_"$7}'  <<< $instance_name | awk -F . '{print $1}')
    if [[ ! -d "./result/$data_name" ]]
    then
        mkdir "./result/$data_name"
    # else
    #     continue
    fi
    python ./lade_CVRP_baseline.py --problem ${problem^^} --data_path $instance \
            --num_candidates 20 --work_dir ./evaluation/   \
            --output_file ./result/$data_name/"baseline.pkl" \
            --num_trials 30000 || exit $?;
done
