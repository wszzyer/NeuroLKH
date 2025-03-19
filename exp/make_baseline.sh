#!/bin/bash
data_dir="./data"
result_dir="./result"
if [[ ! -z "$1" ]]
then
  data_dir="$1"
fi
if [[ ! -z "$2" ]]
then
  result_dir="$2"
else
  exit 3
fi


for data_name in $(ls "$data_dir"/raw_instance/ |grep -v geo)
do
    if [[ ! -d "$result_dir"/"$data_name/" ]]
    then
        mkdir "$result_dir"/"$data_name/"
    # else
    #     continue
    fi
    python ./lade_CVRP_baseline.py --problem CVRP --data_dir "$data_dir"/raw_instance/$data_name \
            --num_candidates 20 --work_dir ./evaluation/   \
            --baselines lkh hgs \
            --output_file "$result_dir"/$data_name/"baseline.pkl" \
            --num_trials 30000 || exit $?;
done
