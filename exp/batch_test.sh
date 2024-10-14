#!/bin/bash
device="cuda:0"
if [[ ! -z "$2" ]]
then
    device=$2
fi

problem=$(awk -F _ "{print \$1;}" <<< $1)
feat_type=$(awk -F _ "{print \$2;}" <<< $1)
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

for instance in $(ls ./data/raw_instance/* |grep val)
do
    instance_problem=$(echo $instance | awk -F / "{print \$4;}" | awk -F _ "{print \$1}")
    if [[ $instance_problem != ${problem^^} ]]
    then
        continue
    fi
    if [[ ! -d "./result/$1" ]]
    then
        mkdir "./result/$1"
    else
        rm -rf "./result/$1/*"
    fi
    city=$(awk -F _ "{print \$6;}" <<< $instance)
    instance_name=$(awk -F _ "{print \$6\"_\"\$7\"_\"\$8;}" <<< $instance)
    python ./lade_CVRP_test.py --problem ${problem^^} --file_path $instance --model_path ./saved/$1/$city/best.pth --device $device \
            --use_feats $use_feats --output ./result/$1/${instance_name/%pkl/txt} \
            --lkh_trials 3000 --neurolkh_trials 5000
done
