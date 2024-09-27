#!/bin/bash
device="cuda:0"
if [[ ! -z "$2" ]]
then
    device=$2
fi
if [[ $1 == "baseline"* ]]
then
    use_feats="sssp"
elif [[ $1 == "featall"* ]]
then
    use_feats="nodeheat od spacesyntax"
fi

for instance in $(ls ./data/raw_instance/* |grep val)
do
    if [[ ! -d "./result/$1" ]]
    then
        mkdir "./result/$1"
    fi
    instance_name=$(awk -F _ "{print \$6\"_\"\$7\"_\"\$8;}" <<< $instance)
    python ./lade_CVRP_test.py --file_path $instance --model_path ./saved/$1/best.pth --device $device \
            --use_feats $use_feats --output ./result/$1/${instance_name/%pkl/txt}
done
