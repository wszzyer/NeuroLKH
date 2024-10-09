#!/bin/bash
device="cuda:0"
if [[ -z "$1" ]]
then
    echo "Usage: $0 <exp_name> [device]"
    # exp_name is an arbitary format which is hard to use and should be blamed.
    # Some useless samples: cvrp_baseline_01_0, cvrptw_featall_005_01
    exit 1
else
    problem=$(awk -F _ "{print \$1;}" <<< $1)
    feat_type=$(awk -F _ "{print \$2;}" <<< $1)
    ramuda=0.$(awk -F _ "{print \$3;}" <<< $1)
    l_a=0.$(awk -F _ "{print \$4;}" <<< $1)
    if [[ -z $problem || -z $feat_type || -z $ramuda || -z $l_a ]]
    then
        echo "Please check your input."
        exit 2
    fi
    if [[ $feat_type == featall ]]
    then
        use_feats="od nodeheat spacesyntax"
    elif [[ $feat_type == baseline ]]
    then
        use_feats="sssp"
    else
        echo "Please check your exp_name. Allowed feat combination are: featall, baseline."
        exit 2
    fi
    if [[ $problem != "cvrp" && $problem != "cvrptw" ]]
    then
        echo "Please chech your exp_name. Allowed problems are: cvrp, cvrptw."
        exit 2
    fi
fi

tmux_session_name=$(tmux display-message -p "#S")
if [[ ! -z "$2" ]]
then
    device=$2
elif [[ tmux_session_name==lkh_* ]]
then
    device=cuda:$(($(awk -F _ "{print \$2;}" <<< $tmux_session_name) - 1))
fi

for train_instance in $(ls ./data/generated/* |grep train)
do
    val_instance=${train_instance/train/val}
    city=$(awk -F _ "{print \$4;}" <<< $train_instance)
    if [[ ! -d "./saved/$1" ]]
    then
        mkdir "./saved/$1"
    else
        rm -rf "./saved/$1/*"
    fi
    python ./lade_CVRP_train.py \
        --problem ${problem^^} \
        --file_path $train_instance \
        --eval_file_path $val_instance \
        --save_dir ./saved/$1/$city --ramuda $ramuda --l_a $l_a \
        --use_feats $use_feats --device $device \
        --save_interval 100;
done
