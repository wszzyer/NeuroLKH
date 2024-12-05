#!/bin/bash
if [[ -z "$1" ]]
then
    echo "Usage: $0 <exp_name> [--device device] [--data_dir data_dir]"
    # exp_name is an arbitary format which is hard to use and should be blamed.
    # Some useless samples: cvrp_baseline_01, cvrptw_featall_005
    exit 1
else
    exp_name=$1
    problem=$(awk -F _ "{print \$1;}" <<< $exp_name)
    feat_type=$(awk -F _ "{print \$2;}" <<< $exp_name)
    ramuda=0.$(awk -F _ "{print \$3;}" <<< $exp_name)
    if [[ -z $problem || -z $feat_type || -z $ramuda ]]
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
shift;

device="cuda:0"
tmux_session_name=$(tmux display-message -p "#S")
if [[ tmux_session_name==lkh_* ]]
then
    device=cuda:$(($(awk -F _ "{print \$2;}" <<< $tmux_session_name) - 1))
fi
data_dir="./data"
batch_size=16
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
    --batch_size)
      batch_size="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

for train_instance in $(ls $data_dir/generated/* |grep train)
do    
    train_file_name=$(echo $train_instance | awk -F / "{print \$NF;}")
    instance_problem=$(awk -F _ "{print \$1}" <<< $train_file_name)
    if [[ $instance_problem != ${problem^^} ]]
    then
        continue
    fi
    val_instance=${train_instance/train/val}
    data_name=$(awk -F _ '{print $4"_"$6"_"$7}' <<< $train_file_name | awk -F . '{print $1}')
    if [[ ! -d "./saved/$exp_name" ]]
    then
        mkdir "./saved/$exp_name"
    else
        rm -rf "./saved/$exp_name/*"
    fi
    python ./lade_CVRP_train.py \
        --problem ${problem^^} \
        --file_path $train_instance --eval_file_path $val_instance \
        --save_dir ./saved/$exp_name/$data_name --ramuda $ramuda \
        --use_feats $use_feats --device $device \
        --batch_size $batch_size --save_interval 50;
done
