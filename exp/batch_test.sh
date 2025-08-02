#!/bin/bash
exp_name=$1
problem=$(awk -F _ "{print \$1;}" <<< $exp_name)
feat_type=$(awk -F _ "{print \$2;}" <<< $exp_name)
if [[ $feat_type == "featnone" ]]
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
result_dir="./result"
command_prefix=""
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
    --result_dir)
      result_dir="$2"
      shift 2
      ;;
    --dry_run)
      command_prefix="echo "
      shift 1
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

for data_name in $(ls "$data_dir"/raw_instance/ |grep geo -v)
do
    if [[ ! -d "$result_dir/$data_name" ]]
    then
        echo "Please run LKH for $data_name first!"
        exit 1
    fi
  if [[ ! -d "./saved/$exp_name" ]]
    then
        continue
    fi
    $command_prefix python ./lade_CVRP_test.py --problem CVRP --data_dir $data_dir/raw_instance/$data_name \
            --geo_path $data_dir/raw_instance/CVRP_geo_raw_scatter_$data_name.pkl \
            --model_path ./saved/$exp_name/$data_name/best.pth --device $device \
            --use_feats $use_feats --output_file $result_dir/$data_name/$exp_name".pkl" \
            --num_trials 10000 --num_candidates 10 || exit $?;
    rm -rf ./evaluation/$data_name/
done
