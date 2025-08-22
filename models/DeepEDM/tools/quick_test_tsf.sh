prefix="rerun"

dataset=$1
seed=$2
exp_idx=$3

if [ "$dataset" = "ETTh1" ]; then
    bash scripts/long_term_forecast/ETT_script/DeepEDM_ETTh1.sh $prefix $seed $exp_idx > rerun_2021_etth1_${exp_idx}.out
elif [ "$dataset" = "ETTh2" ]; then
    bash scripts/long_term_forecast/ETT_script/DeepEDM_ETTh2.sh $prefix $seed $exp_idx > rerun_2021_etth2_${exp_idx}.out
elif [ "$dataset" = "ETTm1" ]; then
    bash scripts/long_term_forecast/ETT_script/DeepEDM_ETTm1.sh $prefix $seed $exp_idx > rerun_2021_ettm1_${exp_idx}.out
elif [ "$dataset" = "ETTm2" ]; then
    bash scripts/long_term_forecast/ETT_script/DeepEDM_ETTm2.sh $prefix $seed $exp_idx > rerun_2021_ettm2_${exp_idx}.out
elif [ "$dataset" = "Weather" ]; then
    bash scripts/long_term_forecast/Weather_script/DeepEDM.sh $prefix $seed $exp_idx > rerun_2021_weather_${exp_idx}.out
elif [ "$dataset" = "ILI" ]; then
    bash scripts/long_term_forecast/ILI_script/DeepEDM.sh $prefix $seed $exp_idx > rerun_2021_ili_${exp_idx}.out
elif [ "$dataset" = "Exchange" ]; then
    bash scripts/long_term_forecast/Exchange_script/DeepEDM.sh $prefix $seed $exp_idx > rerun_2021_exchange_${exp_idx}.out
elif [ "$dataset" = "ECL" ]; then
    bash scripts/long_term_forecast/ECL_script/DeepEDM.sh $prefix $seed $exp_idx > rerun_2021_ecl_${exp_idx}.out
elif [ "$dataset" = "Traffic" ]; then
    bash scripts/long_term_forecast/Traffic_script/DeepEDM.sh $prefix $seed $exp_idx > rerun_2021_traffic_${exp_idx}.out
fi