
prefix="m4_exp"

bash scripts/short_term_forecast/DeepEDM_Daily.sh $prefix > m4_Daily.out &

bash scripts/short_term_forecast/DeepEDM_Hourly.sh $prefix > m4_Hourly.out &

bash scripts/short_term_forecast/DeepEDM_Quarterly.sh $prefix > m4_Quarterly.out &

bash scripts/short_term_forecast/DeepEDM_Monthly.sh $prefix > m4_Monthly.out &

bash scripts/short_term_forecast/DeepEDM_Weekly.sh $prefix > m4_Weekly.out &

bash scripts/short_term_forecast/DeepEDM_Yearly.sh $prefix > m4_Yearly.out