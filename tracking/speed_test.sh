echo "########## Profiling STARK-S50 model ##########"
python tracking/profile_model.py --script stark_s --config baseline
echo "########## Profiling STARK-ST50 model ##########"
python tracking/profile_model_ST.py --script stark_st2 --config baseline
echo "########## Profiling STARK-ST101 model ##########"
python tracking/profile_model_ST.py --script stark_st2 --config baseline_R101
