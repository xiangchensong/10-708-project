for dataset in cross_species GSM Norman2019_prep pachter
do
    for p in 0 1 2
    do
	    echo ${dataset} ${p}
        python test_neo.py --dataset=${dataset} --treatment_selection_bias=2.0 --dosage_selection_bias=2.0  --batch_size=128 --h_dim=64 --h_inv_eqv_dim=64 --alpha=1.0 --model_name=TransTEE --rep=5 --p=${p}
    done
done