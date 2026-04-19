ore/
├─ pipeline/
│  ├─ build_pseudo_los.py
│  ├─ adaptive_measurement_noise.py
│  ├─ imm_ukf_estimator.py
│  ├─ risk_prediction_from_ukf.py
│  ├─ calculate_collision_probability.py
│  └─ active_sensing_plan_one_sample.py
│
├─ sim/
│  ├─ core_orekit_sim.py
│  ├─ apply_active_micro_dv_to_truth.py
│  ├─ nomove_render_relative_sequence.py
│  └─ batch_validate_exp3_hard_fov.py
│
├─ data_tools/
│  ├─ generate_dataset.py
│  └─ generate_risky_samples.py
│
├─ assets/
│  ├─ target.obj
│  └─ orekit-data.zip
│
└─ outputs/
   ├─ renders/
   ├─ dataset_rtn/
   ├─ dataset_risky/
   ├─ dataset_active/
   └─ validation_runs/