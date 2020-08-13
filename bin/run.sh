cd ../src

# ===============================================================================
# lightgbm   
# ===============================================================================
# python train.py -m 'lgbm_001' -c 'test'
# python train.py -m 'lgbm_002' -c 'lr=0.1'
# python train.py -m 'lgbm_003' -c 'add oof'
# python train.py -m 'lgbm_004' -c 'custom_002'
# python train.py -m 'lgbm_005' -c 'custom_003, StratifiedKFold'
# python train.py -m 'lgbm_006' -c 'lgbmtuner'
# python train.py -m 'lgbm_007' -c 'seed=2021'
# python train.py -m 'lgbm_008' -c 'seed=2022'
# python train.py -m 'lgbm_009' -c 'seed=2023'
# python train.py -m 'lgbm_010' -c 'seed=2024'
# python train.py -m 'lgbm_011' -c 'seed=2025'
# python train.py -m 'lgbm_012' -c 'custom_004'
# python train.py -m 'lgbm_013' -c 'custom_005'
# python train.py -m 'lgbm_014' -c 'custom_006'
# python train.py -m 'lgbm_015' -c 'custom_007'
# python train.py -m 'lgbm_016' -c 'predict residual'
# python train.py -m 'lgbm_017' -c 'drop large_diff'
# python train.py -m 'lgbm_018' -c 'drop large_diff2'
# python train.py -m 'lgbm_019' -c 'seed=2021'
# python train.py -m 'lgbm_020' -c 'seed=2022'
# python train.py -m 'lgbm_021' -c 'seed=2023'
# python train.py -m 'lgbm_022' -c 'seed=2024'
# python train.py -m 'lgbm_023' -c 'seed=2025'
# python train.py -m 'lgbm_024' -c 'seed=2026'
# python train.py -m 'lgbm_025' -c 'seed=2027'
# python train.py -m 'lgbm_026' -c 'seed=2028'
# python train.py -m 'lgbm_027' -c 'seed=2029'


# ===============================================================================
# catboost   
# ===============================================================================
# python train.py -m 'catboost_001' -c 'test'
# python train.py -m 'catboost_002' -c 'lr=0.1'
# python train.py -m 'catboost_003' -c 'custom_003, StratifiedKFold'
# python train.py -m 'catboost_004' -c 'seed=2021'
# python train.py -m 'catboost_005' -c 'seed=2022'
# python train.py -m 'catboost_006' -c 'seed=2023'
# python train.py -m 'catboost_007' -c 'seed=2024'
# python train.py -m 'catboost_008' -c 'seed=2025'
# python train.py -m 'catboost_009' -c 'custom_007, drop large_diff2, seed=2025'


# ===============================================================================
# ensemble
# ===============================================================================
# python ensemble.py -m 'ensemble_001' -c 'test'
# python ensemble.py -m 'ensemble_002' -c '...'
# python ensemble.py -m 'ensemble_003' -c '...'
# python ensemble.py -m 'ensemble_004' -c '...'
# python ensemble.py -m 'ensemble_005' -c '...'
# python ensemble.py -m 'ensemble_006' -c '...'
# python ensemble.py -m 'ensemble_007' -c '...'
python ensemble.py -m 'ensemble_008' -c '...'