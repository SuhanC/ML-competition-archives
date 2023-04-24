#!/usr/bin/sh
python preproc_dat44.py
python main_lgbm.py
python GenerateSubmission.py

