# python main.py configs/config_exp.cfg
# python postprocessing.py
python postprocessing.py bubble_223_kinematics.csv --start 11 --end 18 --mmperpix 0.0155  --window 101 --poly 3

# python createVideo.py /media/dgamdha/data/Dhruv/ISU/PhD/Projects/LEAP_HI/software/dataAnalysis/fluidFlow/results/fluidFlow6/version_02/bubbleOfInterest/subframes/frames -o my_movie.mp4 --fps 30
