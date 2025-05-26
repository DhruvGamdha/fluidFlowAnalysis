# python main.py configs/config_exp.cfg

RUN_DIR='/media/dgamdha/data/Dhruv/ISU/PhD/Projects/LEAP_HI/software/dataAnalysis/fluidFlow/results/fluidFlow6/version_04'
KIN_FILE='bubble_223_kinematics.csv'
ELLIPSE_FILE='bubble_223_ellipse.csv'
TIME_START=0.323
TIME_END=0.550
MM_PER_PIX=0.0155

cp $RUN_DIR/analysis/bubbleTracking/kinematics/$KIN_FILE .
cp $RUN_DIR/analysis/bubbleTracking/ellipse/$ELLIPSE_FILE .

python postprocessing.py $KIN_FILE --ellipse $ELLIPSE_FILE --start $TIME_START --end $TIME_END --mmperpix $MM_PER_PIX --window 101 --poly 3

# python createVideo.py /media/dgamdha/data/Dhruv/ISU/PhD/Projects/LEAP_HI/software/dataAnalysis/fluidFlow/results/fluidFlow6/version_02/bubbleOfInterest/subframes/frames -o my_movie.mp4 --fps 30
