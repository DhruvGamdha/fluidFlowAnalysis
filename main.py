import sys
import logging
import libconf
import git
from pathlib import Path

from bubbleAnalysis import BubbleAnalysis
from directories import directories, updateTemplateIndex

def load_parameters(config_path: str):
    """
    Load configuration parameters from the given config file.
    Exits the program if the file is not found.
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"ERROR: Configuration file {config_path} does not exist.")
        sys.exit(1)
    with open(cfg_path, 'r') as f:
        para = libconf.load(f)
    return para

def get_git_commit_hash():
    """
    Retrieve the current git commit hash of the repository, if available.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return None

def save_config(para, outpDirObj):
    """
    Save the current configuration dictionary to a new config file in the output template directory.
    """
    cfgTempIndex = updateTemplateIndex(
        outpDirObj.getDirPathObj('__template__'),
        para['cfgFileTemplate'],
        -1
    )
    config_file_path = outpDirObj.getDirPathObj('__template__') / para['cfgFileTemplate'].format(cfgTempIndex)
    with open(config_file_path, 'w') as f:
        libconf.dump(para, f)
    logging.info(f"Configuration saved to {config_file_path}")

def configure_logging(outpDirObj):
    """
    Configure logging to output both to console and a file in the output directory.
    """
    log_file_path = outpDirObj.getDirPathObj('__template__') / "analysis.log"
    
    # Clear any existing handlers
    logging.getLogger().handlers = []
    
    # Set logger level
    logging.getLogger().setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)
    
    logging.info("========================================")
    logging.info(f"Logging configured. Writing logs to {log_file_path}")

def run_analysis(para):
    """
    Execute the full bubble analysis workflow based on the given parameters.
    """

    # Create input directories
    inpDirObj = directories(
        para['inpPth_base'],
        para['inpTemplate'],
        para['inpTemplateIndex'],
        para['inpDirsToCreate_wrtTemplate']
    )

    # Add additional directories if specified
    additional_dirs = para.get('additionalDirs', [])
    if len(additional_dirs) == 2:
        inpDirObj.addDir_usingKey(additional_dirs[0], additional_dirs[1])

    # Create output directories
    outpDirObj = directories(
        para['outpPth_base'],
        para['outpTemplate'],
        para['outpTemplateIndex'],
        para['outpDirsToCreate_wrtTemplate']
    )

    # Configure logging after output directory is created
    configure_logging(outpDirObj)

    # Add git commit hash to parameters if available
    sha = get_git_commit_hash()
    if sha:
        para['gitCommitHash'] = sha

    # Save current configuration in output directory
    save_config(para, outpDirObj)

    # Initialize the analysis object
    analysis = BubbleAnalysis(para)

    # =========== Analysis Steps ===========
    logging.info("Starting analysis workflow...")
    analysis.getFramesFromVideo(inpDirObj.getDirPathObj('original/frames'))
    analysis.getCroppedFrames(inpDirObj.getDirPathObj('original/frames'), inpDirObj.getDirPathObj('all/frames'))
    analysis.createVideoFromFrames(inpDirObj.getDirPathObj('all/frames'))

    analysis.getBinaryImages(inpDirObj.getDirPathObj('all/frames'), outpDirObj.getDirPathObj('binary/all/frames'))
    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('binary/all/frames'))

    analysis.createConcatVideo(
        inpDirObj.getDirPathObj('all/frames'),
        outpDirObj.getDirPathObj('binary/all/frames'),
        outpDirObj.getDirPathObj('binary/all/frames').parent
    )

    analysis.extractFrameObjects(
        outpDirObj.getDirPathObj('binary/all/frames'),
        outpDirObj.getDirPathObj('analysis/pixSize/frames').parents[1]
    )

    analysis.evaluateBubbleTrajectory(
        outpDirObj.getDirPathObj('analysis/bubbleTracking/frames').parents[1]
    )
    
    analysis.plotBubbleKinematics(
        outpDirObj.getDirPathObj('analysis/bubbleTracking/kinematics')
    )

    analysis.checkVideoFilesOnDisk(
        outpDirObj.getDirPathObj('analysis/bubbleTracking/frames').parents[1]
    )

    analysis.markBubblesOnFrames(
        outpDirObj.getDirPathObj('binary/all/frames'),
        outpDirObj.getDirPathObj('analysis/bubbleTracking/frames')
    )

    analysis.createVideoFromFrames(outpDirObj.getDirPathObj('analysis/bubbleTracking/frames'))

    analysis.createConcatVideo(
        inpDirObj.getDirPathObj('all/frames'),
        outpDirObj.getDirPathObj('analysis/bubbleTracking/frames'),
        outpDirObj.getDirPathObj('analysis/bubbleTracking/frames').parent
    )

    logging.info("Analysis completed successfully.")

def main():
    # Read config file name from command line
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)
    para = load_parameters(sys.argv[1])
    run_analysis(para)

if __name__ == "__main__":
    main()
