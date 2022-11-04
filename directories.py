import os
from os.path import join
import re

class directories:
    def __init__(self, baseDirPath, versionDirTemplate, subDirNames, newResultsDir=True):
        self.baseDirPath    = baseDirPath
        self.subDirNames    = subDirNames
        self.subDirPaths    = {}
        
        if newResultsDir:
            ''' 
            Create a new results directories
            '''
            self.verDirPath         = None
            self.versionDirTemplate = versionDirTemplate
            self.createVersionDirectory()
        else:
            ''' 
            Use the existing results directory, sub directories are created if they do not exist already
            ''' 
            if not os.path.exists(os.path.join(baseDirPath, versionDirTemplate)):      # Check if the results directory exists
                exit("ERROR: Results directory does not exist")
            else:
                self.verDirPath = join(baseDirPath, versionDirTemplate)
        
        self.createSubDirs()
                
    def createVersionDirectory(self):
        ResFolderName   = self.versionDirTemplate + '_' # example: 'version_'
        lenFolderName   = len(ResFolderName)
        dirlist         = [int(item[lenFolderName:]) for item in os.listdir(self.baseDirPath) if os.path.isdir(os.path.join(self.baseDirPath,item)) and re.search(ResFolderName, item) != None and len(item)> lenFolderName] 
        
        if len(dirlist) != 0:
            latestVersion = max(dirlist)
        else:
            latestVersion = 0

        self.verDirPath = createDirectory(self.baseDirPath, ResFolderName + str(latestVersion + 1))
    
    def createSubDirs(self):
        for subDirName in self.subDirNames:
            self.subDirPaths[subDirName] = createDirectory(self.verDirPath, subDirName)
            
    def getSubDirPaths(self):
        return self.subDirPaths
    
    def getVerDirPath(self):
        return self.verDirPath

def createDirectory(path, directoryName):
        import os
        from os.path import join
        directory = join(path, directoryName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory