import os
from os.path import join
import re
# from utils import createDir

class directories:
    def __init__(self, baseDirPath, versionDirName, subDirNames):
        self.baseDirPath    = baseDirPath
        self.versionDirName = versionDirName
        self.verDirPath     = None
        self.subDirNames    = subDirNames
        self.subDirPaths    = {}
        self.createVersionDirectory()
        self.createSubDirs()

    def createVersionDirectory(self):
        ResFolderName   = self.versionDirName + '_' # example: 'version_'
        lenFolderName   = len(ResFolderName)
        dirlist         = [int(item[lenFolderName:]) for item in os.listdir(self.baseDirPath) if os.path.isdir(os.path.join(self.baseDirPath,item)) and re.search(ResFolderName, item) != None and len(item)> lenFolderName] 
        
        if len(dirlist) != 0:
            latestVersion = max(dirlist)
        else:
            latestVersion = 0

        self.verDirPath = createDirectory(self.baseDirPath, ResFolderName + str(latestVersion + 1))
        # print(ResFolderName + str(latestVersion + 1) + '\n')
    
    def createSubDirs(self):
        for subDirName in self.subDirNames:
            self.subDirPaths[subDirName] = createDirectory(self.verDirPath, subDirName)

    def getSubDirPaths(self):
        return self.subDirPaths
    
    def getVerDirPath(self):
        return self.verDirPath

def createDirectory(path, directoryName):
        print("path: ", path)
        print('Creating directory: ', directoryName)
        import os
        from os.path import join
        directory = join(path, directoryName)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        print('Directory created: ', directory)
        print('Dir type: ', type(directory))
        return directory