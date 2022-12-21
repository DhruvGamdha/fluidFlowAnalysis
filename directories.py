import os
from os.path import join
import re

class directories:
    def __init__(self, baseDirPath, dirNames, versionIndex):
        self.baseDirPath    = baseDirPath
        self.dirNames       = dirNames
        self.dirPaths       = {}
        self.verDirName     = self.getVersionDirName(self.dirNames[0], versionIndex)
        self.createDirs()
    
    def getVersionDirName(self, template, versionIndex):
        if versionIndex == -1:
            lenFolderName   = len(template)
            dirlist         = [int(item[lenFolderName:]) for item in os.listdir(self.baseDirPath) if os.path.isdir(os.path.join(self.baseDirPath,item)) and re.search(template, item) != None and len(item)> lenFolderName] 
            versionIndex    = 1
            if len(dirlist) != 0:
                versionIndex = max(dirlist) + 1
        
        latestVersion   = versionIndex
        versionDirName  = template + str(latestVersion)            
        return versionDirName
                
    def createDirs(self):
        self.dirPaths[self.dirNames[0]] = self.createDirectory(self.baseDirPath, self.verDirName)   # Create version directory
        for subDirName in self.dirNames[1:]:                                                        # Create sub directories wrt version directory
            self.dirPaths[subDirName] = self.createDirectory(self.dirPaths[self.dirNames[0]], subDirName)
            
    def getDirPaths(self):
        return self.dirPaths

    def getDirNames(self):
        return self.dirNames

    def createDirectory(self, path, directoryName):
        directory = join(path, directoryName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory