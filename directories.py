import os
from os.path import join
import re
import pathlib as pl

class directories:
    def __init__(self, baseDirPath, versionDirTemplate, versionDirIndex, dirs_wrtVersionDir):
        self.baseDirPathObj     = pl.Path(baseDirPath)
        self.versionDirTemplate = versionDirTemplate     # template: 'version_{:02d}'
        self.versionDirIndex    = self.updateTemplateIndex(versionDirIndex)      # Update function to be compatible with pathlib library
        self.versionDirName     = self.versionDirTemplate.format(self.versionDirIndex)
        self.versionDirPathObj  = self.baseDirPathObj / self.versionDirName
        self.pathDict           = dict.fromkeys(dirs_wrtVersionDir)       # Key: directory name, Value: pathlib object of directory
        self.generateDirectories()
    
    def updateTemplateIndex(self, versionIndex):
        if versionIndex <= 0:
            versionIndex = 0
            while True:
                versionIndex += 1
                if not (self.baseDirPathObj / self.versionDirTemplate.format(versionIndex)).exists():
                    break
        return versionIndex
                 
    def generateDirectories(self):
        # Create the version directory
        self.versionDirPathObj.mkdir(parents=True, exist_ok=True)
        
        # Create the directories inside version directory
        keys = self.pathDict.keys()
        for subDirName in keys:
            self.pathDict[subDirName] = self.versionDirPathObj / subDirName
            self.pathDict[subDirName].mkdir(parents=True, exist_ok=True)
    
    def getPathDict(self):
        return self.pathDict

    def getPathDictKeys(self):
        return self.pathDict.keys()
    
    def getVerDirName(self):
        return self.versionDirName
    
    def getVerDirPathObj(self):
        return self.versionDirPathObj
    
    def getBaseDirPathObj(self):
        return self.baseDirPathObj

    def getDirPathObj(self, key):
        if key == '__base__':
            return self.baseDirPathObj
        if key == '__template__':
            return self.versionDirPathObj
        if key in self.pathDict.keys():
            return self.pathDict[key]
        else:
            print('Invalid key')
            return
    
    def addDir_usingKey(self, wrtKey, key):
        if wrtKey != '__base__' or wrtKey != '__template__':
            print('Invalid wrtKey')
            return
        
        if key in self.pathDict.keys():
            print('Key already exists')
            return
        
        if wrtKey == '__base__':
            self.pathDict[key] = self.baseDirPathObj / key
        if wrtKey == '__template__':
            self.pathDict[key] = self.versionDirPathObj / key
            
        self.pathDict[key].mkdir(parents=True, exist_ok=True)