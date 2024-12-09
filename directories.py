import pathlib as pl

class directories:
    """
    A class to handle directory structure creation and retrieval.
    Uses a template and a version index to create versioned output directories.
    """
    def __init__(self, baseDirPath, versionDirTemplate, versionDirIndex, dirs_wrtVersionDir):
        self.baseDirPathObj     = pl.Path(baseDirPath)
        self.versionDirTemplate = versionDirTemplate     # e.g. 'version_{:02d}'
        self.versionDirIndex    = updateTemplateIndex(self.baseDirPathObj, self.versionDirTemplate, versionDirIndex)
        self.versionDirName     = self.versionDirTemplate.format(self.versionDirIndex)
        self.versionDirPathObj  = self.baseDirPathObj / self.versionDirName

        # Create a dictionary of directories keyed by their relative paths
        self.pathDict = {dname: None for dname in dirs_wrtVersionDir}
        self.generateDirectories()
                 
    def generateDirectories(self):
        # Create the version directory if it doesn't exist
        self.versionDirPathObj.mkdir(parents=True, exist_ok=True)
        
        # Create the directories inside the version directory
        for subDirName in self.pathDict.keys():
            self.pathDict[subDirName] = self.versionDirPathObj / subDirName
            self.pathDict[subDirName].mkdir(parents=True, exist_ok=True)
    
    def getPathDict(self):
        return self.pathDict

    def getPathDictKeys(self):
        return list(self.pathDict.keys())
    
    def getVerDirName(self):
        return self.versionDirName
    
    def getVerDirPathObj(self):
        return self.versionDirPathObj
    
    def getBaseDirPathObj(self):
        return self.baseDirPathObj

    def getDirPathObj(self, key):
        """
        Retrieve a Path object for a given directory key.
        Special keys:
          '__base__' -> base directory
          '__template__' -> version directory
        """
        if key == '__base__':
            return self.baseDirPathObj
        if key == '__template__':
            return self.versionDirPathObj
        if key in self.pathDict:
            return self.pathDict[key]
        else:
            print('Invalid key')
            return None
    
    def addDir_usingKey(self, wrtKey, key):
        """
        Add a new directory key, either relative to base or template directory.
        """
        if wrtKey not in ['__base__', '__template__']:
            print('Invalid wrtKey. Must be "__base__" or "__template__"')
            return
        
        if key in self.pathDict:
            print('Key already exists')
            return
        
        base_path = self.baseDirPathObj if wrtKey == '__base__' else self.versionDirPathObj
        self.pathDict[key] = base_path / key
        self.pathDict[key].mkdir(parents=True, exist_ok=True)
        
def updateTemplateIndex(baseDirPathObj, versionDirTemplate, versionIndex):
    """
    Determine the version index for creating a new directory.
    If versionIndex <= 0, automatically find the next available version index.
    """
    if versionIndex <= 0:
        versionIndex = 0
        while True:
            versionIndex += 1
            if not (baseDirPathObj / versionDirTemplate.format(versionIndex)).exists():
                break
    return versionIndex
