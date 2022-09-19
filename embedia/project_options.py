# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 12:11:24 2021

@author: cesar
"""

class BinaryBlockSize:
    (Bits8, Bits16, Bits32) = (0,1,2)

class ModelDataType:
    (FLOAT, FIXED32, FIXED16, FIXED8, BINARY) = (0,1,2,3,4)
    
class ProjectType:
    (C, CPP ,ARDUINO, CODEBLOCK) = (0,1,2,3)
    
class ProjectFiles:
    (LIBRARY, MAIN, MODEL) = (1, 2, 4)
    (ALL) = {LIBRARY, MAIN, MODEL}
    
class DebugMode:
    (DISCARD, DISABLED, HEADERS, DATA) = (-1,0,1,2)
    
class ProjectOptions:
    project_type = ProjectType.C
    data_type = ModelDataType.FLOAT 
    example_data = None
    example_name = ''
    example_comment = ''
    files = ProjectFiles.ALL
    debug_mode = DebugMode.DISABLED
    tamano_bloque = BinaryBlockSize.Bits8
    

    
