import numpy as np
import re
import os
from fileinput import filename
from pathlib import Path

def  read_cell(*args):#filename,linesep,cellsep):
	nargin  = len(args) 
	filename = args[0]
	if nargin < 2:
		linesep='\n'
	if nargin < 3:
		cellsep = '\t'
		
	print(filename)
	#return filename.pop()

	if os.path.isfile(filename):
    		fid = open(filename)
	else:
    		fid = filename
	# Assume that filename is either a file ide or a string

	with open(filename) as f:
    		lines = f.read().splitlines()

	return lines