from ConfigParser import ConfigParser
import os

this_path = os.path.dirname(os.path.abspath(__file__))

def parse():
	parser = ConfigParser()
	parser.read(this_path + '/config.ini')
	return parser