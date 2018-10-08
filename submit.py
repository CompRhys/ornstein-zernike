import os 
import numpy as np

def main():
	path = os.path.expanduser('~')
	input_path = path+'/Liquids/data/input'
	files = os.listdir(input_path)
	print(len(files))
	for i in range(len(files)):
		print('ding', i)
		index = np.random.randint(0, len(files))
		print(files[index])
		# os.system("sbatch "+path+"/Liquids/random_test "+)
		
if __name__ == "__main__":
    main()