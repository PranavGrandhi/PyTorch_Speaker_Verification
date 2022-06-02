from random import randrange, choice
from pathlib import Path
from typing import List, Iterator
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, stack, concat
from torch import load as torch_load
from numpy import load as numpy_load


def reservoir_sampling(stream : Iterator, k : int, n : int = None):

	'''
	An efficient Python3 program to randomly select k items
	from a stream of n items, where n is very large 
	
	Author: Lim Zhi Hao		Date: 2018-06-01
	
	'''

	reservoir = []
	for i, s in enumerate(stream):

		if i < k:
			# Initialize with first k elements from stream
			reservoir.append(s)

		elif i >= k and i < n: 
			# After that, Pick a random index from 0 to i
			j = randrange(i+1)

			# If the randomly picked index is smaller than k,
			# then replace the element present at the index
			# with new element from stream
			if j < k:
				reservoir[j] = s

	return reservoir


class directory():

	def __init__(self, dirpath : Path, ext : str = None):

		''' Main directory '''

		if not isinstance(dirpath, Path):
			dirpath = Path(dirpath)

		self.dirpath = dirpath
		self.ext = ext

		if self.ext:
			# If extension is specified, only 
			# search for files with extension
			self.condition = lambda F: str(F).endswith(self.ext)
		else:
			# Otherwise, only list folders in directory
			self.condition = lambda F: F.is_dir()
		
		self.count_files()	

	def iterate_files(self):
		for FILE in self.dirpath.iterdir():
			if self.condition(FILE):
				yield FILE

	def count_files(self):
		''' Count number of files in directory '''
		
		num_files = 0
		for _ in self.iterate_files():
			num_files += 1
			
		self.num_files = num_files


class class_directory(directory):

        def __init__(self, dirpath : Path, ext : str = 'wav'):

                ''' Class directory '''
                
                directory.__init__(self, dirpath = dirpath, ext = ext)


        def random_pick(self, k : int = 1):

                ''' Pick k files from directory '''

                stream = self.iterate_files()
                n = self.num_files

                if n < k:
                        return list(stream)
                else:
                        return reservoir_sampling(stream, n = n, k = k)



class torch_dataset(Dataset):

	def __init__(self, 
		torch_dirpath : Path, 
		ext : str = 'pt',
		mini_batch_size : int = 5,
		to_shuffle : bool = True):
		
		self.torch_dirpath = torch_dirpath
		self.ext = ext
		self.mini_batch_size = mini_batch_size
		self.to_shuffle = to_shuffle

		self.init_classes()

	def init_classes(self):
		class_folders = []
		num_files = []
	
		self.torch_dir = directory(self.torch_dirpath, ext = None)
		for class_folder in self.torch_dir.iterate_files():
			class_folder = class_directory(class_folder, self.ext)
			class_folders.append(class_folder)
			num_files.append(class_folder.num_files)

		self.class_folders = class_folders
		self.num_files = num_files

	def __len__(self):

		return sum(self.num_files)

	''' if no shuffle check'''
	def __getitem__(self, idx : int = 0):

		if self.to_shuffle:
			class_folder = choice(self.class_folders)
		else:
			class_folder = self.class_folders[0]
		
		return class_folder.random_pick(k = self.mini_batch_size)


class collate_fn():

	def __init__(self, device : str = 'cpu', from_numpy : bool = False):

		self.device = device
		self.from_numpy = from_numpy

	def load(self, array_path : Path):

		if self.from_numpy:
			return from_numpy(numpy_load(array_path))
		else:
			return torch_load(array_path)


	def __call__(self, batch : List[List[Path]]):

		batch_array = []
		for class_samples in batch:
			class_array = []
			for class_sample in class_samples:
				array = self.load(class_sample)
				class_array.append(array)
				
			batch_array.append(stack(class_array))

		return concat(batch_array).to(self.device)


def main():

	
	torch_dirpath = Path('C:\\Users\\prana\\Desktop\\GE2E\\PyTorch_Speaker_Verification\\train_tisv_NoiseSplit')
	ext = 'npy'
	device = 'cpu'
	from_numpy = True
	to_shuffe = True
	batch_size = 4
	mini_batch_size = 32
	to_shuffle = True

	dataset = torch_dataset(torch_dirpath = torch_dirpath, 
				ext = ext,
				mini_batch_size = mini_batch_size,
				to_shuffle = to_shuffle)

	print(len(dataset))
	
	custom_collate = collate_fn(device = device, from_numpy = from_numpy)

	dataloader = DataLoader(dataset, 
				batch_size = batch_size, 
				shuffle = to_shuffle, 
				collate_fn = custom_collate)

	#for data in dataloader:
	#	print(data.shape)





if __name__ == '__main__':

	main()

	
