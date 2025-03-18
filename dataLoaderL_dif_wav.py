'''
DataLoader for training and validating
'''

import torch
import numpy as np
import soundfile as sf
from preprocess.melspec.compute_mel import mel_spectrogram

class train_loader(object):
	def __init__(self, train_path, **kwargs):
		self.trn_file = train_path

		self.trn_ref_file_list, self.trn_test_file_list, self.trn_label_list = [], [], []
		if self.trn_file:
			with open(self.trn_file, 'r') as ft:
				file_list = ft.readlines()
			for i in range(len(file_list)):
				label, ref_wav_path, test_wav_path = file_list[i][:-1].split('\t')
				self.trn_ref_file_list.append(ref_wav_path)
				self.trn_test_file_list.append(test_wav_path)
				self.trn_label_list.append(label)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		ref_wav, ref_sr = sf.read(self.trn_ref_file_list[index])
		test_wav, test_sr = sf.read(self.trn_test_file_list[index])

		assert ref_sr == 16000 and test_sr == 16000, 'should keep wav sr=16000'

		# fixed length: 2s
		if len(ref_wav) <= 2 * ref_sr:
			len_left = int((2 * ref_sr - len(ref_wav)) // 2)
			len_right = 2 * ref_sr - len(ref_wav) - len_left
			ref_wav = np.pad(ref_wav, ((len_left, len_right)), 'edge')
		else:
			random_index = np.random.randint(0, len(ref_wav) - 2 * ref_sr)
			ref_wav = ref_wav[random_index:random_index + 2 * ref_sr]
		
		if len(test_wav) <= 2 * ref_sr:
			len_left = int((2 * test_sr - len(test_wav)) // 2)
			len_right = 2 * test_sr - len(test_wav) - len_left
			test_wav = np.pad(test_wav, ((len_left, len_right)), 'edge')
		else:
			random_index = np.random.randint(0, len(test_wav) - 2 * test_sr)
			test_wav = test_wav[random_index:random_index + 2 * test_sr]


		ref_spec = mel_spectrogram(torch.FloatTensor(ref_wav).unsqueeze(0),
                                        n_fft=1024, num_mels=80, sampling_rate=ref_sr, hop_size=160, win_size=1024, 
                                        fmin=0, fmax=8000, center=False)
		test_spec = mel_spectrogram(torch.FloatTensor(test_wav).unsqueeze(0),
                                        n_fft=1024, num_mels=80, sampling_rate=test_sr, hop_size=160, win_size=1024, 
                                        fmin=0, fmax=8000, center=False)



		# ref_spec = np.load(self.trn_ref_file_list[index])			# shape: [80, T]
		# test_spec = np.load(self.trn_test_file_list[index])
		# # segment length: 2s for sr=16k, i.e., 200 frames
		# if ref_spec.shape[1] > 200:
		# 	random_index = np.random.randint(0, ref_spec.shape[1] - 200)
		# 	ref_spec = ref_spec[:, random_index: random_index + 200]
		# else:
		# 	len_left = (200 - ref_spec.shape[1]) // 2
		# 	len_right = 200 - ref_spec.shape[1] - len_left
		# 	ref_spec = np.pad(ref_spec, ((0, 0), (len_left, len_right)), 'reflect')
		# 	# print(1)

		# if test_spec.shape[1] > 200:
		# 	random_index = np.random.randint(0, test_spec.shape[1] - 200)
		# 	test_spec = test_spec[:, random_index: random_index + 200]
		# else:
		# 	len_left = (200 - test_spec.shape[1]) // 2
		# 	len_right = 200 - test_spec.shape[1] - len_left
		# 	test_spec = np.pad(test_spec, ((0, 0), (len_left, len_right)), 'reflect')

		label = int(self.trn_label_list[index])
		# return torch.FloatTensor(audio[0]), self.data_label[index]
		return ref_spec.squeeze(0), test_spec.squeeze(0), label

	def __len__(self):
		return len(self.trn_ref_file_list)
	

class validate_loader(object):
	def __init__(self, val_path, **kwargs):
		self.val_file = val_path

		self.val_file_list, self.val_label_list = [], []
		if self.val_file:
			with open(self.val_file, 'r') as fv:
				file_list = fv.readlines()
			for i in range(len(file_list)):
				wav_path, label = file_list[i][:-1].split('\t')
				self.val_file_list.append(wav_path)
				self.val_label_list.append(label)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		spec = np.load(self.val_file_list[index])
		# segment length: 2s for sr=16k, i.e., 200 frames
		if spec.shape[1] >= 200:
			random_index = np.random.randint(0, spec.shape[1] - 200)
			spec = spec[:, random_index: random_index + 200]
		else:
			len_left = (200 - spec.shape[1]) // 2
			len_right = 200 - spec.shape[1] - len_left
			spec = np.pad(spec, ((0, 0), (len_left, len_right)), 'reflect')
			# print(1)

		label = int(self.val_label_list[index])
		# return torch.FloatTensor(audio[0]), self.data_label[index]
		return torch.FloatTensor(spec), label

	def __len__(self):
		return len(self.val_file_list)