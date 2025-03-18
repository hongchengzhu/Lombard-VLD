'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import numpy as np
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
# from modelL import ECAPA_TDNN
from preprocess.melspec.compute_mel import mel_spectrogram
from ECAPA_TDNNL_dif_1_wav import ECAPA_TDNN

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, device, **kwargs):
		super(ECAPAModel, self).__init__()
		self.device = device
		## ECAPA-TDNN
		# self.speaker_encoder = ECAPA_TDNN(C=C).to(self.device)
		self.speaker_encoder = ECAPA_TDNN(input_size=C, device=self.device).to(self.device)
		## Classifier
		self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).to(self.device)

		self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (ref_data, test_data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels = labels.to(self.device)
			# speaker_embedding = self.speaker_encoder.forward(data.to(self.device), aug=False)
			ref_data = ref_data.transpose(1, 2).to(self.device)			# data original shape: [B, C, T] --> [B, T, C]
			test_data = test_data.transpose(1, 2).to(self.device)			# data original shape: [B, C, T] --> [B, T, C]
			speaker_embedding = self.speaker_encoder.forward(ref_data, test_data)	# TDNN input shape: [B, T, C], output shape: [B, 1, 192]
			speaker_embedding = speaker_embedding.squeeze(1)		# speaker embedding shape: [B, 192]
			nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)			
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	
	def eval_network(self, val_path):
		self.eval()
		files = []
		embeddings = {}
		with open(val_path, 'r') as fv:
			lines = fv.readlines()
		# lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1] + '\t' + line.split()[2])
			files.append(line.split()[3] + '\t' + line.split()[4])

			# files.append(line.split('\t')[1])
			# files.append(line.split('\t')[2])

			# files.append(line.split()[2])
			# files.append(line.split()[3])
			# files.append(line.split()[4])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
			# wav1, sr  = soundfile.read(file)
			# assert sr == 16000, 'should keep sr=16000'
			# spec1 = mel_spectrogram(torch.FloatTensor(wav1).unsqueeze(0), 
			#    						n_fft=1024, num_mels=80, sampling_rate=sr, 
			# 						hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False)
			# Full utterance
			# data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to(self.device)

			ref_file, test_file = file.split('\t')

			ref_wav, ref_sr = soundfile.read(ref_file)
			test_wav, test_sr = soundfile.read(test_file)
			assert ref_sr == 16000 and test_sr == 16000, 'should keep sr=16000'

			# if len(ref_wav) < 2 * ref_sr:
			# 	len_left = int((2 * ref_sr - len(ref_wav)) // 2)
			# 	len_right = 2 * ref_sr - len(ref_wav) - len_left
			# 	ref_wav = np.pad(ref_wav, ((len_left, len_right)), 'edge')
			# if len(test_wav) < 2 * test_sr:
			# 	len_left = int((2 * test_sr - len(test_wav)) // 2)
			# 	len_right = 2 * test_sr - len(test_wav) - len_left
			# 	test_wav = np.pad(test_wav, ((len_left, len_right)), 'edge')

			ref_spec_1 = mel_spectrogram(torch.FloatTensor(ref_wav).unsqueeze(0), 
			   						n_fft=1024, num_mels=80, sampling_rate=ref_sr, 
									hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False).to(self.device)
			test_spec_1 = mel_spectrogram(torch.FloatTensor(test_wav).unsqueeze(0), 
			   						n_fft=1024, num_mels=80, sampling_rate=test_sr, 
									hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False).to(self.device)

			# ref_spec = np.load(ref_file).T		# shape: [C, T] --> [T, C]
			# test_spec = np.load(test_file).T		# shape: [C, T] --> [T, C]
			
			# ref_data_1 = torch.FloatTensor(np.stack([ref_spec],axis=0)).to(self.device)
			# test_data_1 = torch.FloatTensor(np.stack([test_spec],axis=0)).to(self.device)

			# Spliited utterance matrix, 暂时理解：整个句子和定长句子（多次定长切分）的embedding均值作为最终embedding
			# max_audio = 300 * 160 + 240
			if len(ref_wav) <= 2 * ref_sr:
				len_left = int((2 * ref_sr - len(ref_wav)) // 2)
				len_right = 2 * ref_sr - len(ref_wav) - len_left
				ref_wav_2 = np.pad(ref_wav, ((len_left, len_right)), 'edge')
			else:
				ref_wav_2 = ref_wav
			feats = []
			startframe = numpy.linspace(0, ref_wav_2.shape[0] - 2 * ref_sr, num=5)
			for asf in startframe:
				sub_ref_wav = ref_wav_2[int(asf): int(asf) + 2 * ref_sr]
				sub_ref_spec_2 = mel_spectrogram(torch.FloatTensor(sub_ref_wav).unsqueeze(0), 
			   						n_fft=1024, num_mels=80, sampling_rate=ref_sr, 
									hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False).squeeze(0)
				feats.append(sub_ref_spec_2)
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			ref_data_2 = torch.FloatTensor(feats).to(self.device)

			if len(test_wav) <= 2 * test_sr:
				len_left = int((2 * test_sr - len(test_wav)) // 2)
				len_right = 2 * test_sr - len(test_wav) - len_left
				test_wav_2 = np.pad(test_wav, ((len_left, len_right)), 'edge')
			else:
				test_wav_2 = test_wav
			feats = []
			startframe = numpy.linspace(0, test_wav_2.shape[0] - 2 * test_sr, num=5)
			for asf in startframe:
				sub_test_wav = test_wav_2[int(asf): int(asf) + 2 * test_sr]
				sub_test_spec_2 = mel_spectrogram(torch.FloatTensor(sub_test_wav).unsqueeze(0), 
			   						n_fft=1024, num_mels=80, sampling_rate=test_sr, 
									hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False).squeeze(0)
				feats.append(sub_test_spec_2)
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			test_data_2 = torch.FloatTensor(feats).to(self.device)

			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(ref_spec_1.transpose(1, 2), test_spec_1.transpose(1, 2))			# [1, C, T] --> [1, T, C]
				embedding_1 = embedding_1.squeeze(1)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(ref_data_2.transpose(1, 2), test_data_2.transpose(1, 2))			# [5, C, T] --> [5, T, C]
				embedding_2 = embedding_2.squeeze(1)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1] + '\t' + line.split()[2]]
			embedding_21, embedding_22 = embeddings[line.split()[3] + '\t' + line.split()[4]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF


	def eval_network_confusion_matrix(self, val_path):
		self.eval()
		with open(val_path, 'r') as fv:
			setfiles = fv.readlines()

		embeddings = torch.zeros([len(setfiles), 192])
		labels = torch.LongTensor(len(setfiles))

		for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
			label, ref_file, test_file = file[:-1].split('\t')
			labels[idx] = int(label)

			ref_wav, ref_sr = soundfile.read(ref_file)
			test_wav, test_sr = soundfile.read(test_file)
			assert ref_sr == 16000 and test_sr == 16000, 'should keep sr=16000'

			ref_spec_1 = mel_spectrogram(torch.FloatTensor(ref_wav).unsqueeze(0), 
			   						n_fft=1024, num_mels=80, sampling_rate=ref_sr, 
									hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False).to(self.device)
			test_spec_1 = mel_spectrogram(torch.FloatTensor(test_wav).unsqueeze(0), 
			   						n_fft=1024, num_mels=80, sampling_rate=test_sr, 
									hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False).to(self.device)


			# Spliited utterance matrix, 暂时理解：整个句子和定长句子（多次定长切分）的embedding均值作为最终embedding
			# max_audio = 300 * 160 + 240
			if len(ref_wav) <= 2 * ref_sr:
				len_left = int((2 * ref_sr - len(ref_wav)) // 2)
				len_right = 2 * ref_sr - len(ref_wav) - len_left
				ref_wav_2 = np.pad(ref_wav, ((len_left, len_right)), 'edge')
			else:
				ref_wav_2 = ref_wav
			feats = []
			startframe = numpy.linspace(0, ref_wav_2.shape[0] - 2 * ref_sr, num=5)
			for asf in startframe:
				sub_ref_wav = ref_wav_2[int(asf): int(asf) + 2 * ref_sr]
				sub_ref_spec_2 = mel_spectrogram(torch.FloatTensor(sub_ref_wav).unsqueeze(0), 
			   						n_fft=1024, num_mels=80, sampling_rate=ref_sr, 
									hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False).squeeze(0)
				feats.append(sub_ref_spec_2)
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			ref_data_2 = torch.FloatTensor(feats).to(self.device)

			if len(test_wav) <= 2 * test_sr:
				len_left = int((2 * test_sr - len(test_wav)) // 2)
				len_right = 2 * test_sr - len(test_wav) - len_left
				test_wav_2 = np.pad(test_wav, ((len_left, len_right)), 'edge')
			else:
				test_wav_2 = test_wav
			feats = []
			startframe = numpy.linspace(0, test_wav_2.shape[0] - 2 * test_sr, num=5)
			for asf in startframe:
				sub_test_wav = test_wav_2[int(asf): int(asf) + 2 * test_sr]
				sub_test_spec_2 = mel_spectrogram(torch.FloatTensor(sub_test_wav).unsqueeze(0), 
			   						n_fft=1024, num_mels=80, sampling_rate=test_sr, 
									hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False).squeeze(0)
				feats.append(sub_test_spec_2)
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			test_data_2 = torch.FloatTensor(feats).to(self.device)

			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(ref_spec_1.transpose(1, 2), test_spec_1.transpose(1, 2))			# [1, C, T] --> [1, T, C]
				embedding_1 = embedding_1.squeeze(1)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(ref_data_2.transpose(1, 2), test_data_2.transpose(1, 2))			# [5, C, T] --> [5, T, C]
				embedding_2 = embedding_2.squeeze(1)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)

				embeddings[idx] = torch.mean(torch.cat((embedding_1, embedding_2), 0), 0)

				embeddings = embeddings.to(self.device)
				labels = labels.to(self.device)

		_, tpr, far, trr, frr = self.speaker_loss.forward_confusion_matrix(embeddings, labels)

		return tpr, far, trr, frr

	def eval_network_bk(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to(self.device)

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).to(self.device)
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)