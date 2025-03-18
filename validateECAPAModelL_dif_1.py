'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoaderL_dif_wav import train_loader, validate_loader
from ECAPAModelL_dif_1_wav import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=100,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=16,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=1,       help='Number of loader threads')
parser.add_argument('--device',     type=str,   default="cuda:0")
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--initial_model', 
					type=str, 
					default="results/data_3_1_apple_retrain/model/model_0016.model", 
					help='Path of the initial_model')
# parser.add_argument('--train_path', 
# 					type=str, 
# 					default="file_list/trn_list_no_recording.txt")
parser.add_argument('--val_path', 
					type=str, 
					default="file_list/mel/dif/data_3_1/apple/val_apple_pair_list_replay_1_2.txt")

## Model and Loss settings
# parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--C',       type=int,   default=80,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
# parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
parser.add_argument('--n_class', type=int,   default=2,   help='Number of classes')

## Command
# parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')
parser.add_argument('--eval', type=bool, default=True, help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
# torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
# args = init_args(args)


## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(val_path=args.val_path)
	print("EER %2.2f%%"%(EER))

	# print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	
	# accuracy1 = s.eval_network_confusion_matrix(val_path=args.val_path)
	# print(accuracy1)
	quit()
