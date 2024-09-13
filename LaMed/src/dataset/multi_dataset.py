import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset
import json
import SimpleITK as sitk
import pandas as pd

import ast
import monai.transforms as mtf
from monai.data import set_track_meta, NibabelReader

from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import *
from .term_dictionary import term_dict



def normalize(ct_tensor):
    # ct_tensor 복사 및 평탄화
    ct_numpy = ct_tensor.numpy()
    ct_voxel_numpy = ct_numpy.copy().flatten()
    
    # 모든 데이터에 대해 평균 계산
    thred = np.mean(ct_voxel_numpy)
    voxel_filtered = ct_voxel_numpy[ct_voxel_numpy > thred]
    
    # 전경 데이터에 대해 상한 및 하한 계산
    upper_bound = np.percentile(voxel_filtered, 99.95)
    lower_bound = np.percentile(voxel_filtered, 00.05)

    mean = np.mean(voxel_filtered)
    std  = np.std(voxel_filtered)
    
    # 각 환자의 CT 스캔에 따라 변환
    ct_numpy = np.clip(ct_numpy, lower_bound, upper_bound)
    ct_numpy = (ct_numpy - mean) / max(std, 1e-8)
    
    # min-max 정규화
    ct_numpy = ct_numpy - np.min(ct_numpy)
    ct_numpy = ct_numpy / max(np.max(ct_numpy), 1e-8)
    
    # 채널 차원 추가 및 permute
    ct_numpy = np.expand_dims(ct_numpy, axis=0)
    ct_numpy = np.transpose(ct_numpy, (0, 3, 1, 2))
    
    return ct_numpy

class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_df = pd.read_csv(args.cap_data_train_path)
        elif mode == "valid":
            self.data_df = pd.read_csv(args.cap_data_val_path)[args.start_idx:args.end_idx]
            # self.data_df = pd.read_csv(args.cap_data_val_path)
        else:
            print("The mode is not desired ! ")

        self.caption_prompts = Caption_templates
        self.caption_system_prompts = Cap_System_templates

        train_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear C, D, H, W,

                # augmentation 방향을 바꾸는 건 critical 해보임...
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                # common
                mtf.ToTensor(dtype=torch.float)
                ]
            )

        val_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear

                # common
                mtf.ToTensor(dtype=torch.float)
                ]    
            )
        
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'valid':
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                sample = self.data_df.iloc[idx]
                # IMAGE
                image_abs_path = os.path.join(self.data_root, sample["image_final_path"].replace('./', ''))
                image = self.transform(image_abs_path)

                # TEXT
                question = self.image_tokens + ' \n\n' \
                            + random.choice(self.caption_prompts) + ' \n\n' \
                            + 'Answer: '
                answer = "{}".format(sample["findings"])

                # question = '<|start_header_id|>system<|end_header_id|>' + ' \n' \
                #             + random.choice(self.caption_system_prompts) + '<|eot_id|> \n\n' \
                #             +'<|start_header_id|>user<|end_header_id|>' + ' \n' \
                #             + self.image_tokens + ' \n' \
                #             + 'Instruction: ' + random.choice(self.caption_prompts) + ' \n\n' \
                #             + 'Answer: ' + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
                # answer = "{}".format(sample["findings"])

                text_tensor = self.tokenizer(question+answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                    'region': sample["region"],
                    'path': image_abs_path,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_df) - 1)

class VQADataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_df = pd.read_csv(args.vqa_data_train_path)
        elif mode == "valid":
            self.data_df = pd.read_csv(args.vqa_data_val_path)[args.start_idx:args.end_idx]
        else:
            print("The mode is not desired ! ")

        self.vqa_prompts    = VQA_templates
        self.reason_prompts = Reasoning_templates
        self.answer_prompts = Answer_templates
        self.qtype_prompts  = QuestionType_templates
        self.vqa_system_prompts = VQA_System_templates

        train_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear

                # augmentation # 방향을 바꾸는 건 critical 해보임...
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                # common
                mtf.ToTensor(dtype=torch.float)
                ]
            )


        val_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear

                # common
                mtf.ToTensor(dtype=torch.float)
                ]    
            )
        
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'valid':
            self.transform = val_transform
        else:
            print("The mode is not desired ! ")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                sample = self.data_df.iloc[idx]
                # IMAGE
                image_abs_path = os.path.join(self.data_root, sample["image_final_path"].replace('./', ''))
                image = self.transform(image_abs_path)

                # TEXT
                options = ast.literal_eval(sample["options"])

                # question = '<|start_header_id|>system<|end_header_id|>' + ' \n' \
                #             + random.choice(self.vqa_system_prompts) + '<|eot_id|> \n\n' \
                #             +'<|start_header_id|>user<|end_header_id|>' + ' \n' \
                #             + self.image_tokens + ' \n' \
                #             + 'Type: ' + sample['type'] + ' \n' \
                #             + 'Question: ' + sample['question'] + ' \n' \
                #             + "Choices: A. {} B. {} C. {} D. {}".format(options['A'], options['B'], options['C'], options['D']) + ' \n\n' \
                #             + 'Answer: ' + random.choice(self.reason_prompts) + ' ' + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
                # answer = sample['reasoning'] + ' ' + random.choice(self.answer_prompts) + ' ' + sample["answer"]

                # 가장 최근
                question = self.image_tokens + ' \n\n' \
                            + random.choice(self.vqa_prompts) + ' ' \
                            + random.choice(self.qtype_prompts) + ' \n\n' \
                            + 'Type: ' + sample['type'] + ' \n' \
                            + 'Question: ' + sample['question'] + ' \n' \
                            + "Choices: A. {}, B. {}, C. {}, D. {}".format(options['A'], options['B'], options['C'], options['D']) + ' \n\n' \
                            + 'Answer: ' + random.choice(self.reason_prompts) + ' '
                answer = sample['reasoning'] + ' ' + random.choice(self.answer_prompts) + ' ' + sample["answer"]
                
                text_tensor = self.tokenizer(question+answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",)

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100   # question 부분은 -100으로 masking
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': sample["type"],
                    'region': sample["region"],
                    'path': image_abs_path,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_df) - 1)

class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode=mode),
            VQADataset(args, tokenizer, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class Cap_Review_Dataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_df = pd.read_csv(args.cap_errornote_data_train_path)
            self.data_df = self.data_df[self.data_df['mode'] == 'train']
        elif mode == "valid":
            self.data_df = pd.read_csv(args.cap_errornote_data_val_path)
            self.data_df = self.data_df[self.data_df['mode'] == 'valid']
        else:
            print("The mode is not desired ! ")

        self.caption_prompts  = Caption_templates
        self.review_prompts   = Review_templates
        self.review_system_prompts = Review_System_templates

        train_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear C, D, H, W,

                # augmentation # 방향을 바꾸는 건 critical 해보임...
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                # common
                mtf.ToTensor(dtype=torch.float)
                ]
            )

        val_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear

                # common
                mtf.ToTensor(dtype=torch.float)
                ]    
            )
        
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'valid':
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                sample = self.data_df.iloc[idx]
                # IMAGE
                image_abs_path = os.path.join(self.data_root, sample["image_final_path"].replace('./', ''))
                image = self.transform(image_abs_path)

                # TEXT
                question = '<|start_header_id|>system<|end_header_id|>' + ' \n' \
                            + random.choice(self.review_system_prompts) + '<|eot_id|> \n\n' \
                            +'<|start_header_id|>user<|end_header_id|>' + ' \n' \
                            + self.image_tokens + ' \n' \
                            + 'Problem: ' + random.choice(self.caption_prompts) + ' \n' \
                            + 'Your Incorrect Answer: ' + sample['Pred'] + ' \n\n' \
                            + 'Explanation: ' + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
                answer = sample['explanations'] + ' \n' \
                        + 'Correct Answer: ' + sample["findings"]
                
                # question = self.image_tokens + ' \n\n' \
                #             + random.choice(self.review_prompts) + ' ' \
                #             + random.choice(self.analysis_prompts) + ' \n\n' \
                #             + '"""Problem: ' + random.choice(self.caption_prompts) + ' \n' \
                #             + 'Your Incorrect Answer: ' + sample['Pred'] + '""" \n\n' \
                #             + 'Explanation: '

                # answer = sample['explanations'] + ' \n' \
                #         + 'Correct Answer: ' + sample["findings"]
                
                text_tensor = self.tokenizer(question+answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                    'region': sample["region"],
                    'path': image_abs_path,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_df) - 1)

class VQA_Review_Dataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_df = pd.read_csv(args.vqa_errornote_data_train_path)
            self.data_df = self.data_df[self.data_df['mode'] == 'train']
        elif mode == "valid":
            self.data_df = pd.read_csv(args.vqa_errornote_data_val_path)
            self.data_df = self.data_df[self.data_df['mode'] == 'valid']
        else:
            print("The mode is not desired ! ")

        self.vqa_prompts      = VQA_templates
        self.reason_prompts   = Reasoning_templates
        self.answer_prompts   = Answer_templates
        self.qtype_prompts    = QuestionType_templates
        self.review_prompts   = Review_templates
        self.review_system_prompts = Review_System_templates

        train_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear

                # augmentation 
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                # common
                mtf.ToTensor(dtype=torch.float)
                ]
            )


        val_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear

                # common
                mtf.ToTensor(dtype=torch.float)
                ]    
            )
        
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'valid':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                sample = self.data_df.iloc[idx]
                # IMAGE
                image_abs_path = os.path.join(self.data_root, sample["image_final_path"].replace('./', ''))
                image = self.transform(image_abs_path)

                # TEXT
                options  = ast.literal_eval(sample["options"])
                question = '<|start_header_id|>system<|end_header_id|>' + ' \n' \
                            + random.choice(self.review_system_prompts) + '<|eot_id|> \n\n' \
                            +'<|start_header_id|>user<|end_header_id|>' + ' \n' \
                            + self.image_tokens + ' \n' \
                            + 'Type: ' + sample['type'] + ' \n' \
                            + 'Question: ' + sample['question'] + ' \n' \
                            + "Choices: A. {}, B. {}, C. {}, D. {}".format(options['A'], options['B'], options['C'], options['D']) + ' \n' \
                            + 'Your Incorrect Solution: ' + random.choice(self.reason_prompts) + ' ' + sample['Pred'] + ' \n\n' \
                            + 'Explanation: ' + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
                answer = sample['reasoning'] + ' \n' \
                            + 'Correct Solution: the correct answer is' + ' ' + sample["answer"]

                # choices  = "Choices: A. {} B. {} C. {} D. {}".format(options['A'], options['B'], options['C'], options['D'])                
                # question = self.image_tokens + ' \n\n' \
                #             + random.choice(self.review_prompts) + ' ' \
                #             + random.choice(self.analysis_prompts) + ' \n\n' \
                #             + '"""Type: ' + sample['type'] + ' \n' \
                #             + 'Question: ' + sample['question'] + ' \n' \
                #             + choices + ' \n' \
                #             + 'Your Incorrect Solution: ' + random.choice(self.reason_prompts) + ' ' + sample['Pred'] + '""" \n\n' \
                #             + 'Explanation: '
                # answer   = sample['reasoning'] + ' \n' \
                #             + 'Correct Solution: the correct answer is' + ' ' + sample["answer"]
                
                text_tensor = self.tokenizer(question+answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",)

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100   # question 부분은 -100으로 masking
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': sample["type"],
                    'region': sample["region"],
                    'path': image_abs_path,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_df) - 1)

class Text_With_ErrorNote_Datasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(Text_With_ErrorNote_Datasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode=mode),
            VQADataset(args, tokenizer, mode=mode),            
            Cap_Review_Dataset(args, tokenizer, mode=mode),
            VQA_Review_Dataset(args, tokenizer, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# TEST --------------------------------------------------------------
class CapDataset_TEST(Dataset):
    def __init__(self, args, tokenizer, mode="test"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "test":
            self.data_df = pd.read_csv(args.cap_data_test_path)[args.start_idx:args.end_idx]
        else:
            print("The mode is not desired ! ")

        self.caption_prompts = Caption_templates
        self.caption_system_prompts = Cap_System_templates

        val_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear

                # common
                mtf.ToTensor(dtype=torch.float)
                ]    
            )
        
        set_track_meta(False)

        if mode == 'test':
            self.transform = val_transform
        else:
            print("The mode is not desired ! ")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                sample = self.data_df.iloc[idx]
                # IMAGE
                image_abs_path = os.path.join(self.data_root, sample["image_final_path"].replace('./', ''))
                image = self.transform(image_abs_path)

                # TEXT
                # question = self.image_tokens + ' \n\n' \
                #             + random.choice(self.caption_prompts) + ' \n\n' \
                #             + 'Answer: '

                question = '<|start_header_id|>system<|end_header_id|>' + ' \n' \
                            + random.choice(self.caption_system_prompts) + '<|eot_id|> \n\n' \
                            +'<|start_header_id|>user<|end_header_id|>' + ' \n' \
                            + self.image_tokens + ' \n' \
                            + 'Instruction: ' + random.choice(self.caption_prompts) + ' \n\n' \
                            + 'Answer: ' + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

                ret = {
                    'image': image,
                    'question': question,
                    'region': sample["region"],
                    'path': image_abs_path,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_df) - 1)

class VQADataset_TEST(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "test":
            self.data_df = pd.read_csv(args.vqa_data_test_path)[args.start_idx:args.end_idx]
        else:
            print("The mode is not desired ! ")

        self.vqa_prompts    = VQA_templates
        self.reason_prompts = Reasoning_templates
        self.answer_prompts = Answer_templates
        self.qtype_prompts  = QuestionType_templates
        self.vqa_system_prompts = VQA_System_templates

        val_transform = mtf.Compose(
            [
                # preprocessing
                mtf.LoadImage(image_only=True, ensure_channel_first=False, reader=NibabelReader()),
                mtf.Lambda(func=normalize),
                mtf.Flip(spatial_axis=2),
                mtf.Rotate90(k=1, spatial_axes=(0, 1)),         
                mtf.EnsureType(track_meta=False),
                mtf.CropForeground(source_key="image"),
                mtf.Resize(spatial_size=[32, 256, 256], mode='bilinear'),  # trilinear

                # common
                mtf.ToTensor(dtype=torch.float)
                ]    
            )
        
        set_track_meta(False)

        if mode == 'test':
            self.transform = val_transform
        else:
            print("The mode is not desired ! ")            

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                sample = self.data_df.iloc[idx]
                # IMAGE
                image_abs_path = os.path.join(self.data_root, sample["image_final_path"].replace('./', ''))
                image = self.transform(image_abs_path)

                # TEXT
                options  = ast.literal_eval(sample["options"])
                
                question = self.image_tokens + ' \n\n' \
                            + random.choice(self.vqa_prompts) + ' ' \
                            + random.choice(self.qtype_prompts) + ' \n\n' \
                            + 'Type: ' + sample['type'] + ' \n' \
                            + 'Question: ' + sample['question'] + ' \n' \
                            + "Choices: A. {}, B. {}, C. {}, D. {}".format(options['A'], options['B'], options['C'], options['D']) + ' \n\n' \
                            + 'Answer: ' + random.choice(self.reason_prompts) + ' '

                # question = '<|start_header_id|>system<|end_header_id|>' + ' \n' \
                #             + random.choice(self.vqa_system_prompts) + '<|eot_id|> \n\n' \
                #             +'<|start_header_id|>user<|end_header_id|>' + ' \n' \
                #             + self.image_tokens + ' \n' \
                #             + 'Type: ' + sample['type'] + ' \n' \
                #             + 'Question: ' + sample['question'] + ' \n' \
                #             + "Choices: A. {} B. {} C. {} D. {}".format(options['A'], options['B'], options['C'], options['D']) + ' \n\n' \
                #             + 'Answer: ' + random.choice(self.reason_prompts) + ' ' + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'

                ret = {
                    'image': image,
                    'question': question,
                    'question_type': sample["type"],
                    'region': sample["region"],
                    'path': image_abs_path,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_df) - 1)
