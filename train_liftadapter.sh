#!/usr/bin/env bash
python train.py --dataset "ucf101" --batch_size 128 --shots 16 --adapter
python train.py --dataset "dtd" --batch_size 128 --shots 16 --adapter
python train.py --dataset "eurosat" --batch_size 128 --shots 16 --adapter
python train.py --dataset "fgvc" --batch_size 128 --shots 16 --adapter
python train.py --dataset "caltech101" --batch_size 128 --shots 16 --adapter
python train.py --dataset "food101" --batch_size 128 --shots 16 --adapter
python train.py --dataset "sun397" --batch_size 128 --shots 16 --adapter
python train.py --dataset "stanford_cars" --batch_size 128 --shots 16 --adapter
python train.py --dataset "oxford_flowers" --batch_size 128 --shots 16 --adapter

python train.py --dataset "ucf101" --batch_size 128 --shots 8 --adapter
python train.py --dataset "dtd" --batch_size 128 --shots 8 --adapter
python train.py --dataset "eurosat" --batch_size 128 --shots 8 --adapter
python train.py --dataset "fgvc" --batch_size 128 --shots 8 --adapter
python train.py --dataset "caltech101" --batch_size 128 --shots 8 --adapter
python train.py --dataset "food101" --batch_size 128 --shots 8 --adapter
python train.py --dataset "sun397" --batch_size 128 --shots 8 --adapter
python train.py --dataset "stanford_cars" --batch_size 128 --shots 8 --adapter
python train.py --dataset "oxford_flowers" --batch_size 128 --shots 8 --adapter

python train.py --dataset "ucf101" --batch_size 128 --shots 6 --adapter
python train.py --dataset "dtd" --batch_size 128 --shots 6 --adapter
python train.py --dataset "eurosat" --batch_size 128 --shots 6 --adapter
python train.py --dataset "fgvc" --batch_size 128 --shots 6 --adapter
python train.py --dataset "caltech101" --batch_size 128 --shots 6 --adapter
python train.py --dataset "food101" --batch_size 128 --shots 6 --adapter
python train.py --dataset "sun397" --batch_size 128 --shots 6 --adapter
python train.py --dataset "stanford_cars" --batch_size 128 --shots 6 --adapter
python train.py --dataset "oxford_flowers" --batch_size 128 --shots 6 --adapter

python train.py --dataset "ucf101" --batch_size 16 --shots 4 --adapter
python train.py --dataset "dtd" --batch_size 16 --shots 4 --adapter
python train.py --dataset "eurosat" --batch_size 16 --shots 4 --adapter
python train.py --dataset "fgvc" --batch_size 16 --shots 4 --adapter
python train.py --dataset "caltech101" --batch_size 16 --shots 4 --adapter
python train.py --dataset "food101" --batch_size 128 --shots 4 --adapter
python train.py --dataset "sun397" --batch_size 128 --shots 4 --adapter
python train.py --dataset "stanford_cars" --batch_size 128 --shots 4 --adapter
python train.py --dataset "oxford_flowers" --batch_size 128 --shots 4 --adapter

python train.py --dataset "ucf101" --batch_size 16 --shots 2 --adapter
python train.py --dataset "dtd" --batch_size 16 --shots 2 --adapter
python train.py --dataset "eurosat" --batch_size 16 --shots 2 --adapter
python train.py --dataset "fgvc" --batch_size 16 --shots 2 --adapter
python train.py --dataset "caltech101" --batch_size 16 --shots 2 --adapter
python train.py --dataset "food101" --batch_size 16 --shots 2 --adapter
python train.py --dataset "sun397" --batch_size 16 --shots 2 --adapter
python train.py --dataset "stanford_cars" --batch_size 16 --shots 2 --adapter
python train.py --dataset "oxford_flowers" --batch_size 16 --shots 2 --adapter

python train.py --dataset "pets" --batch_size 16 --shots 2 --adapter
python train.py --dataset "pets" --batch_size 16 --shots 4 --adapter
python train.py --dataset "pets" --batch_size 16 --shots 6 --adapter
python train.py --dataset "pets" --batch_size 16 --shots 8 --adapter
python train.py --dataset "pets" --batch_size 16 --shots 16 --adapter
python train.py --dataset "imagenet" --batch_size 128 --shots 2 --adapter
python train.py --dataset "imagenet" --batch_size 128 --shots 4 --adapter
python train.py --dataset "imagenet" --batch_size 128 --shots 6 --adapter
python train.py --dataset "imagenet" --batch_size 128 --shots 8 --adapter
python train.py --dataset "imagenet" --batch_size 128 --shots 16 --adapter