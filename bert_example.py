#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : ZHANG Jingzhuo
# @Contact : zhangjingzhuo@pku.edu.cn
# @File    : bert_example.py
# @Time    : 2019-06-06 09:37
# @Desc    : 

# Train & Use
from keras_bert import get_base_dict, get_model, gen_batch_inputs
import keras

# A toy input example
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],]


# Build token dictionary
token_dict = get_base_dict()  # A dict that contains some special tokens
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word


# Build & train the model
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
model.summary()

def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )

model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)


# Use the trained model
inputs, output_layer = get_model(  # `output_layer` is the last feature extraction layer (the last transformer)
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,  # The input layers and output layer will be returned if `training` is `False`
)




# Custom Feature Extraction
def _custom_layers(x, trainable=True):
    return keras.layers.LSTM(
        units=768,
        trainable=trainable,
        name='LSTM',
    )(x)

model = get_model(
    token_num=200,
    embed_dim=768,
    custom_layers=_custom_layers,
)