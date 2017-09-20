## Project 4 - Language Translation

This project demonstrates the use of a sequence-to-sequence model applied to translating from English to French.

A sequence-to-sequence network is made up of two RNNs. One of the RNNs is used to process the input to the network (encoder), and the second generates the output of the network (decoder). The sequence-to-sequence network can be trained to generate a sequence of vectors when fed a given sequence of input vectors. The vectors could represent images, text, anything, but in this case we will use a small selection of English and French text.

![alt text](seq2seq1.png "Seq2Seq Model")

The encoder and decoder are comprised of a stack of LSTM cells. The encoder is fed the input and yeilds a new state as well as output for each time step. We ignore the outputs of the encoder but the state (weights) of the encoder are passed on to the decoder. The decoder utilizes the encoder state or context to help improve the accuracy of it's predictions. An inference decoder is also trained which will later be used to process real world data once training is complete. The inference decoder differs from the training decoder in that it feeds the output of each time step into the next LSTM cell, where as the training decoder inputs are the target sequences from the training dataset. 

![alt text](decoders.png "Comparison of Training & Inference Decoders")

[First paper introducing sequence-to-sequence models](https://arxiv.org/abs/1406.1078)
