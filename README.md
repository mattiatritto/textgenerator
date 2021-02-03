# RNN Neural Networks for generating text from previous WhatsApp Messages


RNNs are a special type of neural network that deals with **sequential data** to make predictions.
In this network, the information moves in only one direction, forward, from the input nodes, through
the hidden nodes (if any) and to the output nodes.
There are **no cycles** in the network.

![Image](https://miro.medium.com/max/800/1*8wOnrZ2UHFItcTdDNSZnTQ.png)



## How to train a Recurrent Neural Network

Training a typical neural network involves the following steps:

* Input an example from a dataset;
* The network will take that example and apply some computations to it using randomly initialised variables (**weights and biases**);
* A predicted result will be produced;
* Comparing that result to the expected value will give us an error;
* Propagating the error back through the same path will adjust the variables;
* Steps 1-5 are repeted until we are confident to say that our variables are well-defined;
* A predication is made by applying these variables to a new unseen input.


Of course, that is a quite naive explanation of a neural network, but, at least, gives a good overview and might be useful.

So, how do we start? As explained above, we input one example at a time and produce one result, both of which are single words. the difference with a feedforward network comes in teh fact that we also need to be informed about the previous inputs before evaluating the result. So we can view RNNs as multiple feedforward neural networks, passing information from one to the other.

![Image](https://miro.medium.com/max/1400/1*4KwIUHWL3sTyguTahIxmJw.png)



Here x_1, x_2,...,x_t represent the input words form the text, y_1,...,y_t represent the predicted next words and h_0, h_1,...,h_t hold the information for the previous input words.


Let's define the equations needed for training:


![Image](https://miro.medium.com/max/1400/1*bJonSV6knypXR9StvEBYZw.png)


1) holds information about the previous words in the sequence. As you can see, h_t is calculated using the previous h_(t-1) vector and current word vector x_t. We also apply a non-linear activation function f (in our case is **logit**) to the final summation. It is acceptable to assume that h_0 is a vector of zeros.

2) calculates the predicted word vector at a given time step t. We use the **softmax function** to produce a (V,1) vector with all elements summing up to 1. This probability distribution gives us the index of the most likely next word from the vocabulary.

3) uses the **cross-entropy** loss function at each time step t to calculate the error between the predicted and actual word.

Each of W's represents the weights of the network at a certain stage. As mentioned above, the weights are matricies initialised with random elements, adjusted using the error from the loss function. We do this adjusting using back-propagation algorithm which updates the weights.


## Problems with a standard RNN

Unfortunately, if you implement the above steps, we won't be so delighted with the results. That is beacuse the simplest RNN model has a major drawback, called **vanishing gradient problem**.


In a nutshell, the problems comes form the fact that at each time step during training we are using the same weights to calculate y_t. That multiplication is also done during back-propagation. The further we move backwards, the bigger or smaller our error signal becomes. This means that the network experiences difficulty in memorising words from far away in the sequence and makes predictions based on only the most recent ones.


