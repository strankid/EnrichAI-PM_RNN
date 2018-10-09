
			       			RNN for Predictive Maintenence 


The project consists of 3 steps. 

Preparing the dataset: In this step, columns that are not required for training are dropped. Then, the dataset is sorted in time to make sure the rows are in chronological order. Finally, non-textual (real-numbered) columns are normalized using scikit-learns’ Robust Scaler. Normalizing is neccesary to make sure that all values are roughly in the same range. If some values are excessively larger than others, their influence on the model will be larger in that proportion. Robust Scaler was chosen because even though it normalizes the values, it preserves the outliers. In our application, outliers may be important indicators of anamolies and are therefore preserved. The output is then passed to the preprocessing function.

Preprocessing the dataset: This is the most crucial step of the RNN pipeline. This step transforms the dataset into sequences that can be fed into the first LSTM layer of the model. Initially, we extract all the data points for a particular head and sort it by time. We create a new column in this data frame called ‘target’. The target contains the value of parent_event that is FUTURE_LENGTH data points ahead. This allows us to predict failure/non-failure FUTURE_LENGTH days ahead. Now, we must create sequences. This is achieved by using a deque. A deque is data-structure that has a fixed length. In our case, the length of the deque is SEQ_LENGTH = 10. Initially, the deque is empty. As elements are added to the deque, it keeps expanding until the number of elements it holds are equal to SEQ_LENGTH. From this point on, the deque behaves like a normal queue. Every time a new element is added, the first element is automatically removed from the queue. This data structure aids us in preparing the sequences we require. 

We iterate over the rows of the data frame that contains the data points for a single head. We keep adding each row to our deque. If there is a break in the sequence, such as non-continuous data points or a failure event, the deque is cleared. This way only contiguous data points are added to our sequence and after each sequence that ends with a failure, a completely new sequence is created. This ensures that our RNN model recieves proper sequences that it can work with. Every unique sequence in the deque (with size = SEQ_LEN) is accepted as a training sequence for our model with the output of the entire sequence equal to the value of the target column of the latest row in that sequence. We add all these sequences and their outputs to a new data frame. This process is repeated for all the heads in the data.

An additional requirement for RNN’s to perform well is data balancing. Number of sequences with target equal to 1 should be approximately equal to the number of sequnces with target equal to 0.  This is the greatest challenge faced by our model as our data is highly skewed towards the non-failure class. To overcome this problem, the number of negatives sequences selected for training is twice the number of positive sequences. These negative sequences are chosen at random from the total pool of negative sequences. This ensures that our training data is somewhat balanced. However, this reduces the number of sequences in our training data to less than 10% of the total sequences. The best way to improve the accuracy of this model is to have more variance in the training data so that the model has more information to learn from. 

Building the model: Our model is built using keras. It consists of 3 LSTM layers with 128 neurons each. 2 dense layers with 64 and 32 neurons respectively and RELU activation. Finally the output layer consists of 2 neurons and softmax activation.

Long short-term memory (LSTM) units are units of a recurrent neural network (RNN).  A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.  A great article that explains LSTM: http://colah.github.io/posts/2015-08-Understanding-LSTMs/. 

Dense layers are just regular densely connected NN Layer. Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True). Find out more at: https://keras.io/layers/core/






 
