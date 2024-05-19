# Neural-Network
This project has been developed for the Neural Network class. It is a Python library crafted to facilitate the creation, optimization, and validation of Neural Networks. It encompasses two optimizers, namely RMSProp and AdamW, alongside a range of activation functions and error functions.
### V 0.1.4
By: @Dan1543

Added the option to choose the batch size for the training, by default is 0, any number below 0 will result in a batch training any number between 2-num_samples will result in a mini-batch training and a 1 batch size will result in an stochastic training. **BE CAREFULL WITH THE LOW BATCH SIZE IN LARGE DATASETS, IT MAY RESULT IN A BAD PERFORMANCE**
### V 0.1.3
By: @Dan1543

Added the dropout option, by default in the Neural Network class is deactivated whit a 0, to take the dropout in count is needes to pass a value between (0,1], where is not recommended that this value is greater than 0.3.
### V 0.1.2
By: @Dan1543

Added the option to initialize the weights by the Nguyen-Widraw method. 
## V 0.1 
By: @Dan1543

For this version the DNN is working, the input data should be in a format like the following:
| Speed | Fuel rate |
|-------|-----------|
| 45.3  | 43.2      |
| 20.15 | -3.1      |
| 23.5  |  12.4     |

For following versions it's planned to added L1,L2 and dropout regularizers, also to include the Nguyen-Widraw method for the weigths initialization, so take them in count for complicated datasets.