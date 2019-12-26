# Winner-Prediction-on-RTS-Game-Robots
Winner Prediction Project
## Environments
**Eclipse**<br>
**Python 3.7**<br>
* tensorflow 1.14.0<br>
* pytorch 1.3.1<br>
* numpy 1.17.2<br>
* matplotlib 3.1.1<br>
* sklearn<br>
## Data
### 1. Replay Data
* Using [microRTS](https://github.com/santiontanon/microrts), an RTS game platform for testing AI technology, to generate gameplay replay data for multiple groups of game robots.
* The dataset was created by playing round-robin tournaments between 10 different microRTS bots using 8x8 map with 23 different initial starting conditions. These tournaments were played under four different time limits: maximums of 100ms, 200ms, 100 playouts and 200 playouts per search episode.
* Replay data format is `xml`.
### 2. Data Sampling
* There are two kinds of sampling methods:<br>
	* sample 3 random positions from each game
		* run `processdata.java` (modify data path)
		* generate one `randomsample` dataset
	* sample 20 positions from each game in turn
		* run `Sample_20_part.java` (modify data path)
		* generate 20 `samplepart*` datasets
* Sampling data format is `xml`.
### 3. Data Encoding
* There are two kinds of encoding methods:<br>
	* one-hot
		* run `Parse_one_hot.py` (modify data path)
		* generate 3D array: `(8,8,38)`
	* feature count
		* run `Parse_feature_count.java` (modify data path)
		* generate 3D array: `(8,8,7)`
* Encoding data format is `npy`.
## Model
### 1、CNN model
* run `/CNN model/cnn_train.py`,rootpath is the path of npy files.
* code refers to (https://github.com/hwalsuklee/tensorflow-mnist-cnn)
### 2、MSCNN model
* run `/MSCNN model/mscnn_train.py`,rootpath is the path of npy files.
* code refers to (https://blog.csdn.net/loveliuzz/article/details/79135583)
### 3、CNP model
* run `/CNP model/cnp_train.py`,rootpath is the path of npy files.
* code refers to (https://github.com/deepmind/neural-processes)
### 4、BNN model
* run `/BNN model/train_BayesByBackprop.py`,rootpath is the path of npy files.
* code refers to (https://github.com/JavierAntoran/Bayesian-Neural-Networks)
### 5、LSTM model
* run `/LSTM model/lstm_train.py`,rootpath is the path of npy files.
## Reference
* M. Stanescu et al.(2016). Evaluating Real-Time Strategy Game States Using Convolutional Neural Networks. 10.1109/CIG.2016.7860439.
* Jie Huang et al.(2018). A Multi-size Convolution Neural Network for RTS Games Winner Prediction.MATEC Web of Conferences,232.
* Marta Garnelo, Dan Rosenbaum et al.(2018). Conditional neural processes.ICML.
* Blundell, C. , Cornebise, J. , Kavukcuoglu, K. , & Wierstra, D. . (2015). Weight uncertainty in neural networks. Computer Science.
