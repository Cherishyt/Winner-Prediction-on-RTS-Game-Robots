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
