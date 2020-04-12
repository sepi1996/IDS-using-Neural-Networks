# NIDS based on Artificial Neural Networks

In this project I will design and implement a network intrusion detection system using artificial neural networks. For so I am going to use the high-level neural networks API <a href="https://keras.io/" target="_blank">**Keras**</a> along with the dataset provided by the <a href="https://www.unb.ca/cic/datasets/ids-2018.html" target="_blank">**Communications Security Establishment (CSE) & the Canadian Institute for Cybersecurity (CIC)**</a> 

##### Table of Contents  
- [Dataset](#Dataset)
- [Cleanup](#Cleanup)
- [Architecture](#Architecture)
- [BestIds](#Installation)
- [Results](#Results)


## Dataset
In order for the neural network to be able to distinguish between normal traffic and an abnormal one, it will require a set of input values categorized, in order to learn from these how to classify futures data that it will receive.
For so, the dataset to be used will be the one created by Communications Security Establishment (CSE) and the Canadian Cybersecurity Institute (CIC) as I said before dated in 2018. The traffic generated is collected in the datasets provided
for several days on the same stage, but with different types of attacks, that is why, we are going to focus on the traffic generated on day 14-02-2018 since this corresponds to brute force attacks on the SSH and FTP protocols.


## Cleanup
So that the dataset can be used by the algorithm training, previously we will have to perform the following data processing tasks, as we can see in the import.py file
- Row cleanup
- Extraction of unnecessary columns
- Mix data
- Normalization
- Codification

In the next table we can see an overview of the fianl dataset content:
| Traffic type    | Number of rows   | Percentage(from cleaned data) | 
| --------------- | ---------------- | ------------ |
| Total           | 1048550          |              | 
| Deleted rows    | 3799             |              |
| New total       | 1044751          |   100%       |
| FTP brute force | 193354           |   18,52%     |
| SSH brute force | 187589           |   17,95%     |
| Benign          | 663808           |   63,53%     |
Finally, as we can see in the percentages, we have much more Benign traffic than FTP or SSH, so we will reduce this so that percentage it is more or less the same as for the other two types of traffic.

## Architecture
As we can see in the Kerastuner.py file, I will use <a href="https://github.com/autonomio/talos" target="_blank">**Talos**</a> for a fully automating hyperparameter tuning and model evaluation. The best results obtained are the following.

So we can extract this conclusions
1. The model works best with a single hidden layer.
2. No need to apply dropout.
3. The best number of neurons in the hidden layers is 256
neurons.
4. Both optimizers obtain similar results.
5. The number of neurons in the input layer seems indifferent

## BestIds
From the previous results, the best neural network architecture will be this one:


## Results
Finally, in the next images we can see, the results:
