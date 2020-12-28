# An Autonomous Spectrum Management Scheme for Unmanned Aerial Vehicle Networks in Disaster Relief Operations using Multi Independent Agent Reinforcement Learning

## Paper
You can find the **article** related to this code [here at IEEE](https://ieeexplore.ieee.org/abstract/document/9046033) or
You can find the **preprint** from the [Arxiv website](https://arxiv.org/pdf/1911.11343.pdf).

* The system model of this paper is based on:
![Alt text](/images/system.JPG)
![Alt text](/images/system2.JPG)

Abstract: This paper studies the problem of spectrum shortage in an unmanned aerial vehicle (UAV) network during critical missions such as wildfire monitoring, search and rescue, and disaster monitoring. Such applications involve a high demand for high-throughput data transmissions such as real-time video-, image-, and voice- streaming where the assigned spectrum to the UAV network may not be adequate to provide the desired Quality of Service (QoS). In these scenarios, the aerial network can borrow additional spectrum from the available terrestrial networks in trade of a relaying service for them. We propose a spectrum sharing model in which the UAVs are grouped into two classes of relaying UAVs that service the spectrum owner and the sensing UAVs that perform the disaster relief mission using the obtained spectrum. The operation of the UAV network is managed by a hierarchical mechanism in which a central controller assign the tasks of the UAVs based on their resources and determine their operation region based on the level of priority of impacted areas and then the UAVs autonomously fine-tune their position using a modelfree reinforcement learning algorithm to maximize the individual throughput and prolong their lifetime. We analyze the performance and the convergence for the proposed method analytically and with extensive simulations in different scenarios.

## Code
This code is run and tested on Python 3.6 on both linux machine with no issues. There is a config file in this directoy which shows all the configuration parameters such as transmit power, the grid size, number of steps, number of epochs, number of runs, number of UAVs, number of regions, etc. The number of UAVs in this study is variable. You can simply run the main.py file to run the code. It doesn't need any input argument, all you need to configure is available in the config.py. All dependency files are available in the root directory of this repository. You can change the Mode of this code with a variable "Mode" in the config file. Here are five possible options for this program:<br/>
  ***** Modes ==>  0: Normal: Search primary region, search primary uav, search secondary regions + UAVs + RL<br/>
  *****            1: Search Primary/Secondary regions + random UAV allocation + RL<br/>
  *****            2: Random Region assignment + Random UAV Allocation + RL<br/>
  *****            3: Random Region assignment + base-station UAV selection + RL<br/>
  *****            4: Base-station Region assignment + Base-station UAV selection + Random actions(not RL)<br/>

The proposed approach in our paper is Mode 0. Other modes are used for the comparison. Runing the main file will run the app:

```
python main.py
```

## Required Packages
* os
* copy
* time
* match
* Numpy
* Scipy
* Random
* matplotlib.pyplot

## Results
<!--- 
![Alt text](/image/throughput.JPG)
![Alt text](/image/movement.JPG)
![Alt text](/image/table.JPG)
--->


## Citation
If you find the code or the article useful, please cite our paper using this BibTeX:
```
@article{shamsoshoara2020autonomous,
  title={An Autonomous Spectrum Management Scheme for Unmanned Aerial Vehicle Networks in Disaster Relief Operations},
  author={Shamsoshoara, Alireza and Afghah, Fatemeh and Razi, Abolfazl and Mousavi, Sajad and Ashdown, Jonathan and Turk, Kurt},
  journal={IEEE Access},
  volume={8},
  pages={58064--58079},
  year={2020},
  publisher={IEEE}
}
```

## License
For academtic and non-commercial usage 
