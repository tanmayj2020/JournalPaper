from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np 

# #Below is for node 10 
# [42.8, 39.4, 33.0, 25.4, 22.0, 21.6, 15.2, 13.4, 8.0, 4.8, 0.0]
# [42.8, 44.0, 42.8, 37.6, 37.0, 42.8, 39.8, 44.8, 41.8, 50.6, 42.4]
# [42.8, 44.0, 46.2, 42.6, 46.4, 47.2, 49.6, 56.2, 56.0, 55.6, 58.0]

# #Below is for node 15
# [62.8, 58.8, 55.4, 46.8, 40.4, 39.4, 37.4, 29.4, 33.6, 22.2, 23.6, 12.0, 12.6, 6.8, 4.4, 0.0]
# [62.8, 61.8, 62.0, 62.4, 58.8, 60.4, 60.2, 52.2, 71.6, 60.8, 70.4, 61.8, 63.0, 57.4, 66.2, 64.8]
# [62.8, 65.0, 68.8, 65.6, 64.4, 67.4, 70.6, 70.6, 77.2, 74.2, 79.4, 76.0, 75.4, 79.2, 82.8, 83.0]


#Below is for node 20

# mitigation_0 = [91.6, 90.6, 79.8, 76.8, 77.6, 69.2, 66.0, 56.8, 57.4, 51.2, 44.0, 42.0, 37.6, 27.4, 23.8, 21.4, 20.2, 10.6, 7.8, 5.4, 0.0]
# mitigation_probability = [91.6, 93.8, 88.6, 91.4, 97.2, 90.0, 92.0, 91.4, 82.8, 88.0, 87.8, 91.6, 92.2, 90.0, 88.0, 87.4, 94.4, 89.6, 93.2, 88.0, 93.6]
# mitigation_t_1 = [91.6, 93.8, 91.4, 94.6, 99.8, 97.8, 100.4, 98.0, 104.6, 100.4, 99.4, 106.4, 108.2, 101.6, 106.0, 108.0, 111.6, 113.8, 108.2, 114.0, 115.0]


#Below is for node 15 with GAN based also included
mitigation_0 = np.abs(62.8 - np.array([62.8, 58.2, 58.0, 50.6, 45.6, 42.8, 35.8, 33.8, 22.2, 25.4, 22.6, 17.8, 14.2, 9.6, 5.4, 0.0]))
mitigation_probability =np.abs(62.8 - np.array([62.8, 61.2, 67.4, 66.8, 62.4, 68.0, 59.8, 63.0, 63.0, 75.0, 61.8, 67.2, 68.6, 70.6, 70.6, 65.6]))
mitigation_t_1 = np.abs(62.8 - np.array([62.8, 65.2, 68.2, 69.2, 69.4, 72.6, 69.0, 77.2, 66.6, 74.2, 82.0, 78.8, 81.6, 82.6, 83.6, 83.0]))
mitigation_gan = np.abs(62.8 - np.array([62.8, 64.0, 68.4, 62.8, 66.0, 70.6, 61.8, 56.8, 61.2, 72.8, 82.4, 70.4, 73.2, 64.6, 78.6, 66.2]))

# normalized_arr = preprocessing.normalize([mitigation_0])
# mitigation_0 = normalized_arr[0]
# normalized_arr = preprocessing.normalize([mitigation_probability])
# mitigation_probability = normalized_arr[0]
# normalized_arr = preprocessing.normalize([mitigation_t_1])
# mitigation_t_1 = normalized_arr[0]
# normalized_arr = preprocessing.normalize([mitigation_gan])
# mitigation_gan = normalized_arr[0]

node_20 = [(x / 15)*100 for x in range(16)]



plt.title("Comparision Plot")

plt.xlabel("% Nodes Fail")
plt.ylabel("Error")
plt.grid()
print(mitigation_0)
plt.plot(node_20 , mitigation_0 , label = "No Mitigation")
plt.plot(node_20 , mitigation_t_1 , label = "t-1 Based Mitigation")
plt.plot(node_20 , mitigation_probability , label = "Probability Based Mitigation")
plt.plot(node_20 , mitigation_gan, label = "GAN Based Mitigation")

plt.legend()
plt.show()