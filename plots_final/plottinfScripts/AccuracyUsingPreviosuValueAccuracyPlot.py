from sklearn import preprocessing
import matplotlib.pyplot as plt 
import numpy as np

main_array_15 = [62.8, 65.0, 68.8, 65.6, 64.4, 67.4, 70.6, 70.6, 77.2, 74.2, 79.4, 76.0, 75.4, 79.2, 82.8, 83.0]
main_array_10 = [42.8, 44.0, 46.2, 42.6, 46.4, 47.2, 49.6, 56.2, 56.0, 55.6, 58.0]
main_array_20 = [91.6, 93.8, 91.4, 94.6, 99.8, 97.8, 100.4, 98.0, 104.6, 100.4, 99.4, 106.4, 108.2, 101.6, 106.0, 108.0, 111.6, 113.8, 108.2, 114.0, 115.0]


normalized_arr = preprocessing.normalize([main_array_10])
main_array_10 = normalized_arr[0]
normalized_arr = preprocessing.normalize([main_array_15])
main_array_15 = normalized_arr[0]
normalized_arr = preprocessing.normalize([main_array_20])
main_array_20 = normalized_arr[0]

node_10 = [(x / 10)*100 for x in range(11)]
node_15 = [(x / 15) * 100 for x in range(16)]
node_20 = [(x / 20)* 100 for x in range(21)]


plt.title("t-1  Based Mitigation")

plt.xlabel("% Nodes Fail")
plt.ylabel("Normalised Sum")
plt.grid()
plt.plot(node_10 , main_array_10 , label = "Total Node - 10")
plt.plot(node_15 , main_array_15 , label = "Total Node - 15")
plt.plot(node_20 , main_array_20 , label = "Total Node - 20")
plt.legend()
plt.show()