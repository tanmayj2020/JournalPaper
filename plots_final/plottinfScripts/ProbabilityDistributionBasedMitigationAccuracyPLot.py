from sklearn import preprocessing
import matplotlib.pyplot as plt 
import numpy as np 


main_array_20 = [91.6, 93.8, 88.6, 91.4, 97.2, 90.0, 92.0, 91.4, 82.8, 88.0, 87.8, 91.6, 92.2, 90.0, 88.0, 87.4, 94.4, 89.6, 93.2, 88.0, 93.6]
main_array_10 = [42.8, 44.0, 42.8, 37.6, 37.0, 42.8, 39.8, 44.8, 41.8, 50.6, 42.4]
main_array_15 = [62.8, 61.8, 62.0, 62.4, 58.8, 60.4, 60.2, 52.2, 71.6, 60.8, 70.4, 61.8, 63.0, 57.4, 66.2, 64.8]




normalized_arr = preprocessing.normalize([main_array_10])
main_array_10 = normalized_arr[0]
normalized_arr = preprocessing.normalize([main_array_15])
main_array_15 = normalized_arr[0]
normalized_arr = preprocessing.normalize([main_array_20])
main_array_20 = normalized_arr[0]




node_10 = [(x / 10)*100 for x in range(11)]
node_15 = [(x / 15) * 100 for x in range(16)]
node_20 = [(x / 20)* 100 for x in range(21)]



plt.title("Probability Distribution Based Mitigation")

plt.xlabel("% Nodes Fail")
plt.ylabel("Normalised Sum")
plt.grid()
plt.plot(node_10 , main_array_10 , label = "Total Node - 10")
plt.plot(node_15 , main_array_15 , label = "Total Node - 15")
plt.plot(node_20 , main_array_20 , label = "Total Node - 20")
plt.legend()
plt.show()