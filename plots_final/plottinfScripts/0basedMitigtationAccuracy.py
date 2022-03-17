import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np 


main_array_10 = [42.8, 39.4, 33.0, 25.4, 22.0, 21.6, 15.2, 13.4, 8.0, 4.8, 0.0]
main_array_15 = [62.8, 58.8, 55.4, 46.8, 40.4, 39.4, 37.4, 29.4, 33.6, 22.2, 23.6, 12.0, 12.6, 6.8, 4.4, 0.0]
main_array_20 = [91.6, 90.6, 79.8, 76.8, 77.6, 69.2, 66.0, 56.8, 57.4, 51.2, 44.0, 42.0, 37.6, 27.4, 23.8, 21.4, 20.2, 10.6, 7.8, 5.4, 0.0]
normalized_arr = preprocessing.normalize([main_array_10])
main_array_10 = normalized_arr[0]
normalized_arr = preprocessing.normalize([main_array_15])
main_array_15 = normalized_arr[0]
normalized_arr = preprocessing.normalize([main_array_20])
main_array_20 = normalized_arr[0]

node_10 = [(x / 10)*100 for x in range(11)]
node_15 = [(x / 15) * 100 for x in range(16)]
node_20 = [(x / 20)* 100 for x in range(21)]



plt.title("0 Based Mitigation")

plt.xlabel("% Nodes Fail")
plt.ylabel("Normalised Sum")
plt.grid()
plt.plot(node_10 , main_array_10 , label = "Total Node - 10")
plt.plot(node_15 , main_array_15 , label = "Total Node - 15")
plt.plot(node_20 , main_array_20 , label = "Total Node - 20")
plt.legend()
plt.show()