import matplotlib.pyplot as plt

main_array_15 = [0.8655624389648438, 0.3767526149749756, 0.3526163101196289, 0.37534308433532715, 0.3168487548828125, 0.28359174728393555, 0.2337639331817627, 0.27099609375, 0.24103975296020508, 0.21013903617858887, 0.22745299339294434, 0.20376014709472656, 0.13222146034240723, 0.14896726608276367, 0.1251382827758789, 0.22132182121276855]
main_array_10 = [0.813542366027832, 0.27394986152648926, 0.2268538475036621, 0.18676280975341797, 0.2247917652130127, 0.17294836044311523, 0.15031862258911133, 0.14691543579101562, 0.15618181228637695, 0.08150124549865723, 0.09121441841125488]
main_array_20 = [1.1826097965240479, 0.5418133735656738, 0.4459869861602783, 0.32338953018188477, 0.32566094398498535, 0.3371548652648926, 0.25885534286499023, 0.24915027618408203, 0.2501823902130127, 0.242875337600708, 0.22600030899047852, 0.24275708198547363, 0.16894912719726562, 0.20495104789733887, 0.17049336433410645, 0.12111592292785645, 0.13261961936950684, 0.18824553489685059, 0.16649341583251953, 0.15215182304382324, 0.11341667175292969]

node_10 = [(x / 10)*100 for x in range(11)]
node_15 = [(x / 15) * 100 for x in range(16)]
node_20 = [(x / 20)* 100 for x in range(21)]



plt.title("0 Based Mitigation")

plt.xlabel("% Nodes Fail")
plt.ylabel("Latency(s)")
plt.grid()
plt.plot(node_10 , main_array_10 , label = "Total Node - 10")
plt.plot(node_15 , main_array_15 , label = "Total Node - 15")
plt.plot(node_20 , main_array_20 , label = "Total Node - 20")
plt.legend()
plt.show()