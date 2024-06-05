import matplotlib.pyplot as plt

x = range(0, 100)
y = range(0, 100)
CNN_train_loss = [0.4244313713139308, 0.2917146061910495, 0.25459939360555045, 0.2297610328602257, 0.2124724380656092, 0.1966104982281799, 0.17864250247158223, 0.1670271474828344, 0.15426523300376274, 0.14189494340055026, 0.1314902723645732, 0.12072351574897766, 0.11172111629486592, 0.1014497921680972, 0.09698249324203045, 0.09125860452429573, 0.07955026716740528, 0.07449435504244716, 0.06772213248111038, 0.06529247316954805, 0.05736066266370099, 0.05366734611486067, 0.04742364113781053, 0.04756571961614464, 0.043711712229440904, 0.04075318546409705, 0.038689647045637814, 0.03492059906982362, 0.03225573654503012, 0.030675984826733244, 0.028857473351522042, 0.027593674590445753, 0.025434980380422335, 0.03038090541422653, 0.02198905682465288, 0.018539218335791724, 0.02017742131939972, 0.022192051031141243, 0.02805911775019123, 0.01988924650807005, 0.012687149176399893, 0.013473842772089246, 0.03350251212493697, 0.014555361168061906, 0.012981266811438429, 0.01612528954828178, 0.01815807352215846, 0.01646758444897179, 0.020463755282797792, 0.011680899737234006, 0.007438872191833574, 0.010818079637456884, 0.025823529275543274, 0.018783819260148167, 0.007205537113640061, 0.0028837903396291996, 0.001309657876783394, 0.0007441902269067159, 0.000571800358644349, 0.0004947375572592602, 0.00045143610984670307, 0.0004169056719818861, 0.00633612451743201, 0.13167778806967426, 0.013982135107490554, 0.003359820947350402, 0.001533921131355835, 0.0010601833911621566, 0.0008539207453001986, 0.0007293604033279802, 0.0006326180595436941, 0.0005863284303579353, 0.032129172388227656, 0.08080419164790369, 0.011934256322110004, 0.0034396025957961654, 0.001508879846571061, 0.001066500362638545, 0.0007285446849688123, 0.0006247843753907389, 0.0005528649891977872, 0.0005018567955401887, 0.0004669830894598098, 0.0004771123590284903, 0.048927634259870054, 0.06467029472851732, 0.009777863177461927, 0.0026144979424578057, 0.0010909250590700542, 0.000754757680861192, 0.0006122889191677759, 0.0005470236367694954, 0.00047238178088017965, 0.00041902838719150884, 0.000384328700080792, 0.0003669230550345627, 0.0012101767923843527, 0.1081377492402432, 0.010289440299374368, 0.003391089356549805]

KAN_train_loss = [0.6447327192912478, 0.38579556133065906, 0.3418883963434427, 0.3182253381336676, 0.3007666337083398, 0.29165989033448925, 0.28228722055202354, 0.2771679826700357, 0.2692778025354658, 0.2635213600229353, 0.26128497063668804, 0.2568003361794486, 0.2537170816967482, 0.2505466050939011, 0.24650442104603945, 0.24537414657091028, 0.24214204715322585, 0.24056568101588596, 0.23839798693590836, 0.23577837673013907, 0.23531655265069973, 0.23295845142178445, 0.23343514144293534, 0.2286035074894108, 0.22917610211476588, 0.2271659438575763, 0.22587848313327538, 0.22369769685812343, 0.2246361409963321, 0.22268653462436408, 0.22100378639662444, 0.22102316316447532, 0.21969333194148566, 0.21877583310103366, 0.2163653936403901, 0.21698833862220301, 0.21831928101429807, 0.21607887252434485, 0.215724399126669, 0.21403046138187462, 0.21381886031772537, 0.2137280706721328, 0.2124757102208097, 0.21231680697024757, 0.21132981986887672, 0.21052510642420763, 0.20966341778604206, 0.2105770324831451, 0.20882416746890875, 0.20997845729403913, 0.2087791347935764, 0.2085956040062884, 0.20733326266823546, 0.20876909600201446, 0.20605828773492435, 0.20686754678040425, 0.20653817100502025, 0.20674615010206124, 0.2064038350193231, 0.2052008692802651, 0.20530581509253618, 0.20498481030657348, 0.20431665669499174, 0.20349215904373857, 0.20461679607439143, 0.2035466679759117, 0.20333911316481226, 0.20284019669553618, 0.20420577688448466, 0.20408065547185666, 0.20096571911880964, 0.20183303542355738, 0.2020468719796077, 0.20128602884026733, 0.2007596194585249, 0.20120327802164467, 0.19989692608811963, 0.20004295525965152, 0.20083758454205894, 0.19925017758949734, 0.199299660175721, 0.19773679553890533, 0.19983111339400828, 0.1990195300533319, 0.19881494595869773, 0.19900365806083437, 0.19826599776045853, 0.19918296897589272, 0.19852107720397938, 0.19962322859685305, 0.198226948298498, 0.196073251460661, 0.19758017059328206, 0.19580003206949753, 0.19545793352223662, 0.19695973335934092, 0.1973157526968893, 0.19745974302260097, 0.19743982286277864, 0.19421645638340318]


plt.plot(x, CNN_train_loss, linestyle='-', label='CNN_train_loss',color='r')
plt.plot(x, KAN_train_loss, linestyle='-', label='KAN_train_loss',color='b')

plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
# 添加标题和标签
plt.title('The Training loss of CNN and KAN ')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# 显示图形
plt.savefig('CNN和KAN训练损失图')