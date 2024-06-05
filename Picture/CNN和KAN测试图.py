import matplotlib.pyplot as plt

x = range(0, 100)
y = range(0, 100)
CNN_test = [0.8587816953659058, 0.8782634735107422, 0.8879549503326416, 0.888053834438324, 0.8933939933776855, 0.8966574668884277, 0.8953718543052673, 0.8860759735107422, 0.883999228477478, 0.8957674503326416, 0.8966574668884277, 0.8991297483444214, 0.8943829536437988, 0.8987342119216919, 0.8826147317886353, 0.8984375, 0.8980419635772705, 0.9020965695381165, 0.8981408476829529, 0.888647198677063, 0.8973497152328491, 0.8975474834442139, 0.8973497152328491, 0.8968552350997925, 0.8938884735107422, 0.8925039768218994, 0.8921083807945251, 0.8965585827827454, 0.8856803774833679, 0.8934928774833679, 0.8936907052993774, 0.8910205960273743, 0.8901305794715881, 0.8933939933776855, 0.8914161324501038, 0.892405092716217, 0.8925039768218994, 0.8837025761604309, 0.8930973410606384, 0.8944818377494812, 0.8936907052993774, 0.8901305794715881, 0.8951740860939026, 0.8960641026496887, 0.8980419635772705, 0.8937895894050598, 0.894679605960846, 0.888053834438324, 0.8891416192054749, 0.8935918211936951, 0.8955696225166321, 0.8925039768218994, 0.8825158476829529, 0.8901305794715881, 0.8956685066223145, 0.8979430794715881, 0.8982397317886353, 0.8991297483444214, 0.8987342119216919, 0.8985364437103271, 0.8999208807945251, 0.8988330960273743, 0.8502769470214844, 0.892405092716217, 0.894679605960846, 0.8995253443717957, 0.8983386158943176, 0.8991297483444214, 0.899624228477478, 0.8991297483444214, 0.8988330960273743, 0.8985364437103271, 0.8782634735107422, 0.8917128443717957, 0.8982397317886353, 0.8967563509941101, 0.8965585827827454, 0.8982397317886353, 0.8997231125831604, 0.900217592716217, 0.9003164768218994, 0.8997231125831604, 0.8999208807945251, 0.900217592716217, 0.8762856125831604, 0.8904272317886353, 0.896459698677063, 0.8983386158943176, 0.8985364437103271, 0.899030864238739, 0.9003164768218994, 0.9000198245048523, 0.9001187086105347, 0.900217592716217, 0.8994264602661133, 0.9007120728492737, 0.8714398741722107, 0.8923062086105347, 0.895866334438324, 0.8994264602661133]

KAN_test =  [0.8244659900665283, 0.8540348410606384, 0.8608583807945251, 0.8692642450332642, 0.8694620728492737, 0.8686708807945251, 0.8706487417221069, 0.8721321225166321, 0.8751978278160095, 0.8747033476829529, 0.8769778609275818, 0.8735166192054749, 0.8795490860939026, 0.8776701092720032, 0.8735166192054749, 0.8794502019882202, 0.876780092716217, 0.8805379867553711, 0.8753955960273743, 0.8805379867553711, 0.8807357549667358, 0.8781645894050598, 0.8737144470214844, 0.8784612417221069, 0.8781645894050598, 0.8792523741722107, 0.8823180794715881, 0.8816258311271667, 0.8801424503326416, 0.8768789768218994, 0.8804391026496887, 0.8795490860939026, 0.880834698677063, 0.8814280033111572, 0.8795490860939026, 0.8778678774833679, 0.8771756291389465, 0.8779668211936951, 0.8810324668884277, 0.8806368708610535, 0.8775712251663208, 0.8797468543052673, 0.8811313509941101, 0.8788568377494812, 0.8798457384109497, 0.8759889602661133, 0.8805379867553711, 0.8771756291389465, 0.8810324668884277, 0.8799446225166321, 0.8791534900665283, 0.8817247152328491, 0.8792523741722107, 0.8810324668884277, 0.8789557218551636, 0.8769778609275818, 0.8791534900665283, 0.8789557218551636, 0.8816258311271667, 0.8776701092720032, 0.8747033476829529, 0.8793512582778931, 0.8804391026496887, 0.880241334438324, 0.8756922483444214, 0.8758900761604309, 0.8779668211936951, 0.876186728477478, 0.8803402185440063, 0.879054605960846, 0.880241334438324, 0.880241334438324, 0.8801424503326416, 0.8783623576164246, 0.8786590695381165, 0.879647970199585, 0.880834698677063, 0.8778678774833679, 0.873022198677063, 0.8799446225166321, 0.8781645894050598, 0.8825158476829529, 0.8788568377494812, 0.8816258311271667, 0.8787579536437988, 0.8787579536437988, 0.8779668211936951, 0.8805379867553711, 0.8803402185440063, 0.8788568377494812, 0.8794502019882202, 0.8778678774833679, 0.8754944801330566, 0.8793512582778931, 0.8774723410606384, 0.8793512582778931, 0.8807357549667358, 0.8797468543052673, 0.8822191953659058, 0.8762856125831604]

plt.plot(x, CNN_test, linestyle='-', label='CNN_test',color='r')
plt.plot(x, KAN_test, linestyle='-', label='KAN_test',color='b')

plt.yticks([0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95, 1])
# 添加标题和标签
plt.title('The Training Accuracy of MLP and KAN ')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# 显示图形
plt.savefig('CNN和KAN测试图')