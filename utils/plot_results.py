# 并列柱状图
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# plt.rcParams['font.sans-serif']=['SimHei'] #设置字体以便支持中文
import numpy as np
from mrr_by_degree import dataset_reslust_by_degree

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30}) # 改变所有字体大小，改变其他性质类似

linewidth = 3
markersize = 10


# '''Results w.r.t training ratio'''
# x = np.arange(5)
# hit1_pale = np.array([8.15, 8.74, 10.20, 11.68, 14.36])
# hit1_sea = np.array([8.50, 10.60, 12.27, 14.97, 15.61])
# hit1_deguil = np.array([7.22, 9.67, 13.51, 16.46, 20.11])
#
# mrr_pale = np.array([14.79, 16.22, 18.02, 20.06, 22.31])
# mrr_sea = np.array([15.19, 17.40, 19.23, 22.18, 23.09])
# mrr_deguil = np.array([13.55, 17.20, 21.57, 25.01, 28.23])
#
# # y_start = 6
# tick_label = [i*10 for i in range(2, 7)]
# linewidth = 4
# markersize = 13
#
# plt.figure(figsize=(8, 7))
# # 绘制并列柱状图
# plt.plot(x, hit1_deguil, '*-', color='#f27970', label='DegUIL', linewidth=linewidth, markersize=markersize)
# plt.plot(x, hit1_sea, '*-', color='#2f7fc1', label='SEA', linewidth=linewidth, markersize=markersize)
# plt.plot(x, hit1_pale, '*-', color='#b883d4', label='PALE', linewidth=linewidth, markersize=markersize)
# plt.plot(x, mrr_deguil, 'o-', color='#f27970', label='DegUIL', linewidth=linewidth, markersize=markersize)
# plt.plot(x, mrr_sea, 'o-', color='#2f7fc1', label='SEA', linewidth=linewidth, markersize=markersize)
# plt.plot(x, mrr_pale, 'o-', color='#b883d4', label='PALE', linewidth=linewidth, markersize=markersize)
#
# legend_methods = [Line2D([0], [0], color='#f27970', lw=4, label='DegUIL'),
#                   Line2D([0], [0], color='#2f7fc1', lw=4, label='SEA'),
#                   Line2D([0], [0], color='#b883d4', lw=4, label='PALE'),
#                   Line2D([0], [0], marker='*', label='Hit@1', color='k', linewidth=0, markerfacecolor='k', markersize=12),
#                   Line2D([0], [0], marker='o', label='MRR', color='k', linewidth=0, markerfacecolor='k', markersize=10)
#                   ]
#
#
# plt.legend(handles=legend_methods, ncol=2, columnspacing=0.5, loc='lower right', fontsize=24)  # 显示图例，即label
# # plt.legend(loc='best', fontsize=24)  # 显示图例，即label
# # plt.xlim(6, 25)
# plt.ylim(0, 30)
# plt.xticks(x, tick_label)
# plt.xlabel('Training ratio(%)')
# plt.ylabel('Performance(%)', labelpad=8)
# plt.tight_layout(pad=0.2, h_pad=None, w_pad=None, rect=None)
# plt.savefig('perf_by_train_ratio.pdf', bbox_inches='tight')
# plt.show()

'''Results by degree'''
plt.figure(figsize=(13, 6))
plt.subplot(121)
dataset_reslust_by_degree('FT')
legend = [Patch(color='#F0988C', label='DegUIL'),
          Patch(color='#82B0D2', label='PALE')]
plt.legend(handles=legend, loc='upper left', fontsize=26)

plt.subplot(122)
dataset_reslust_by_degree('DBLP')

plt.savefig('./mrr_comparison_by_degree.pdf', bbox_inches='tight')
plt.show()
