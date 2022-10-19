import json
import numpy as np
import matplotlib.pyplot as plt


def plot_lc(title,data):
    fig, ax = plt.subplots()
    lc_all_train=[]
    lc_all_test=[]
    for pt in data:
        all_train_err=[]
        all_test_err=[]
        all_train_err.append(pt["trainsoap_mean MAE"])
        all_test_err.append(pt["testsoap_mean MAE"])
        lc_all_train.append(all_train_err)
        lc_all_test.append(all_test_err)
        ax.set_title(title)
        ax.boxplot(lc_all_train, patch_artist=True, boxprops={'facecolor': '#EE6A50'},
                   medianprops={'color':'firebrick'})
        ax.boxplot(lc_all_test,patch_artist=True, boxprops={'facecolor': '#7AC5CD'},
                   medianprops={'color':'purple'})
    ax.set_ylabel("MAE(eV)")
    return ax.figure.savefig(title+'.png')


result_add='/Users/gravityphy/PycharmProjects/defect_lithium/results_dicts/krr/'
pt1=json.load(open(result_add+'soap_mean-1000.json'))
pt2=json.load(open(result_add+'soap_mean-2000.json'))
pt3=json.load(open(result_add+'soap_mean-3000.json'))
pt4=json.load(open(result_add+'soap_mean-4000.json'))
pt5=json.load(open(result_add+'soap_mean-5000.json'))
data=[pt1,pt2,pt3,pt4,pt5]

plot_lc('SOAP',data)
