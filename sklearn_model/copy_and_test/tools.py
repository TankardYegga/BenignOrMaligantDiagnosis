# -*- encoding: utf-8 -*-
"""
@File    : tools.py
@Time    : 11/12/2021 6:26 PM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
from sklearn.svm import SVC
import sklearn
import inspect
import re
import matplotlib.pyplot as plt
sklearn.set_config(display='diagram')


def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)


def find_all_properties(object):
    for prop in dir(object):
        print(prop, " ", str(type(eval(varname(object) + '.' + prop))))


## This first part is PURELY bc I wanted color.
from IPython.display import Markdown, display


def printmd(string, color=None):
    if '__' in string:
        colorstr = "__<span style='color:{}'>&#95;{}</span>__".format(color, string)
    else:
        colorstr = "__<span style='color:{}'>{}</span>__".format(color, string)
    display(Markdown(colorstr))


# The actual solution.
# If you don't want formatting/don't want to deal with the above code,
# replace 'printmd' with print.
def describe(model):
    # get name of model
    model_name = [objname for objname, oid in globals().items() if id(oid) == id(model)][0]

    # create list, sort, but __doc__ at front
    calls = sorted([model_name + '.' + i for i in dir(PCA)])
    doc_index = calls.index(model_name + '.__doc__')
    calls.pop(doc_index)
    calls = [model_name + '.__doc__'] + calls

    # print
    for c in calls:
        try:
            if '__' in c:
                printmd(f'{c}', color='blue')
            else:
                printmd(c, color='blue')

            printmd(str(eval(c + '()')))
        except Exception as e:
            if 'is not callable' in str(e).lower():
                try:
                    printmd(str(eval(c)))
                except Exception as e:
                    print(e)
            else:
                print(e)
        printmd('/' * 20, color='gray')


# if __name__ == '__main__':
#
#     svc_model = SVC(kernel='linear')
#     # find_all_properties(svc_model, 'svc_model')
#     find_all_properties(svc_model)
#     printmd("*italics*, **bold**", color='blue')
#     # describe(svc_model)
