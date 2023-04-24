import timm
print(timm.list_models('*res*'))
# import pandas as pd
# import numpy as np
# x= 'log_value/tf_efficientnet_b6_ap.npy'
# x= np.load(x)
# # print(x)
# # print(len(x))
# # y= 'log_ml/lgbm.npy'
# # y= np.load(y)
# # # print(len(y[0]))
# # s= np.mean(y, axis=0)
# # print(len(s))
# # print(s)


# # total= x+s
# total= np.argmax(x, 1)
# print(total)
# # task= np.argmax(y, 0)
# # print(task)
# sub= pd.read_csv('sample_submission.csv')
# sub['N_category']= total

# sub.to_csv('only_img.csv', index=False)


# # res= np.argmax(x, axis=1)
# # print(res)