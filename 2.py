pred_name = "pred0.txt"
f = open(pred_name,"w")
for i in xrange(10):
    f.writelines(str(i) + '\n')