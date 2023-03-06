import glob
f = glob.glob('./*')
print (len(f))

f = [x for x in f if ".ipynb" not in x]
print (len(f))

print ("Scenario, Validation_loss, Naive_model_loss")
for fi in f:
    with open(fi) as f2:
        count = 0 
        for row in f2:
#             listed = row.strip().split()
            count += 1
            
            if count == 18 :#  and "hh" in fi: # or count == 5:
                listed = row.strip().split(",")
#                 print (listed, f2)
                print (fi.replace("./val_csv_", ""), listed[-2], listed[-4], sep=",")
