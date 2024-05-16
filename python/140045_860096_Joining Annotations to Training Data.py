import dill

data = dill.load(open("stock_price_training_slices.pkl", "r"))

print data[1]

scoreText = open("score_sheet_complete.txt", "r").readlines()

split_scores = [x.split() for x in scoreText]

scores = []
for x in split_scores:
    scores.append([x[0], x[1], int(x[2])])

annotated_data = []
for datum, score in zip(data, scores):
    #print datum[0], datum[1], "|", score[0], score[1]
    if datum[0] == score[0] and datum[1] == score[1]:
        annotated_data.append((datum[0], datum[1], score[2], datum[2]))
    else:
        print "got one wrong"

annotated_data[:5]

dill.dump(annotated_data, open("annotated_traning_data.pkl","w"))

