import sys
sys.path.append('../coco-caption/')


from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
import augment
import csv
import glob

metrics = {'meteor' : Meteor(), 'rouge':Rouge()}

def process_input(dev, aug_dev, wi):
    gts = {}
    res = {}
    size = min(len(dev), len(aug_dev))
    for i in range(size):
        gts[i] = [wi.print_seq(dev[i])]
        res[i] = [wi.print_seq(aug_dev[i])]
        
    return gts, res


def evaluate(gts, res, metrics):
    result = {}
    for m in metrics:
        avg_score, scores = metrics[m].compute_score(gts, res)
        result[m] = avg_score
    return result


def evaluate_model(dir_name, dev, wi, metrics):
    aug_train, aug_dev = augment.load_dataset(dir_name, 0.0, len(dev[0]), len(dev[0]), wi, 25, 15)
    gts, res = process_input(dev[1], aug_dev[1], wi)
    return evaluate(gts, res, metrics)
    

def evaluate_all_models(dev, wi, metrics):
    csvf =  open('mt_metrics.csv', 'wb')
    writer = csv.writer(csvf)
    writer.writerow(['Model'] + metrics.keys())
    dirs = glob.glob('models/real*')

    for dir in dirs:
        result = evaluate_model(dir, dev, wi, metrics)
        model=  dir.split('real')[1]
        print model, result
        writer.writerow([model] + result.values())    
