'''
Created on 28 Feb 2019
@author: camilo thorne
'''


import glob, numpy as np
from nltk.metrics import scores
import codecs   


def avgEval(inpath1, inpath2):
    
    print('\n=============================')      
    print('NER evaluation (single entity class/mention-level, no offsets/document-level, avg. of abstract-level)')
    print('=============================')  
    print('==> gold', inpath1)
    print('==> pred', inpath2)
    print('=============================')   
    recs  = []
    pres  = []
    fscs  = []   
    for filename1 in glob.glob(inpath1+"/*ann"):
        filen1   = filename1.split('/')[len(filename1.split('/'))-1]
        for idx,filename2 in enumerate(glob.glob(inpath2+"/*ann")):
            filen2   = filename2.split('/')[len(filename2.split('/'))-1]
            if filen1 == filen2:
                preds = set([])
                refrs = set([])
                file1 = codecs.open(filename1, 'r', encoding='utf-8')
                file2 = codecs.open(filename2, 'r', encoding='utf-8')
                for line1 in file1.readlines():
                    if len(line1.split('\t')) > 1:
                        men1  = line1.split('\t')[2].strip()
                        off1  = '-'.join(line1.split('\t')[1].split(' ')[:1]).strip()
                        gold = men1 + '_' + off1  
                        refrs.add(gold)
                for line2 in file2.readlines():
                    if len(line2.split('\t')) > 1:
                        men2  = line2.split('\t')[2].strip()
                        off2  = '-'.join(line2.split('\t')[1].split(' ')[:1]).strip()
                        pred = men2 + '_' + off2  
                        #print('\t', gold, '--', pred)
                        preds.add(pred)        
                if len(preds)>0 and len(refrs)>0:       
                    rec = scores.recall(refrs, preds)
                    pre = scores.precision(refrs, preds)
                    fsc = scores.f_measure(refrs, preds)
                else:
                    rec = 0
                    pre = 0
                    fsc = 0
                recs.append(rec)
                pres.append(pre)
                fscs.append(fsc)
    print('average \t R={R} \t P={P} \t F1={F}'.format(R=str(np.mean(recs)),P=str(np.mean(pres)),F=str(np.mean(fscs))))        
    print('=============================\n')      



def macroEval(inpath1, inpath2):
    
    print('\n=============================')      
    print('NER evaluation (single entity class/mention-level, no offsets/document-level, corpus-level)')
    print('=============================')  
    print('==> gold', inpath1)
    print('==> pred', inpath2)
    print('=============================')    
    for filename1 in glob.glob(inpath1+"/*ann"):
        filen1   = filename1.split('/')[len(filename1.split('/'))-1]
        for filename2 in glob.glob(inpath2+"/*ann"):
            filen2   = filename2.split('/')[len(filename2.split('/'))-1]
            if filen1 == filen2:
                preds = set([])
                refrs = set([])
                file1 = codecs.open(filename1, 'r', encoding='utf-8')
                file2 = codecs.open(filename2, 'r', encoding='utf-8')
                for line1 in file1.readlines():
                    if len(line1.split('\t')) > 1:
                        men1  = line1.split('\t')[2].strip()
                        off1  = '-'.join(line1.split('\t')[1].split(' ')[:1]).strip()
                        gold = men1 + '_' + off1  
                        refrs.add(gold)
                for line2 in file2.readlines():
                    if len(line2.split('\t')) > 1:
                        men2  = line2.split('\t')[2].strip()
                        off2  = '-'.join(line2.split('\t')[1].split(' ')[:1]).strip()
                        pred = men2 + '_' + off2  
                        preds.add(pred)        
    rec = scores.recall(refrs, preds)
    pre = scores.precision(refrs, preds)
    fsc = scores.f_measure(refrs, preds)                      
    print('macro \t R={R} \t P={P} \t F1={F}'.format(R=str(rec),P=str(pre),F=str(fsc)))        
    print('=============================\n')  

    

def avgOffEval(inpath1, inpath2):
    
    print('\n=============================')      
    print('NER evaluation (single entity class/mention-level, full/offsets, avg. of abstract-level)')
    print('=============================')  
    print('==> gold', inpath1)
    print('==> pred', inpath2)
    print('=============================')   
    recs  = []
    pres  = []
    fscs  = []   
    for filename1 in glob.glob(inpath1+"/*ann"):
        filen1   = filename1.split('/')[len(filename1.split('/'))-1]
        for filename2 in glob.glob(inpath2+"/*ann"):
            filen2   = filename2.split('/')[len(filename2.split('/'))-1]
            if filen1 == filen2:
                preds = set([])
                refrs = set([])
                file1 = codecs.open(filename1, 'r', encoding='utf-8')
                file2 = codecs.open(filename2, 'r', encoding='utf-8')
                for line1 in file1.readlines():
                    if len(line1.split('\t')) > 1:
                        men1  = line1.split('\t')[2].strip()
                        off1  = '-'.join([w.strip() for w in line1.split('\t')[1].split(' ')])
                        gold = men1 + '_' + off1
                        refrs.add(gold)
                for line2 in file2.readlines():
                    if len(line2.split('\t')) > 1:
                        men2  = line2.split('\t')[2].strip()
                        off2  = '-'.join([w.strip() for w in line2.split('\t')[1].split(' ')])
                        pred = men2 + '_' + off2  
                        preds.add(pred)              
                if len(preds)>0 and len(refrs)>0:       
                    rec = scores.recall(refrs, preds)
                    pre = scores.precision(refrs, preds)
                    fsc = scores.f_measure(refrs, preds)
                else:
                    rec = 0
                    pre = 0
                    fsc = 0
                recs.append(rec)
                pres.append(pre)
                fscs.append(fsc)                           
    print('average \t R={R} \t P={P} \t F1={F}'.format(R=str(np.mean(recs)),P=str(np.mean(pres)),F=str(np.mean(fscs))))        
    print('=============================\n')  


def macroOffEval(inpath1, inpath2):
    
    print('\n=============================')      
    print('NER evaluation (single entity class/mention-level, full/offsets, corpus-level)')
    print('=============================')  
    print('==> gold', inpath1)
    print('==> pred', inpath2)
    print('=============================')   
    preds = set([])
    refrs = set([])    
    for filename1 in glob.glob(inpath1+"/*ann"):
        filen1   = filename1.split('/')[len(filename1.split('/'))-1]
        for filename2 in glob.glob(inpath2+"/*ann"):
            filen2   = filename2.split('/')[len(filename2.split('/'))-1]
            if filen1 == filen2:
                file1 = codecs.open(filename1, 'r', encoding='utf-8')
                file2 = codecs.open(filename2, 'r', encoding='utf-8')
                for line1 in file1.readlines():
                    if len(line1.split('\t')) > 1:
                        men1  = line1.split('\t')[2].strip()
                        off1  = '-'.join([w.strip() for w in line1.split('\t')[1].split(' ')])
                        gold = men1 + '_' + off1
                        refrs.add(gold)
                for line2 in file2.readlines():
                    if len(line2.split('\t')) > 1:
                        men2  = line2.split('\t')[2].strip()
                        off2  = '-'.join([w.strip() for w in line2.split('\t')[1].split(' ')])
                        pred = men2 + '_' + off2  
                        preds.add(pred)
    rec = scores.recall(refrs, preds)
    pre = scores.precision(refrs, preds)
    fsc = scores.f_measure(refrs, preds)                      
    print('macro \t R={R} \t P={P} \t F1={F}'.format(R=str(rec),P=str(pre),F=str(fsc)))        
    print('=============================\n')


if __name__ == "__main__":
    
    print('(NCBI Gold vs. NERDs)\n')
  
    # gold = '/media/druv022/Data1/Masters/Thesis/Eval/Testing/Train'
    # term = '/media/druv022/Data1/Masters/Thesis/Eval/Testing/Test'

    gold = '/media/druv022/Data1/Masters/Thesis/Data/Experiment/Train'
    term = '/media/druv022/Data1/Masters/Thesis/Data/Experiment/Test'
 
    avgEval(gold, term)
    macroEval(gold,term)
    avgOffEval(gold, term)
    macroOffEval(gold, term)
    