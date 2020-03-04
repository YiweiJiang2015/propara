import sys, collections, pylev
from stemming.porter2 import stem

#--------------------------------------------------------------
# Author: Scott Wen-tau Yih
# Usage: evalQA.py para-ids gold-labels system-predictions
# example usage: python propara/eval/evalQA.py tests/fixtures/eval/para_id.test.txt tests/fixtures/eval/gold_labels.test.tsv tests/fixtures/eval/sample.model.test_predictions.tsv 
#--------------------------------------------------------------

# Data structure for Labels
'''
  PID -> [TurkerLabels]
  TurkerLabels = [TurkerQuestionLabel1, TurkerQuestionLabel2, ... ]  # labels on the same paragraph from the same Turker
  TurkerQuestionLabel -> (SID, Participant, Type, From, To)
'''
TurkerQuestionLabel = collections.namedtuple('TurkerQuestionLabel', 'sid participant event_type from_location to_location')


# Data structure for Predictions
'''
  PID -> Participant -> SID -> PredictionRecord
'''
PredictionRecord = collections.namedtuple('PredictionRecord', 'pid sid participant from_location to_location')

# Fixing tokenization mismatch while alinging participants
manual_participant_map = { 'alternating current':'alternate current', 'fixed nitrogen':'nitrogen',
                           'living things':'live thing', 'red giant star':'star', 'refrigerent liquid':'liquid',
                           'remains of living things':'remains of live thing',
                           "retina's rods and cones":"retina 's rod and cone" } #, 'seedling':'seed'}

#----------------------------------------------------------------------------------------------------------------

def compare_to_gold_labels(participants, system_participants):
    ret = []
    found = False
    for p in participants:
        p = p.lower()
        if p in system_participants:
            ret.append(p)
            continue
        for g in system_participants:
            if (pylev.levenshtein(p,g) < 3):
                #print (p, "===", g)
                ret.append(g)
                found = True
        if not found:
            if p in manual_participant_map:
                ret.append(manual_participant_map[p])
            #else:
            #    print("cannot find", p, system_participants)
    return ret

def preprocess_locations(locations):
    ret = []
    for loc in locations:
        if loc == '-':
            ret.append('null')
        elif loc == '?':
            ret.append('unk')
        else:
            ret.append(loc)
    return ret


def preprocess_question_label(sid, participant, event_type, from_location, to_location, system_participants=None):

    # check if there are multiple participants grouped together
    participants = [x.strip() for x in participant.split(';')]

    # check if there are multiple locations grouped together
    from_locations = preprocess_locations([x.strip() for x in from_location.split(';')])

    # check if there are multiple locations grouped together
    to_locations = preprocess_locations([x.strip() for x in to_location.split(';')])

    #print(participant, participants, system_participants)
    if system_participants != None: # check if the participants are in his list
        participants = compare_to_gold_labels(participants, system_participants)
        #print("legit_participants =", participants)

    #print(from_location, from_locations)
    #print(to_location, to_locations)

    return  [TurkerQuestionLabel(sid, p, event_type, floc, tloc) for p in participants
                                                                 for floc in from_locations
                                                                 for tloc in to_locations]

#----------------------------------------------------------------------------------------------------------------

'''
  Read the gold file containing all records where an entity undergoes some state-change: create/destroy/move.
'''
def readLabels(full_label_file, set_Para_id=None, location_labels=None):
    full_Label = open(full_label_file)
    full_Label.readline()    # skip header
    ret = {}
    TurkerLabels = []
    for ln in full_Label:
        f = ln.rstrip().split('\t')
        if len(f) == 0 or len(f) == 1:
            if not set_Para_id or pid in set_Para_id:
                if pid not in ret:
                    ret[pid] = []
                ret[pid].append(TurkerLabels)
            TurkerLabels = []
        elif len(f) != 11:
            sys.stderr.write("Error: the number of fields in this line is irregular: " + ln)
            sys.exit(-1)
        else:
            if f[1] == '?': continue
            pid, sid, participant, event_type, from_location, to_location = int(f[0]), int(f[1]), f[3], f[4], f[5], f[6]

            if location_labels and set_Para_id and pid in set_Para_id:
                #print("pid=", pid)
                TurkerLabels += preprocess_question_label(sid, participant, event_type, from_location, to_location, location_labels[pid].keys())
            else:
                TurkerLabels += preprocess_question_label(sid, participant, event_type, from_location, to_location)

            #TurkerLabels += (TurkerQuestionLabel(sid, participant, event_type, from_location, to_location))
    return ret

#----------------------------------------------------------------------------------------------------------------

def readPredictions(fnPred):
    ret = {}

    for ln in open(fnPred):
        f = ln.rstrip().split('\t')
        pid, sid, participant, from_location, to_location = int(f[0]), int(f[1]), f[2], f[3], f[4]

        if pid not in ret:
            ret[pid] = {}
        dtPartPred = ret[pid]

        if participant not in dtPartPred:
            dtPartPred[participant] = {}

        dtPartPred[participant][sid] = PredictionRecord(pid, sid, participant, from_location, to_location)

    return ret

#----------------------------------------------------------------------------------------------------------------

def readGold(fn):
    """
    Read location labels without status description (move, created, destroyed)
    :param fn:
    :return:
    """

    labels_Para = {} # dcit of labels sorted by paragraph id
    for ln in open(fn):
        f = ln.rstrip().split('\t')
        parId, sentId, participant, before_after, labels = int(f[0]), int(f[1]), f[2], f[3], f[4:]

        if (before_after != "before") and (before_after != "after"):
            print("Error:", ln)
            sys.exit(-1)

        if sentId == 1 and before_after == "before":
            statusId = 0 # before_location matters only in the first step
        elif before_after == "before":
            continue  # skip this line
        else:
            statusId = sentId

        if parId not in labels_Para:
            labels_Para[parId] = {}
        labels_Para_sub = labels_Para[parId] # labels_Para_sub: sub-dict of labels sorted by entities within a paragraph
                                             # each sub-dict has step+1 entries
        if participant not in labels_Para_sub:
            labels_Para_sub[participant] = {statusId: labels}
        else:
            labels_Para_sub[participant][statusId] = labels
    return labels_Para

#----------------------------------------------------------------------------------------------------------------

def findAllParticipants(lstTurkerLabels):
    setParticipants = set()
    for turkerLabels in lstTurkerLabels:
        for x in turkerLabels:
            setParticipants.add(x.participant)
    return setParticipants

def findCreationStep(prediction_records):
    steps = sorted(prediction_records, key=lambda x: x.sid)
    #print("steps:", steps)

    # First step. This line filters those entities that exist before the process already
    if steps[0].from_location != 'null':    # not created (exists before the process)
        return -1
    # 
    for s in steps:
        if s.to_location != 'null':
            return s.sid
    return -1   # never exists

def findDestroyStep(prediction_records):
    steps = sorted(prediction_records, key=lambda x: x.sid, reverse=True)
    #print("steps:", steps)

    # last step
    if steps[0].to_location != 'null':  # not destroyed (exists after the process)
        return -1

    for s in steps:
        if s.from_location != 'null':
            return s.sid

    return -1   # never exists

def location_match(p_loc, g_loc):
    if p_loc == g_loc:
        return True

    p_string = ' %s ' % ' '.join([stem(x) for x in p_loc.lower().replace('"','').split()])
    g_string = ' %s ' % ' '.join([stem(x) for x in g_loc.lower().replace('"','').split()])

    if p_string in g_string:
        #print ("%s === %s" % (p_loc, g_loc))
        return True

    return False

def findMoveSteps(prediction_records):
    ret = []
    steps = sorted(prediction_records, key=lambda x: x.sid)
    # print(steps)
    for s in steps:
        if s.from_location != 'null' and s.to_location != 'null' and s.from_location != s.to_location:
            ret.append(s.sid)

    return ret

#----------------------------------------------------------------------------------------------------------------

# Q1: Is participant X created during the process?
# total: sum(num_entity per paragraph) 所有段落中出现的实体数目总和
def Q1(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        setParticipants = findAllParticipants(labels[pid])
        # find predictions
        be_created = {}
        for participant in setParticipants:
            pred_creation_step = findCreationStep(predictions[pid][participant].values())
            be_created[participant] = (pred_creation_step != -1)
        for turkerLabels in labels[pid]:
            # labeled as created participants
            lab_created_participants = [x.participant for x in turkerLabels if x.event_type == 'create']
            for participant in setParticipants:
                tp += int(be_created[participant] and (participant in lab_created_participants))
                fp += int(be_created[participant] and (participant not in lab_created_participants))
                tn += int(not be_created[participant] and (participant not in lab_created_participants))
                fn += int(not be_created[participant] and (participant in lab_created_participants))
    return tp,fp,tn,fn

# Q2: Participant X is created during the process. At which step is it created?
# total: sum(created entities per paragraph in labels) label中created实体数目的总和
def Q2(labels, predictions):
    tp = fp = tn = fn = 0.0
    # find all created participants and their creation step
    for pid,lstTurkerLabels in labels.items():
        for turkerLabels in lstTurkerLabels:
            for x in [x for x in turkerLabels if x.event_type == 'create']:
                pred_creation_step = findCreationStep(predictions[pid][x.participant].values())
                tp += int(pred_creation_step != -1 and pred_creation_step == x.sid)
                fp += int(pred_creation_step != -1 and pred_creation_step != x.sid)
                fn += int(pred_creation_step == -1)
    return tp,fp,tn,fn

# Q3: Participant X is created at step Y, and the initial location is known. Where is the participant after it is created?
#
def Q3(labels, predictions):
    tp = fp = tn = fn = 0.0
    # find all created participants and their creation step
    for pid,lstTurkerLabels in labels.items():
        for turkerLabels in lstTurkerLabels:
            for x in [x for x in turkerLabels if x.event_type == 'create' and x.to_location != 'unk']:
                pred_loc = predictions[pid][x.participant][x.sid].to_location
                tp += int(pred_loc != 'null' and pred_loc != 'unk' and location_match(pred_loc, x.to_location))
                fp += int(pred_loc != 'null' and pred_loc != 'unk' and not location_match(pred_loc, x.to_location))
                fn += int(pred_loc == 'null' or pred_loc == 'unk')
    return tp, fp, tn, fn

#----------------------------------------------------------------------------------------------------------------

# Q4: Is participant X destroyed during the process?
# total: sum(num_entity per paragraph)
def Q4(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        setParticipants = findAllParticipants(labels[pid])
        # find predictions
        be_destroyed = {}
        for participant in setParticipants:
            pred_destroy_step = findDestroyStep(predictions[pid][participant].values())
            be_destroyed[participant] = (pred_destroy_step != -1)
        for turkerLabels in labels[pid]:
            # labeled as destroyed participants
            lab_destroyed_participants = [x.participant for x in turkerLabels if x.event_type == 'destroy']
            for participant in setParticipants:
                tp += int(be_destroyed[participant] and (participant in lab_destroyed_participants))
                fp += int(be_destroyed[participant] and (participant not in lab_destroyed_participants))
                tn += int(not be_destroyed[participant] and (participant not in lab_destroyed_participants))
                fn += int(not be_destroyed[participant] and (participant in lab_destroyed_participants))
    return tp,fp,tn,fn

# Q5: Participant X is destroyed during the process. At which step is it destroyed?
# total: sum(destroyed entities per paragraph in labels)
def Q5(labels, predictions):
    tp = fp = tn = fn = 0.0
    # find all destroyed participants and their destroy step
    for pid, lstTurkerLabels in labels.items():
        for turkerLabels in lstTurkerLabels:
            for x in [x for x in turkerLabels if x.event_type == 'destroy']:
                    pred_destroy_step = findDestroyStep(predictions[pid][x.participant].values())
                    tp += int(pred_destroy_step != -1 and pred_destroy_step == x.sid)
                    fp += int(pred_destroy_step != -1 and pred_destroy_step != x.sid)
                    fn += int(pred_destroy_step == -1)
    return tp,fp,tn,fn

# Q6: Participant X is destroyed at step Y, and its location before destroyed is known. Where is the participant right before it is destroyed?
def Q6(labels, predictions):
    tp = fp = tn = fn = 0.0
    # find all created participants and their destroy step
    for pid,lstTurkerLabels in labels.items():
        for turkerLabels in lstTurkerLabels:
            for x in [x for x in turkerLabels if x.event_type == 'destroy' and x.from_location != 'unk']:
                pred_loc = predictions[pid][x.participant][x.sid].from_location
                tp += int(pred_loc != 'null' and pred_loc != 'unk' and location_match(pred_loc, x.from_location))
                fp += int(pred_loc != 'null' and pred_loc != 'unk' and not location_match(pred_loc, x.from_location))
                fn += int(pred_loc == 'null' or pred_loc == 'unk')
    return tp, fp, tn, fn

#----------------------------------------------------------------------------------------------------------------

# Q7 Does participant X move during the process?
# total: sum(num_entity per paragraph)
def Q7(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        setParticipants = findAllParticipants(labels[pid])
        # find predictions
        be_moved = {}
        for participant in setParticipants:
            pred_move_steps = findMoveSteps(predictions[pid][participant].values())
            be_moved[participant] = (pred_move_steps != [])

        # print(be_moved)

        for turkerLabels in labels[pid]:
            lab_moved_participants = [x.participant for x in turkerLabels if x.event_type == 'move']
            for participant in setParticipants:
                tp += int(be_moved[participant] and (participant in lab_moved_participants))
                fp += int(be_moved[participant] and (participant not in lab_moved_participants))
                tn += int(not be_moved[participant] and (participant not in lab_moved_participants))
                fn += int(not be_moved[participant] and (participant in lab_moved_participants))

    return tp,fp,tn,fn

# Q8 Participant X moves during the process.  At which steps does it move?
# total: sum(moved entities per paragraph in labels)
def Q8(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        setParticipants = findAllParticipants(labels[pid])

        # find predictions
        pred_moved_steps = {}
        for participant in setParticipants:
            pred_moved_steps[participant] = findMoveSteps(predictions[pid][participant].values())
        num_steps = len(predictions[pid][participant].keys())

        for turkerLabels in labels[pid]:
            gold_moved_steps = {}
            for x in [x for x in turkerLabels if x.event_type == 'move']:
                if x.participant not in gold_moved_steps:
                    gold_moved_steps[x.participant] = []
                gold_moved_steps[x.participant].append(x.sid)

            for participant in gold_moved_steps:
                res = set_compare(pred_moved_steps[participant], gold_moved_steps[participant], num_steps)
                tp += res[0]
                fp += res[1]
                tn += res[2]
                fn += res[3]
    return tp,fp,tn,fn

def set_compare(pred_steps, gold_steps, num_steps):
    setPred = set(pred_steps)
    setGold = set(gold_steps)
    tp = len(setPred.intersection(setGold))
    fp = len(setPred - setGold)
    fn = len(setGold - setPred)
    tn = num_steps - tp - fp - fn
    return (tp, fp, tn, fn)

# Q9 Participant X moves at step Y, and its location before step Y is known. What is its location before step Y?
def Q9(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        for turkerLabels in labels[pid]:
            for x in turkerLabels:
                if x.event_type == 'move' and x.from_location != 'unk':
                    pred_loc = predictions[pid][x.participant][x.sid].from_location
                    tp += int(pred_loc != 'null' and pred_loc != 'unk' and location_match(pred_loc, x.from_location))
                    fp += int(pred_loc != 'null' and pred_loc != 'unk' and not location_match(pred_loc, x.from_location))
                    fn += int(pred_loc == 'null' or pred_loc == 'unk')
    return tp,fp,tn,fn

# Q10 Participant X moves at step Y, and its location after step Y is known. What is its location after step Y?
def Q10(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        for turkerLabels in labels[pid]:
            for x in turkerLabels:
                if x.event_type == 'move' and x.to_location != 'unk':
                    pred_loc = predictions[pid][x.participant][x.sid].to_location
                    tp += int(pred_loc != 'null' and pred_loc != 'unk' and location_match(pred_loc, x.to_location))
                    fp += int(pred_loc != 'null' and pred_loc != 'unk' and not location_match(pred_loc, x.to_location))
                    fn += int(pred_loc == 'null' or pred_loc == 'unk')
    return tp,fp,tn,fn

#----------------------------------------------------------------------------------------------------------------

def main():
    #if len(sys.argv) != 4:
        #sys.stderr.write("Usage: evalQA.py para-ids gold-labels system-predictions\n")
        #sys.exit(-1)
    paraIds_file = '../../tests/fixtures/eval/para_id.test.txt'#sys.argv[1]
    goldPred_file = '../../tests/fixtures/eval/gold_labels.test.tsv'#sys.argv[2]
    model_Pred_file = '../../data/naacl18/prolocal/output/prolocal.naacl_cr.data_run1.model_run2.test.tsv'#sys.argv[3]
    accuracy_score = {}
    precision_score, recall_score, F1_score = {}, {} ,{}
    accuracy_dict, precision_dict, recall_dict, F1_dict = {}, {} ,{}, {}


    set_Para_id = set([int(x) for x in open(paraIds_file).readlines()])
    location_labels = readGold(goldPred_file)
    full_labels = readLabels('../../tests/fixtures/eval/all-moves.full-grid.tsv', set_Para_id, location_labels)
    predictions = readPredictions(model_Pred_file)

    blHeader = True
    qid = 0
    for Q in [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10]:
        qid += 1
        tp, fp, tn, fn = Q(full_labels, predictions)
        header,results_str, results = metrics(tp,fp,tn,fn,qid)
        if blHeader:
            print("\t%s" % header)
            blHeader = False
        print("Q%d\t%s" % (qid, results_str))
        accuracy_score[qid] = results[5]
        precision_score[qid] = results[6]
        recall_score[qid] = results[7]
        F1_score[qid] = results[8]

    accuracy_dict = average(accuracy_score, accuracy_dict)
    precision_dict = average(precision_score, precision_dict)
    recall_dict = average(recall_score, recall_dict)
    F1_dict = average(F1_score, F1_dict)

    """
    cat1_score = (accuracy_score[1] + accuracy_score[4] + accuracy_score[7]) / 3
    cat2_score = (accuracy_score[2] + accuracy_score[5] + accuracy_score[8]) / 3
    cat3_score = (accuracy_score[3] + accuracy_score[6] + accuracy_score[9] + accuracy_score[10]) / 4

    macro_avg = (cat1_score + cat2_score + cat3_score) / 3
    num_cat1_q = 750
    num_cat2_q = 601
    num_cat3_q = 823
    micro_avg = ((cat1_score * num_cat1_q) + (cat2_score * num_cat2_q) + (cat3_score * num_cat3_q)) / \
                (num_cat1_q + num_cat2_q + num_cat3_q)
    """
    print("\n\nCategory\tAccuracy\tPrecision\tRecall\tF1")
    print("=========\t=====\t\t=====\t\t=====\t=====")
    print(f"Cat-1\t\t{round(accuracy_dict['cat1_score'],2)}\t\t{round(precision_dict['cat1_score'],2)}\t\t{round(recall_dict['cat1_score'],2)}\t{round(F1_dict['cat1_score'],2)}")
    print(f"Cat-2\t\t{round(accuracy_dict['cat2_score'],2)}\t\t{round(precision_dict['cat2_score'],2)}\t\t{round(recall_dict['cat2_score'],2)}\t{round(F1_dict['cat2_score'],2)}")
    print(f"Cat-3\t\t{round(accuracy_dict['cat3_score'],2)}\t\t{round(precision_dict['cat3_score'],2)}\t\t{round(recall_dict['cat3_score'],2)}\t{round(F1_dict['cat3_score'],2)}")
    print(f"macro-avg\t{round(accuracy_dict['macro_avg'],2)}\t\t{round(precision_dict['macro_avg'],2)}\t\t{round(recall_dict['macro_avg'],2)}\t{round(F1_dict['macro_avg'],2)}")
    print(f"micro-avg\t{round(accuracy_dict['micro_avg'],2)}\t\t{round(precision_dict['micro_avg'],2)}\t\t{round(recall_dict['micro_avg'],2)}\t{round(F1_dict['micro_avg'],2)}")

def metrics(tp, fp, tn, fn, qid):
    if (tp+fp > 0):
        prec = tp/(tp+fp)
    else:	 	
        prec = 0.0
    if (tp+fn > 0):
        rec = tp/(tp+fn)
    else:		
        rec = 0.0
    if (prec + rec) != 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0
    accuracy = (tp+tn) / (tp + fp + tn + fn)
    if qid == 8:
        accuracy = f1   # this is because Q8 can have multiple valid answers and F1 makes more sense here
    total = tp + fp + tn + fn

    header = '\t'.join(["Total", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"])
    results = [total, tp, fp, tn, fn, accuracy*100, prec*100, rec*100, f1*100]
    results_str = "%d\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f" % (total, tp, fp, tn, fn, accuracy*100, prec*100, rec*100, f1*100)
    return (header, results_str, results)

#----------------------------------------------------------------------------------------------------------------
def average(score: dict, metric: dict):
    cat1_score = (score[1] + score[4] + score[7]) / 3
    cat2_score = (score[2] + score[5] + score[8]) / 3
    cat3_score = (score[3] + score[6] + score[9] + score[10]) / 4

    metric['cat1_score'] = cat1_score
    metric['cat2_score'] = cat2_score
    metric['cat3_score'] = cat3_score

    metric['macro_avg'] = (cat1_score + cat2_score + cat3_score) / 3
    num_cat1_q = 750
    num_cat2_q = 601
    num_cat3_q = 823
    metric['micro_avg'] = ((cat1_score * num_cat1_q) + (cat2_score * num_cat2_q) + (cat3_score * num_cat3_q)) / \
                (num_cat1_q + num_cat2_q + num_cat3_q)
    return metric

if __name__ == "__main__":
    main()
