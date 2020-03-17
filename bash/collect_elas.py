import sys, os, json

chkpoints = sys.argv[1]
results = open('results.csv','w')
line1 ='iso;ELAS;status\n'
results.write(line1)

for lang in os.listdir(chkpoints):
    last_epoch_file = os.path.join(chkpoints, lang, 'metrics_epoch_49.json')
    done=False
    started=True
    if os.path.exists(last_epoch_file):
        done=True
    else:
        files = os.listdir(os.path.join(chkpoints,lang))
        if 'metrics_epoch_0.json' in files:
            last_epoch = str(sorted([int(f.split('metrics_epoch_')[1].strip('.json'))
                for f in files if f.startswith('metrics_epoch')])[-1])
            last_epoch_file = os.path.join(chkpoints, lang,
                    f'metrics_epoch_{last_epoch}.json')
        else:
            started=False

    if started:
        with open(last_epoch_file) as f:
            info = json.load(f)
        elas = info["best_validation_ELAS"]
        status = 'Done' if done else last_epoch
        results.write(f'{lang};{elas};{status}\n')
    #else:
    #    results.write(f'{lang};N/A;not started\n')

results.close()
