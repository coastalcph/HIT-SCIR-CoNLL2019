import os,sys
#run:
# python collect_stats_eud.py data_dir 
# where data dir contains the UD directory 

eud_dir=sys.argv[1]
langs = os.listdir(eud_dir)
stats={}
bins = [(1,5),(5,10),(10,20), (20,50),(50,100),(100,500),(500,1000)]
bins_dict = {}
for abin in bins:
    bins_dict[str(abin)] = list(range(abin[0],abin[1]))
bins_dict['1000+'] = []

for lang in langs:
    if lang.endswith('PUD') or lang.endswith('FQB'):
        continue
    iso_cmd = f'ls {eud_dir}{lang}/*train.conllu'
    iso = [line for line in os.popen(iso_cmd)][0].split("/")[6].split("-")[0]
    stats[iso] = {'n_head':{},'labels':{}, 'label_freq':{}, 'binned_freq':{}}
    for abin in bins_dict:
        stats[iso]['binned_freq'][abin] = 0
    trainfile = f'{eud_dir}{lang}/{iso}-ud-train.conllu'
    with open(trainfile,'r') as tf:
        for line in tf:
            if line.startswith("#"):
                continue
            elif line == '\n':
                continue
            else:
                eud = line.split('\t')[8]
                all_heads = eud.split('|')
                n_heads = len(all_heads)
                if n_heads not in stats[iso]['n_head']:
                    stats[iso]['n_head'][n_heads] = 0
                stats[iso]['n_head'][n_heads] += 1
                for head in all_heads:
                    label = ''.join(head.partition(':')[2:])
                    if label not in stats[iso]['labels']:
                        stats[iso]['labels'][label] = 0
                    stats[iso]['labels'][label] +=1
    for label in stats[iso]['labels']:
        freq = stats[iso]['labels'][label]
        if freq not in stats[iso]['label_freq']:
            stats[iso]['label_freq'][freq] = 0
        stats[iso]['label_freq'][freq] += 1
        found_bin = False
        for abin in bins_dict:
            if freq in bins_dict[abin]:
                stats[iso]['binned_freq'][abin] += 1
                found_bin = True
        if not found_bin:
            stats[iso]['binned_freq']['1000+'] += 1


for iso in stats:
    outfile = f'stats/{iso}.csv'
    with open(outfile,'w') as out:
        num_labels = len(stats[iso]['labels'])
        out.write('labels \n')
        line1 = f'number of labels\t{num_labels}\n'
        out.write(line1)
        out.write('binned label frequencies \n')
        for abin, freq in stats[iso]['binned_freq'].items():
            line = f'{abin}\t{freq}\n'
            out.write(line)

        max_heads = max(stats[iso]['n_head'])
        out.write('\n heads \n')
        line2 = f'max heads\t{max_heads}\n'
        out.write(line2)
        out.write('freq of head numbers \n')
        for n_head, freq in sorted(stats[iso]['n_head'].items()):
            line = f'{n_head}\t{freq}\n'
            out.write(line)

