import re, os

def read_word_transcript_file(transfile):
    """read ms98 transcript file
    
    Arguments:
        transfile {string} -- file path
    
    Returns:
        dlg -- list of utterances, each is list of dict(start, end, word)    
    """

    ret = []
    with open(transfile) as f:
        lines = f.read().split('\n')
        lines = [list(
            filter(lambda it: it != '', line.split(' '))) if '\t' not in line 
                 else line.split('\t') 
                 for line in lines]
        lines = [dict(
            start=int(float(line[1]) * 100 + 0.05),
            end=int(float(line[2]) * 100 + 0.05),
            id=int(line[0].split('-')[-1]),
            word=line[3].lower(),
            caller=os.path.basename(transfile)[6]
        ) for line in lines if len(line) == 4]
            
        cur, id = None, None
        i = 0
        ignored_ls = ['', '---', '+++', '<e_aside>', '<b_aside>', '-h', '-s']
        splitted_ls = ['[silence]', '[noise]', '[laughter]', '[vocalized-noise]']
            
        while i < len(lines):
            line = lines[i]
            word = line['word']
                
            # [laughter-word] -> word
            re_laughter = re.match(r'\[laughter\-(.*)\]', word)
            if re_laughter is not None: word = re_laughter[1]
                
            # [word1/word2] -> word2
            if '/' in word:
                word = word[word.index('/') + 1:-1]
                
            if word not in splitted_ls:
                for c in ['[', ']', '-']: word = word.replace(c, '')
                    
            if word in ignored_ls: pass
            elif cur is not None \
                and line['id'] == id \
                and (word not in splitted_ls or len(cur['words']) < 3) \
                and word not in splitted_ls:
                    cur['words'].append(dict(start=line['start'], end=line['end'], word=word))
            else:  # new utterance
                if cur is not None and len(cur['words']) > 0:
                    ret.append(cur)
                if line['word'] not in splitted_ls: cur = dict(words=[dict(start=line['start'], end=line['end'], word=line['word'])], caller=line['caller'])
                else: cur = dict(words=[], caller=line['caller'])
                id = line['id']
            i += 1
                
        if cur is not None and len(cur['words']) > 0:
            ret.append(cur)
        return ret