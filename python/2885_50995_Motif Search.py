import copy
import cPickle
import os

from Bio.Seq import Seq
import cdpybio as cpb
import matplotlib.pyplot as plt
import MOODS
import numpy as np
import pandas as pd
import pybedtools as pbt
import seaborn as sns
import weblogolib as logo

import cardipspy as cpy
import ciepy

from IPython.display import Image 

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')

outdir = os.path.join(ciepy.root, 'output',
                      'motif_search')
cpy.makedir(outdir)

private_outdir = os.path.join(ciepy.root, 'private_output',
                              'motif_search')
cpy.makedir(private_outdir)

# fn = os.path.join(ciepy.root, 'output', 'eqtl_processing', 'qvalues.tsv')
# qvalues = pd.read_table(fn, index_col=0)
# qvalues.columns = ['{}_gene'.format(x) for x in qvalues.columns]
# fn = os.path.join(ciepy.root, 'output', 'eqtl_processing', 'most_sig.tsv')
# most_sig = pd.read_table(fn, index_col=0)
# most_sig = most_sig.join(qvalues)
# sig = most_sig[most_sig.sig_gene]

fn = os.path.join(ciepy.root, 'output', 'functional_annotation_analysis',
                  'encode_stem_cell_chip_seq.tsv')
encode_chip_seq = pd.read_table(fn, index_col=0)

gene_info = pd.read_table(cpy.gencode_gene_info, index_col=0)

motif_info_full_fn = os.path.join(outdir, 'motif_info_full.tsv')
motif_info_rep_fn = os.path.join(outdir, 'motif_info_rep.tsv')
matrices_fn = os.path.join(outdir, 'matrices.pickle')

if not sum([os.path.exists(x) for x in [motif_info_full_fn, motif_info_rep_fn, matrices_fn]]) == 3:
    key = []
    tf = []
    cell_line = []
    source = []
    length = []
    with open(cpy.kheradpour_motifs) as f:
        lines = f.read()
    m = lines.split('>')[1:]
    m = [x.split('\n')[:-1] for x in m]
    matrices = {}
    for x in m:
        k = x[0].split()[0]
        key.append(k)
        if 'transfac' in x[0]:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(np.nan)
            source.append('transfac')
        elif 'jolma' in x[0]:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(np.nan)
            source.append('jolma')
        elif 'jaspar' in x[0]:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(np.nan)
            source.append('jaspar')
        elif 'bulyk' in x[0]:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(np.nan)
            source.append('bulyk')
        else:
            tf.append(x[0].split()[1].split('_')[0].upper())
            cell_line.append(x[0].split()[1].split('_')[1])
            source.append('encode')
        t = pd.DataFrame([y.split() for y in x[1:]],
                         columns=['base', 'A', 'C', 'G', 'T'])
        t.index = t.base
        t = t.drop('base', axis=1)
        for c in t.columns:
            t[c] = pd.to_numeric(t[c])
        matrices[k] = t
        length.append(t.shape[0])

    motif_info = pd.DataFrame({'tf': tf, 'cell_line': cell_line, 'source': source, 
                               'length': length}, index=key)
    motif_info.to_csv(motif_info_full_fn, sep='\t')
    
    with open(matrices_fn, 'w') as f:
        cPickle.dump(matrices, f)

    a = motif_info[motif_info.tf.apply(lambda x: x in encode_chip_seq.target.values)]
    b = a[a.cell_line == 'H1-hESC']
    b = b.drop_duplicates(subset='tf')
    a = a[a.cell_line != 'H1-hESC']
    a = a[a.tf.apply(lambda x: x not in b.tf.values)]
    a['so'] = a.source.replace({'jolma': 0, 'bulyk': 1, 'transfac': 2, 
                                'jaspar': 3, 'encode': 4})
    a = a.sort_values(by='so')
    a = a.drop_duplicates(subset='tf').drop('so', axis=1)
    motif_info = pd.concat([b, a])
    motif_info.to_csv(motif_info_rep_fn, sep='\t')

encode_chip_seq = encode_chip_seq[encode_chip_seq.target.apply(lambda x: x in motif_info.tf.values)]
encode_chip_seq = encode_chip_seq.drop_duplicates(subset='target')

sig = sig[sig.vtype == 'snp']

lines = (sig.chrom + '\t' + sig.start.astype(str) + 
         '\t' + sig.end.astype(str) + '\t' + sig.chrom +
         ':' + sig.end.astype(str))
lines = lines.drop_duplicates()
sig_bt = pbt.BedTool('\n'.join(lines + '\n'), from_string=True)
m = max([x.shape[0] for x in matrices.values()])
sig_bt = sig_bt.slop(l=m, r=m, g=pbt.genome_registry.hg19)
seqs = sig_bt.sequence(fi=cpy.hg19)
sig_seqs = [x.strip() for x in open(seqs.seqfn).readlines()]
sig_seqs = pd.Series(sig_seqs[1::2], index=[x[1:] for x in sig_seqs[0::2]])
sig_seqs = sig_seqs.apply(lambda x: x.upper())

snvs = sig[['chrom', 'start', 'end', 'loc', 'marker_id']]
snvs.index = snvs['loc'].values
snvs = snvs.drop_duplicates()
snvs['ref'] = snvs.marker_id.apply(lambda x: x.split('_')[1].split('/')[0])
snvs['alt'] = snvs.marker_id.apply(lambda x: x.split('_')[1].split('/')[1])

snvs['interval'] = ''
snvs['seq'] = ''
snvs['alt_seq'] = ''
for i in sig_seqs.index:
    chrom, start, end = cpb.general.parse_region(i)
    k = '{}:{}'.format(chrom, int(end) - m)
    snvs.ix[k, 'interval'] = i
    snvs.ix[k, 'seq'] = sig_seqs[i]
    ref, alt = snvs.ix[k, ['ref', 'alt']]
    assert sig_seqs[i][m] == ref
    snvs.ix[k, 'alt_seq'] = sig_seqs[i][0:m] + alt + sig_seqs[i][m + 1:]

lines = (sig.chrom + '\t' + sig.start.astype(str) + 
         '\t' + sig.end.astype(str) + '\t' + sig.chrom +
         ':' + sig.end.astype(str))
lines = lines.drop_duplicates()
sig_bt = pbt.BedTool('\n'.join(lines + '\n'), from_string=True)
sig_bt = sig_bt.sort()

snvs_tf = pd.DataFrame(False, index=snvs.index, columns=encode_chip_seq.target)
for i in encode_chip_seq.index:
    c = encode_chip_seq.ix[i, 'target']
    snvs_tf[c] = False
    bt = pbt.BedTool(cpb.general.read_gzipped_text_url(encode_chip_seq.ix[i, 'narrowPeak_url']), 
                     from_string=True)
    bt = bt.sort()
    res = sig_bt.intersect(bt, sorted=True, wo=True)
    for r in res:
        snvs_tf.ix['{}:{}'.format(r.chrom, r.end), c] = True

snv_motifs = {}
for i in snvs_tf[snvs_tf.sum(axis=1) > 0].index:
    se = snvs_tf.ix[i]
    se = se[se]
    keys = motif_info[motif_info.tf.apply(lambda x: x in se.index)].index
    ms = [matrices[x].T.values.tolist() for x in keys]
    # seq_res is a dict whose keys are motif names and whose values are lists 
    # of the hits of that motif. Each hit is a tuple of (pos, score). 
    seq_res = MOODS.search(snvs.ix[i, 'seq'], ms, 0.001, both_strands=True, 
                           bg=[0.25, 0.25, 0.25, 0.25])
    seq_mres = dict(zip(keys, seq_res))
    alt_seq_res = MOODS.search(snvs.ix[i, 'alt_seq'], ms, 0.001, both_strands=True, 
                               bg=[0.25, 0.25, 0.25, 0.25])
    alt_seq_mres = dict(zip(keys, alt_seq_res))
    sp = len(snvs.ix[i, 'seq']) / 2
    if seq_mres != alt_seq_mres:
        for k in seq_mres.keys():
            # Remove motifs where all the hits have the same score.
            if seq_mres[k] == alt_seq_mres[k]:
                seq_mres.pop(k)
                alt_seq_mres.pop(k)
            else:
                # Remove individual hits that have the same score for both sequences.
                shared = set(seq_mres[k]) & set(alt_seq_mres)
                seq_mres[k] = [x for x in seq_mres[k] if x not in shared]
                alt_seq_mres[k] = [x for x in alt_seq_mres[k] if x not in shared]
                a = seq_mres[k]
                to_remove = []
                for v in a:
                    start = v[0]
                    if start < 0:
                        start = start + len(snvs.ix[i, 'seq'])
                    if not start <= sp < start + motif_info.ix[k, 'length']:
                        to_remove.append(v)
                for v in to_remove:
                    a.remove(v)
                seq_mres[k] = a
                a = alt_seq_mres[k]
                to_remove = []
                for v in a:
                    start = v[0]
                    if start < 0:
                        start = start + len(snvs.ix[i, 'seq'])
                    if not start <= sp < start + motif_info.ix[k, 'length']:
                        to_remove.append(v)
                for v in to_remove:
                    a.remove(v)
                alt_seq_mres[k] = a
        snv_motifs[i] = [seq_mres, alt_seq_mres]

def plot_tf_disruption(m, ref, alt, fn, title=None):
    """m is the PWM, ref is the ref sequence, alt is the alt sequence"""
    k = 'SIX5_disc2'
    alphabet = logo.corebio.seq.unambiguous_dna_alphabet
    prior = [0.25, 0.25, 0.25, 0.25]
    counts = m.values
    assert counts.shape[1] == 4
    assert len(ref) == len(alt) == counts.shape[0]
    ref_counts = []
    for t in ref:
        ref_counts.append([int(t.upper() == 'A'), int(t.upper() == 'C'),
                           int(t.upper() == 'G'), int(t.upper() == 'T')])
    alt_counts = []
    for t in alt:
        alt_counts.append([int(t.upper() == 'A'), int(t.upper() == 'C'),
                           int(t.upper() == 'G'), int(t.upper() == 'T')])
    counts = np.concatenate([counts, ref_counts, alt_counts])
    data = logo.LogoData.from_counts(alphabet, counts, prior=None)
    fout = open(fn, 'w')
    options = logo.LogoOptions()
    options.fineprint = ''
    if title:
        options.logo_title = title
    else:
        options.logo_title = ''
    options.stacks_per_line = m.shape[0]
    options.show_xaxis = False
    options.show_yaxis = False
    options.color_scheme = logo.ColorScheme([logo.ColorGroup("G", "orange"), 
                                             logo.ColorGroup("C", "blue"),
                                             logo.ColorGroup("A", "green"),
                                             logo.ColorGroup("T", "red")])
    logo_format = logo.LogoFormat(data, options)
    fout.write(logo.png_print_formatter(data, logo_format))
    #fout.write(logo.pdf_formatter(data, logo_format))
    fout.close()
    Image(filename=fn)

cpy.makedir(os.path.join(outdir, 'tf_plots'))
for snv in snv_motifs.keys():
    seq_mres, alt_seq_mres = snv_motifs[snv]
    for k in seq_mres.keys():
        pwm = matrices[k]
        a = seq_mres[k]
        b = alt_seq_mres[k]
        starts = set([x[0] for x in a]) | set([x[0] for x in b])
        for start in starts:
            ref_seq = snvs.ix[snv, 'seq'][start: start + motif_info.ix[k, 'length']]
            alt_seq = snvs.ix[snv, 'alt_seq'][start: start + motif_info.ix[k, 'length']]
            if start < 0:
                ref_seq = str(Seq(ref_seq).reverse_complement())
                alt_seq = str(Seq(alt_seq).reverse_complement())
            fn = os.path.join(outdir, 'tf_plots', '{}_{}_{}.png'.format(
                snv.replace(':', '_'), k, str(start).replace('-', 'neg')))
            plot_tf_disruption(pwm, ref_seq, alt_seq, fn)

