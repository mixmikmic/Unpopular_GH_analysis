import numpy as np
import scipy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')
import h5py
import allel; print('scikit-allel', allel.__version__)
import collections

callset_fn = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/hdf5/Pf_60.h5'
callset = h5py.File(callset_fn, mode='r')
callset

variants = allel.VariantChunkedTable(callset['variants'], 
                                     names=['CHROM', 'POS', 'FILTER_PASS', 'CDS', 'MULTIALLELIC', 'VARIANT_TYPE', 'AC', 'AN', 'AF', 'VQSLOD'],
                                     index=['CHROM', 'POS'])
variants

np.unique(variants['CDS'][:], return_counts=True)

pca_selection = (
    (variants['FILTER_PASS'][:]) &
    (variants['CDS'][:]) &
    (variants['MULTIALLELIC'][:] == b'BI') &
    (variants['VARIANT_TYPE'][:] == b'SNP') &
    (variants['AF'][:,0] >= 0.05) &
    (variants['AF'][:,0] <= 0.95) &
    (variants['VQSLOD'][:] > 6.0)
)

np.unique(pca_selection, return_counts=True)

variants_pca = variants.compress(pca_selection)
variants_pca

calldata = callset['calldata']
calldata

genotypes = allel.GenotypeChunkedArray(calldata['genotype'])
genotypes

get_ipython().run_cell_magic('time', '', 'genotypes_pca = genotypes.subset(pca_selection)')

genotypes_pca

get_ipython().run_cell_magic('time', '', 'n_variants = len(variants_pca)\npc_missing = genotypes_pca.count_missing(axis=0)[:] * 100 / n_variants')

fig, ax = plt.subplots(figsize=(12, 4))
_ = ax.hist(pc_missing[pc_missing<10], bins=100)

good_samples = (pc_missing < 2.0)
genotypes_pca_good = genotypes_pca.take(np.where(good_samples)[0], axis=1)
genotypes_pca_good

gn = genotypes_pca_good.to_n_alt()[:]
gn

get_ipython().run_cell_magic('time', '', 'coords, model = allel.stats.pca(gn)')

samples_fn = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt'
samples = pandas.DataFrame.from_csv(samples_fn, sep='\t')
samples.head()

continents = collections.OrderedDict()
continents['1001-PF-ML-DJIMDE']               = '1_WA'
continents['1004-PF-BF-OUEDRAOGO']            = '1_WA'
continents['1006-PF-GM-CONWAY']               = '1_WA'
continents['1007-PF-TZ-DUFFY']                = '2_EA'
continents['1008-PF-SEA-RINGWALD']            = '4_SEA'
continents['1009-PF-KH-PLOWE']                = '4_SEA'
continents['1010-PF-TH-ANDERSON']             = '4_SEA'
continents['1011-PF-KH-SU']                   = '4_SEA'
continents['1012-PF-KH-WHITE']                = '4_SEA'
continents['1013-PF-PEGB-BRANCH']             = '6_SA'
continents['1014-PF-SSA-SUTHERLAND']          = '3_AF'
continents['1015-PF-KE-NZILA']                = '2_EA'
continents['1016-PF-TH-NOSTEN']               = '4_SEA'
continents['1017-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1018-PF-GB-NEWBOLD']              = '9_Lab'
continents['1020-PF-VN-BONI']                 = '4_SEA'
continents['1021-PF-PG-MUELLER']              = '5_OC'
continents['1022-PF-MW-OCHOLLA']              = '2_EA'
continents['1023-PF-CO-ECHEVERRI-GARCIA']     = '6_SA'
continents['1024-PF-UG-BOUSEMA']              = '2_EA'
continents['1025-PF-KH-PLOWE']                = '4_SEA'
continents['1026-PF-GN-CONWAY']               = '1_WA'
continents['1027-PF-KE-BULL']                 = '2_EA'
continents['1031-PF-SEA-PLOWE']               = '4_SEA'
continents['1044-PF-KH-FAIRHURST']            = '4_SEA'
# continents['1052-PF-TRAC-WHITE']              = '4_SEA'
continents['1052-PF-TRAC-WHITE']              = '0_MI'
continents['1062-PF-PG-BARRY']                = '5_OC'
continents['1083-PF-GH-CONWAY']               = '1_WA'
continents['1093-PF-CM-APINJOH']              = '1_WA'
continents['1094-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1095-PF-TZ-ISHENGOMA']            = '2_EA'
continents['1096-PF-GH-GHANSAH']              = '1_WA'
continents['1097-PF-ML-MAIGA']                = '1_WA'
continents['1098-PF-ET-GOLASSA']              = '2_EA'
continents['1100-PF-CI-YAVO']                 = '1_WA'
continents['1101-PF-CD-ONYAMBOKO']            = '1_WA'
continents['1102-PF-MG-RANDRIANARIVELOJOSIA'] = '2_EA'
continents['1103-PF-PDN-GMSN-NGWA']           = '1_WA'
continents['1107-PF-KEN-KAMAU']               = '2_EA'
continents['1125-PF-TH-NOSTEN']               = '4_SEA'
continents['1127-PF-ML-SOULEYMANE']           = '1_WA'
continents['1131-PF-BJ-BERTIN']               = '1_WA'
continents['1133-PF-LAB-MERRICK']             = '9_Lab'
continents['1134-PF-ML-CONWAY']               = '1_WA'
continents['1135-PF-SN-CONWAY']               = '1_WA'
continents['1136-PF-GM-NGWA']                 = '1_WA'
continents['1137-PF-GM-DALESSANDRO']          = '1_WA'
continents['1138-PF-CD-FANELLO']              = '1_WA'
continents['1141-PF-GM-CLAESSENS']            = '1_WA'
continents['1145-PF-PE-GAMBOA']               = '6_SA'
continents['1147-PF-MR-CONWAY']               = '1_WA'
continents['1151-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1152-PF-DBS-GH-AMENGA-ETEGO']     = '1_WA'
continents['1155-PF-ID-PRICE']                = '5_OC'

samples['continent'] = pandas.Series([continents[x] for x in samples.study], index=samples.index)

samples['is_SA'] = pandas.Series(samples['continent'] == '6_SA', index=samples.index)

samples.continent.value_counts()

samples.is_SA.value_counts()

samples_subset = samples[good_samples]
samples_subset.reset_index(drop=True, inplace=True)
samples_subset.head()



def plot_pca_coords(coords, model, pc1, pc2, ax, variable='continent', exclude_values=['9_Lab', '3_AF', '0_MI']):
    x = coords[:, pc1]
    y = coords[:, pc2]
    for value in samples_subset[variable].unique():
        if not value in exclude_values:
            flt = (samples_subset[variable] == value).values
            ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=value, markersize=6)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))

fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax)
ax.legend(loc='upper left');

fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax, variable='study')
ax.legend(loc='upper left');

fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax, variable='is_SA')
ax.legend(loc='upper left');

fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 3, 4, ax, variable='continent')
ax.legend(loc='upper left');

fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 3, 4, ax, variable='is_SA')
ax.legend(loc='upper left');

fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 5, 6, ax, variable='is_SA')
ax.legend(loc='upper left');

fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 7, 8, ax, variable='is_SA')
ax.legend(loc='upper left');

fig, ax = plt.subplots(figsize=(15, 15))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 5, ax)
ax.legend(loc='upper left');





get_ipython().run_cell_magic('time', '', 'genotypes_subset = genotypes.subset(variant_selection, sample_selection)')









def plot_windowed_variant_density(pos, window_size, title=None):
    
    # setup windows 
    bins = np.arange(0, pos.max(), window_size)
    
    # use window midpoints as x coordinate
    x = (bins[1:] + bins[:-1])/2
    
    # compute variant density in each window
    h, _ = np.histogram(pos, bins=bins)
    y = h / window_size
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y)
    ax.set_xlabel('Chromosome position (bp)')
    ax.set_ylabel('Variant density (bp$^{-1}$)')
    if title:
        ax.set_title(title)

for current_chrom in chroms:
    plot_windowed_variant_density(
        pos[chrom==current_chrom], window_size=1000, title='Raw variant density %s' % current_chrom.decode('ascii')
    )

dp = variants['DP'][:]
dp

def plot_variant_hist(f, bins=30):
    x = variants[f][:]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.despine(ax=ax, offset=10)
    ax.hist(x, bins=bins)
    ax.set_xlabel(f)
    ax.set_ylabel('No. variants')
    ax.set_title('Variant %s distribution' % f)

plot_variant_hist('DP', bins=50)

plot_variant_hist('MQ')

plot_variant_hist('QD')

plot_variant_hist('num_alleles', bins=np.arange(1.5, 8.5, 1))
plt.gca().set_xticks([2, 3, 4, 5, 6, 7]);

def plot_variant_hist_2d(f1, f2, downsample):
    x = variants[f1][:][::downsample]
    y = variants[f2][:][::downsample]
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.despine(ax=ax, offset=10)
    ax.hexbin(x, y, gridsize=40)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title('Variant %s versus %s joint distribution' % (f1, f2))

plot_variant_hist_2d('QD', 'MQ', downsample=10)

mutations = np.char.add(variants['REF'].subset(variants['is_snp']), variants['ALT'].subset(variants['is_snp'])[:, 0])
mutations

def locate_transitions(x):
    x = np.asarray(x)
    return (x == b'AG') | (x == b'GA') | (x == b'CT') | (x == b'TC')

is_ti = locate_transitions(mutations)
is_ti

def ti_tv(x):
    if len(x) == 0:
        return np.nan
    is_ti = locate_transitions(x)
    n_ti = np.count_nonzero(is_ti)
    n_tv = np.count_nonzero(~is_ti)
    if n_tv > 0:
        return n_ti / n_tv
    else:
        return np.nan

ti_tv(mutations)

def plot_ti_tv(f, downsample, bins):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.despine(ax=ax, offset=10)
    x = variants[f].subset(variants['is_snp'])[:][::downsample]
    
    # plot a histogram
    ax.hist(x, bins=bins)
    ax.set_xlabel(f)
    ax.set_ylabel('No. variants')

    # plot Ti/Tv
    ax = ax.twinx()
    sns.despine(ax=ax, bottom=True, left=True, right=False, offset=10)
    values = mutations[::downsample]
    with np.errstate(over='ignore'):
        # binned_statistic generates an annoying overflow warning which we can ignore
        y1, _, _ = scipy.stats.binned_statistic(x, values, statistic=ti_tv, bins=bins)
    bx = (bins[1:] + bins[:-1]) / 2
    ax.plot(bx, y1, color='k')
    ax.set_ylabel('Ti/Tv')
    ax.set_ylim(0.6, 1.3)

    ax.set_title('Variant %s and Ti/Tv' % f)

plot_ti_tv('QD', downsample=5, bins=np.arange(0, 40, 1))

plot_ti_tv('MQ', downsample=5, bins=np.arange(0, 60, 1))

plot_ti_tv('DP', downsample=5, bins=np.linspace(0, 50000, 50))

def plot_joint_ti_tv(f1, f2, downsample, gridsize=20, mincnt=20, vmin=0.6, vmax=1.4, extent=None):
    fig, ax = plt.subplots()
    sns.despine(ax=ax, offset=10)
    x = variants[f1].subset(variants['is_snp'])[:][::downsample]
    y = variants[f2].subset(variants['is_snp'])[:][::downsample]
    C = mutations[::downsample]
    im = ax.hexbin(x, y, C=C, reduce_C_function=ti_tv, mincnt=mincnt, extent=extent,
                   gridsize=gridsize, cmap='jet', vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title('Variant %s versus %s and Ti/Tv' % (f1, f2))

plot_joint_ti_tv('QD', 'MQ', downsample=5, mincnt=400, extent=(0, 40, 0, 80))

plot_joint_ti_tv('QD', 'DP', downsample=5, mincnt=400, extent=(0, 40, 0, 8e+5))
# plot_joint_ti_tv('QD', 'DP', downsample=5, mincnt=400)

plot_joint_ti_tv('MQ', 'DP', downsample=5, mincnt=400, extent=(0, 80, 0, 8e+5))

variants

filter_expression = '(QD > 5) & (MQ > 40) & (DP > 3e+5) & (DP < 8e+5)'

variant_selection = variants.eval(filter_expression)[:]
variant_selection

np.count_nonzero(variant_selection)

np.count_nonzero(~variant_selection)

variants_pass = variants.compress(variant_selection)
variants_pass

import gc
gc.collect()

for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, eval("%s.nbytes" % object_name))

del(variants)
del(mutations)
del(pos)
del(dp)
del(chrom)
del(_3)
del(_31)
del(_18)
del(_7)
del(_10)
del(_5)
gc.collect()

for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, eval("%s.nbytes" % object_name))

calldata = callset['calldata']
calldata

list(calldata.keys())

genotypes = allel.GenotypeChunkedArray(calldata['genotype'])
genotypes

samples_fn = '/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt'
samples = pandas.DataFrame.from_csv(samples_fn, sep='\t')
samples.head()

samples.study.value_counts()

continents = collections.OrderedDict()
continents['1001-PF-ML-DJIMDE']               = '1_WA'
continents['1004-PF-BF-OUEDRAOGO']            = '1_WA'
continents['1006-PF-GM-CONWAY']               = '1_WA'
continents['1007-PF-TZ-DUFFY']                = '2_EA'
continents['1008-PF-SEA-RINGWALD']            = '4_SEA'
continents['1009-PF-KH-PLOWE']                = '4_SEA'
continents['1010-PF-TH-ANDERSON']             = '4_SEA'
continents['1011-PF-KH-SU']                   = '4_SEA'
continents['1012-PF-KH-WHITE']                = '4_SEA'
continents['1013-PF-PEGB-BRANCH']             = '6_SA'
continents['1014-PF-SSA-SUTHERLAND']          = '3_AF'
continents['1015-PF-KE-NZILA']                = '2_EA'
continents['1016-PF-TH-NOSTEN']               = '4_SEA'
continents['1017-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1018-PF-GB-NEWBOLD']              = '9_Lab'
continents['1020-PF-VN-BONI']                 = '4_SEA'
continents['1021-PF-PG-MUELLER']              = '5_OC'
continents['1022-PF-MW-OCHOLLA']              = '2_EA'
continents['1023-PF-CO-ECHEVERRI-GARCIA']     = '6_SA'
continents['1024-PF-UG-BOUSEMA']              = '2_EA'
continents['1025-PF-KH-PLOWE']                = '4_SEA'
continents['1026-PF-GN-CONWAY']               = '1_WA'
continents['1027-PF-KE-BULL']                 = '2_EA'
continents['1031-PF-SEA-PLOWE']               = '4_SEA'
continents['1044-PF-KH-FAIRHURST']            = '4_SEA'
continents['1052-PF-TRAC-WHITE']              = '4_SEA'
continents['1062-PF-PG-BARRY']                = '5_OC'
continents['1083-PF-GH-CONWAY']               = '1_WA'
continents['1093-PF-CM-APINJOH']              = '1_WA'
continents['1094-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1095-PF-TZ-ISHENGOMA']            = '2_EA'
continents['1096-PF-GH-GHANSAH']              = '1_WA'
continents['1097-PF-ML-MAIGA']                = '1_WA'
continents['1098-PF-ET-GOLASSA']              = '2_EA'
continents['1100-PF-CI-YAVO']                 = '1_WA'
continents['1101-PF-CD-ONYAMBOKO']            = '1_WA'
continents['1102-PF-MG-RANDRIANARIVELOJOSIA'] = '2_EA'
continents['1103-PF-PDN-GMSN-NGWA']           = '1_WA'
continents['1107-PF-KEN-KAMAU']               = '2_EA'
continents['1125-PF-TH-NOSTEN']               = '4_SEA'
continents['1127-PF-ML-SOULEYMANE']           = '1_WA'
continents['1131-PF-BJ-BERTIN']               = '1_WA'
continents['1133-PF-LAB-MERRICK']             = '9_Lab'
continents['1134-PF-ML-CONWAY']               = '1_WA'
continents['1135-PF-SN-CONWAY']               = '1_WA'
continents['1136-PF-GM-NGWA']                 = '1_WA'
continents['1137-PF-GM-DALESSANDRO']          = '1_WA'
continents['1138-PF-CD-FANELLO']              = '1_WA'
continents['1141-PF-GM-CLAESSENS']            = '1_WA'
continents['1145-PF-PE-GAMBOA']               = '6_SA'
continents['1147-PF-MR-CONWAY']               = '1_WA'
continents['1151-PF-GH-AMENGA-ETEGO']         = '1_WA'
continents['1152-PF-DBS-GH-AMENGA-ETEGO']     = '1_WA'
continents['1155-PF-ID-PRICE']                = '5_OC'

samples['continent'] = pandas.Series([continents[x] for x in samples.study], index=samples.index)

samples.continent.value_counts()

samples

sample_selection = samples.continent.isin({'5_OC', '6_SA'}).values
sample_selection[:5]

sample_selection = samples.study.isin(
    {'1010-PF-TH-ANDERSON', '1013-PF-PEGB-BRANCH', '1023-PF-CO-ECHEVERRI-GARCIA', '1145-PF-PE-GAMBOA', '1134-PF-ML-CONWAY', '1025-PF-KH-PLOWE'}
).values
sample_selection[:5]

samples_subset = samples[sample_selection]
samples_subset.reset_index(drop=True, inplace=True)
samples_subset.head()

samples_subset.continent.value_counts()

get_ipython().run_cell_magic('time', '', 'genotypes_subset = genotypes.subset(variant_selection, sample_selection)')

genotypes_subset

get_ipython().run_cell_magic('time', '', 'n_variants = len(variants_pass)\npc_missing = genotypes_subset.count_missing(axis=0)[:] * 100 / n_variants\npc_het = genotypes_subset.count_het(axis=0)[:] * 100 / n_variants')

def plot_genotype_frequency(pc, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    left = np.arange(len(pc))
    palette = sns.color_palette()
    pop2color = {'1_WA': palette[0], '6_SA': palette[1], '4_SEA': palette[2]}
    colors = [pop2color[p] for p in samples_subset.continent]
    ax.bar(left, pc, color=colors)
    ax.set_xlim(0, len(pc))
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Percent calls')
    ax.set_title(title)
    handles = [mpl.patches.Patch(color=palette[0]),
               mpl.patches.Patch(color=palette[1])]
    ax.legend(handles=handles, labels=['1_WA', '6_SA', '4_SEA'], title='Population',
              bbox_to_anchor=(1, 1), loc='upper left')

plot_genotype_frequency(pc_missing, 'Missing')

np.argsort(pc_missing)[-1]

g_strange = genotypes_subset.take([30, 62, 63], axis=1)
g_strange

is_missing = g_strange.is_missing()[:]
is_missing

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_pass['CHROM'][:]==current_chrom
    pos = variants_pass['POS'][:][this_chrom_variant]
    window_size = 10000
    y1, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 0], statistic=np.count_nonzero, size=window_size)
    y2, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 1], statistic=np.count_nonzero, size=window_size)
    y3, windows, _ = allel.stats.windowed_statistic(pos, is_missing[this_chrom_variant, 2], statistic=np.count_nonzero, size=window_size)
    x = windows.mean(axis=1)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y1 * 100 / window_size, lw=1)
    ax.plot(x, y2 * 100 / window_size, lw=1)
    ax.plot(x, y3 * 100 / window_size, lw=1)
    ax.set_title(current_chrom.decode('ascii'))
    ax.set_xlabel('Position (bp)')
    ax.set_ylabel('Percent calls');

plot_genotype_frequency(pc_het, 'Heterozygous')

subpops = {
    'all': list(range(len(samples_subset))),
    'WA': samples_subset[samples_subset.continent == '1_WA'].index.tolist(),
    'SA': samples_subset[samples_subset.continent == '6_SA'].index.tolist(),
    'SEA': samples_subset[samples_subset.continent == '4_SEA'].index.tolist(),
}
subpops['WA'][:5]

get_ipython().run_cell_magic('time', '', 'ac_subpops = genotypes_subset.count_alleles_subpops(subpops, max_allele=6)')

ac_subpops

ac_subpops['SA'][:5]

for pop in 'all', 'WA', 'SA', 'SEA':
    print(pop, ac_subpops[pop].count_segregating())

is_seg = ac_subpops['all'].is_segregating()[:]
is_seg

genotypes_seg = genotypes_subset.compress(is_seg, axis=0)
genotypes_seg

variants_seg = variants_pass.compress(is_seg)
variants_seg

ac_seg = ac_subpops.compress(is_seg)
ac_seg

for object_name in locals().keys():
    if eval("isinstance(%s, np.ndarray)" % object_name) or eval("isinstance(%s, allel.abc.ArrayWrapper)" % object_name):
        print(object_name, int(eval("%s.nbytes" % object_name) / 1e+6))

jsfs = allel.stats.joint_sfs(ac_seg['WA'][:, 1], ac_seg['SA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, WA')
ax.set_ylabel('Alternate allele count, SA');

jsfs = allel.stats.joint_sfs(ac_seg['WA'][:, 1], ac_seg['SEA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, WA')
ax.set_ylabel('Alternate allele count, SEA');
jsfs

jsfs = allel.stats.joint_sfs(ac_seg['SA'][:, 1], ac_seg['SEA'][:, 1])
fig, ax = plt.subplots(figsize=(6, 6))
allel.stats.plot_joint_sfs(jsfs, ax=ax)
ax.set_xlabel('Alternate allele count, SA')
ax.set_ylabel('Alternate allele count, SEA');

jsfs

fst, fst_se, _, _ = allel.stats.blockwise_hudson_fst(ac_seg['WA'], ac_seg['SA'], blen=100000)
print("Hudson's Fst: %.3f +/- %.3f" % (fst, fst_se))

def plot_fst(ac1, ac2, pos, blen=2000, current_chrom=b'Pf3D7_01_v3'):
    
    fst, se, vb, _ = allel.stats.blockwise_hudson_fst(ac1, ac2, blen=blen)
    
    # use the per-block average Fst as the Y coordinate
    y = vb
    
    # use the block centres as the X coordinate
    x = allel.stats.moving_statistic(pos, statistic=lambda v: (v[0] + v[-1]) / 2, size=blen)
    
    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y, 'k-', lw=.5)
    ax.set_ylabel('$F_{ST}$')
    ax.set_xlabel('Chromosome %s position (bp)' % current_chrom.decode('ascii'))
    ax.set_xlim(0, pos.max())

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_seg['CHROM'][:]==current_chrom
    plot_fst(
        ac_seg['WA'].subset(this_chrom_variant),
        ac_seg['SA'].subset(this_chrom_variant),
        variants_seg['POS'][:][this_chrom_variant],
        100,
        current_chrom
    )

is_biallelic_01 = ac_seg['all'].is_biallelic_01()[:]
ac1 = ac_seg['WA'].compress(is_biallelic_01, axis=0)[:, :2]
ac2 = ac_seg['SA'].compress(is_biallelic_01, axis=0)[:, :2]
ac3 = ac_seg['SEA'].compress(is_biallelic_01, axis=0)[:, :2]
ac1

fig, ax = plt.subplots(figsize=(8, 5))
sns.despine(ax=ax, offset=10)
sfs1 = allel.stats.sfs_folded_scaled(ac1)
allel.stats.plot_sfs_folded_scaled(sfs1, ax=ax, label='WA', n=ac1.sum(axis=1).max())
sfs2 = allel.stats.sfs_folded_scaled(ac2)
allel.stats.plot_sfs_folded_scaled(sfs2, ax=ax, label='SA', n=ac2.sum(axis=1).max())
sfs3 = allel.stats.sfs_folded_scaled(ac3)
allel.stats.plot_sfs_folded_scaled(sfs3, ax=ax, label='SEA', n=ac3.sum(axis=1).max())
ax.legend()
ax.set_title('Scaled folded site frequency spectra')
# workaround bug in scikit-allel re axis naming
ax.set_xlabel('minor allele frequency');

for current_chrom in chroms[:-2]:
    this_chrom_variant = variants_seg['CHROM'][:]==current_chrom
    # compute windows with equal numbers of SNPs
    pos = variants_seg['POS'][:][this_chrom_variant]
    windows = allel.stats.moving_statistic(pos, statistic=lambda v: [v[0], v[-1]], size=100)
    x = np.asarray(windows).mean(axis=1)

    # compute Tajima's D
    y1, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['WA'].subset(this_chrom_variant), windows=windows)
    y2, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['SA'].subset(this_chrom_variant), windows=windows)
    y3, _, _ = allel.stats.windowed_tajima_d(pos, ac_seg['SEA'].subset(this_chrom_variant), windows=windows)

    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    ax.plot(x, y1, lw=.5, label='WA')
    ax.plot(x, y2, lw=.5, label='SA')
    ax.plot(x, y3, lw=.5, label='SEA')
    ax.set_ylabel("Tajima's $D$")
    ax.set_xlabel('Chromosome %s position (bp)' % current_chrom.decode('ascii'))
    ax.set_xlim(0, pos.max())
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1));

ac = ac_seg['all'][:]
ac

pca_selection = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 2)
pca_selection

np.count_nonzero(pca_selection)

indices = np.nonzero(pca_selection)[0]
indices

len(indices)

indices_ds = np.random.choice(indices, size=50000, replace=False)
indices_ds.sort()
indices_ds

genotypes_pca = genotypes_seg.take(indices_ds, axis=0)
genotypes_pca

gn = genotypes_pca.to_n_alt()[:]
gn

coords, model = allel.stats.pca(gn)

coords

coords.shape

def plot_pca_coords(coords, model, pc1, pc2, ax):
    x = coords[:, pc1]
    y = coords[:, pc2]
    for pop in ['1_WA', '6_SA', '4_SEA']:
        flt = (samples_subset.continent == pop).values
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=pop, markersize=6)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))

fig, ax = plt.subplots(figsize=(6, 6))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax)
ax.legend();

def plot_pca_coords(coords, model, pc1, pc2, ax):
    x = coords[:, pc1]
    y = coords[:, pc2]
    for study in ['1013-PF-PEGB-BRANCH', '1023-PF-CO-ECHEVERRI-GARCIA', '1145-PF-PE-GAMBOA', '1134-PF-ML-CONWAY', '1025-PF-KH-PLOWE', '1010-PF-TH-ANDERSON']:
        flt = (samples_subset.study == study).values
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=study, markersize=6)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))

fig, ax = plt.subplots(figsize=(6, 6))
sns.despine(ax=ax, offset=10)
plot_pca_coords(coords, model, 0, 1, ax)
ax.legend();

samples.index[sample_selection][(coords[:, 0] > -50) & (coords[:, 0] < -20)]

samples.index[sample_selection][coords[:, 0] > 0]

samples.index[sample_selection][coords[:, 1] > 0]

coords[(samples.index == 'PP0012-C')[sample_selection]]

coords[(samples.index == 'PP0022-C')[sample_selection]]

coords[(samples.index == 'PP0022-Cx')[sample_selection]]

coords[(samples.index == 'PD0047-C')[sample_selection]]

coords[(samples.index == 'PP0018-C')[sample_selection]]

fig, ax = plt.subplots(figsize=(5, 4))
sns.despine(ax=ax, offset=10)
y = 100 * model.explained_variance_ratio_
x = np.arange(len(y))
ax.set_xticks(x + .4)
ax.set_xticklabels(x + 1)
ax.bar(x, y)
ax.set_xlabel('Principal component')
ax.set_ylabel('Variance explained (%)');

x = np.array([0, 4, 7])
x

x.ndim

x.shape

x.dtype

# item access
x[1]

# slicing
x[0:2]

y = np.array([1, 6, 9])
x + y

g = allel.GenotypeArray([[[0, 0], [0, 1]],
                         [[0, 1], [1, 1]],
                         [[0, 2], [-1, -1]]], dtype='i1')
g

isinstance(g, np.ndarray)

g.ndim

g.shape

# obtain calls for the second variant in all samples
g[1, :]

# obtain calls for the second sample in all variants
g[:, 1]

# obtain the genotype call for the second variant, second sample
g[1, 1]

# make a subset with only the first and third variants
g.take([0, 2], axis=0)

# find missing calls
np.any(g < 0, axis=2)

g.n_variants, g.n_samples, g.ploidy

g.count_alleles()

genotypes

genotypes.chunks

genotypes_subset

import datetime
print(datetime.datetime.now().isoformat())

