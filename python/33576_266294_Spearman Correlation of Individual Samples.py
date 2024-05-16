for name, group in thomas.genmap.groupby('genotype'):
    sns.boxplot()
    print(name)
    for p1 in group.project_name:
        for p2 in group.project_name:
            s = sts.spearmanr(thomas.tpm[p1].tpm, thomas.tpm[p2].tpm)
            print('{0}, {1}, {2:2g}'.format(p1, p2, s[0]))

