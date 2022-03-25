from datasets.bp4d.config import AU_ORDER, SEX_ORDER, ETHNICITY_ORDER, TASK_ORDER

def columns_bp4d(labels, **kwargs):
    columns = []

    if 'AU_binary' in labels:
        columns = AU_ORDER
    else:
        if 'AU' in labels:
            for au in AU_ORDER:
                columns += [au, 'no_{}'.format(au)]
        if 'SEX' in labels:
            columns += SEX_ORDER
        if 'ETH' in labels:
            columns += ETHNICITY_ORDER
        if 'TASK' in labels:
            columns += TASK_ORDER
    return columns 

