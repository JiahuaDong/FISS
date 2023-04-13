tasks_voc = {
    "15-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16],
            2: [17],
            3: [18],
            4: [19],
            5: [20]
        },

    "4-4":
        {
            0: [0, 1, 2, 3, 4],
            1: [5, 6, 7, 8],
            2: [9, 10, 11, 12],
            3: [13, 14, 15, 16],
            4: [17, 18, 19, 20]
        },
        
    "8-2":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            1: [9, 10],
            2: [11, 12],
            3: [13, 14],
            4: [15, 16],
            5: [17, 18],
            6: [19, 20]
        },
}

tasks_ade = {
    "100-10":
        {
            0: [x for x in range(0, 101)],
            1: [x for x in range(101, 111)],
            2: [x for x in range(111, 121)],
            3: [x for x in range(121, 131)],
            4: [x for x in range(131, 141)],
            5: [x for x in range(141, 151)]
        },
}


def get_task_list():
    return list(tasks_voc.keys()) + list(tasks_ade.keys())


def get_task_labels(dataset, name, step):
    if dataset == 'voc':
        task_dict = tasks_voc[name]
    elif dataset == 'ade':
        task_dict = tasks_ade[name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    labels_old = [label for s in range(step) for label in task_dict[s]]
    return labels, labels_old, f'data/{dataset}/{name}'

def get_per_task_classes(dataset, name, step):
    if dataset == 'voc':
        task_dict = tasks_voc[name]
    elif dataset == 'ade':
        task_dict = tasks_ade[name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    classes = [len(task_dict[s]) for s in range(step + 1)]
    return classes
