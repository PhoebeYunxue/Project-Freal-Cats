from collections import Counter

def analyse(result, submodelID):
    print(result)
    if result != "Can not detect":
        detectedImg = result[0]
        labels = []
        scores = []
        if len(result) == 2:
            print(result[1])
            for item in result[1]:
                label, score = item
                labels.append(label)
                scores.append(score)
        elif len(result) == 3:
            labels = result[1]
            scores = result[2]
        listStr = labels[0].split('_')
        area = listStr[0]

        if "other" in labels:
            if submodelID == "A":
                other_cats = "swirl_august"
                floder = "annya"
            elif submodelID == "C":
                other_cats = "c_swirl_van, c_tabby_fanny"
                floder = "c"
            elif submodelID == "H":
                other_cats = "hotspur_swirl_4, hotspur_tabby_4"
                floder = "hotspur"
            elif submodelID == "M":
                other_cats = "m_tabby_stripefacedbastard-1"
                floder = "m"
            elif submodelID == "O":
                floder = 'otway'
                other_cats = "ginger_ghost, swirl_belinda, swirl_bertha, swirl_faith, swirl_jill, swirl_leppy, swirl_pam, swirl_tim, tabby_bootscootinbaby, tabby_mohammed, tabby_oswald, tabby_tyrone"

            return "demo_other.html", detectedImg, labels, scores, other_cats, area, floder
        else:
            return 'demo_ok.html', detectedImg, labels, scores, area


    else:
        return "demo_faild.html"

def compareAllModel(result_A,result_C,result_H,result_M,result_O):
    total ={}
    labels_A=[]
    scores_A =[]
    labels_C = []
    scores_C = []
    labels_H = []
    scores_H = []
    labels_M = []
    scores_M = []
    labels_O = []
    scores_O = []
    if result_A[0] == "demo_other.html":
        page_A, detectedImg_A, labels_A, scores_A, other_cats_A, area_A, floder_A = result_A
    elif result_A[0] == "demo_ok.html":
        page_A, detectedImg_A, labels_A, scores_A, area_A = result_A
    if len(labels_A) >0:
        for label, score in zip(labels_A, scores_A):
            total[label] = score
    if result_C[0] == "demo_other.html":
        page_C, detectedImg_C, labels_C, scores_C, other_cats_C, area_C, floder_C = result_C
    elif result_C[0] == "demo_ok.html":
        page_C, detectedImg_C, labels_C, scores_C, area_C = result_C
    if len(labels_C) > 0:
        for label, score in zip(labels_C, scores_C):
            total[label] = score
    if result_H[0] == "demo_other.html":
        page_H, detectedImg_H, labels_H, scores_H, other_cats_H, area_H, floder_H = result_H
    elif result_H[0] == "demo_ok.html":
        page_H, detectedImg_H, labels_H, scores_H, area_H = result_H
    if len(labels_H) > 0:
        for label, score in zip(labels_H, scores_H):
            total[label] = score
    if result_M[0] == "demo_other.html":
        page_M, detectedImg_M, labels_M, scores_M, other_cats_M, area_M, floder_M = result_M
    elif result_M[0] == "demo_ok.html":
        page_M, detectedImg_M, labels_M, scores_M, area_M = result_M
    if len(labels_M) > 0:
        for label, score in zip(labels_M, scores_M):
            total[label] = score
    if result_O[0] == "demo_other.html":
        page_O, detectedImg_O, labels_O, scores_O, other_cats_O, area_O, floder_O = result_O
    elif result_O[0] == "demo_ok.html":
        page_O, detectedImg_O, labels_O, scores_O, area_O = result_O
    if len(labels_O) > 0:
        for label, score in zip(labels_O, scores_O):
            total[label] = score

    k = Counter(total)
    high = k.most_common(3)

    labels = []
    scores = []
    areas = []
    other_cats = []
    for i in high:
        labels.append(i[0])
        if i[0] in labels_A:
            areas.append(area_A)
            if i[0] == "other":
                other_cats.append(other_cats_A)
        if i[0] in labels_C:
            areas.append(area_C)
            if i[0] == "other":
                other_cats.append(other_cats_C)
        if i[0] in labels_H:
            areas.append(area_H)
            if i[0] == "other":
                other_cats.append(other_cats_H)
        if i[0] in labels_M:
            areas.append(area_M)
            if i[0] == "other":
                other_cats.append(other_cats_M)
        if i[0] in labels_O:
            areas.append(area_O)
            if i[0] == "other":
                other_cats.append(other_cats_O)
        scores.append(i[1])
        print(i[0]," :",i[1]," ")
    if labels[0] in labels_A:
        detectedImg = detectedImg_A
    if labels[0] in labels_C:
        detectedImg = detectedImg_C
    if labels[0] in labels_H:
        detectedImg = detectedImg_H
    if labels[0] in labels_M:
        detectedImg = detectedImg_M
    if labels[0] in labels_O:
        detectedImg = detectedImg_O

    print(areas)
    if len(labels) == 0:
        return "demo_faild.html"
    if len(other_cats) != 0:
        return 'demo_all.html', detectedImg, labels, scores, areas, other_cats
    return 'demo_all.html', detectedImg, labels, scores, areas

