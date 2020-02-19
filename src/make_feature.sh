tyes="0 1"
ms="0 1 2"
for sample in {0..6};do
    for t in ${tyes}; do
        for model in ${ms}; do
 
            python make_features.py ${sample} 3 $t ${model}
        done
    done
done
