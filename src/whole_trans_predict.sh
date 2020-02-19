for sample in {0..6};do
    #for types in {2}; do
    types=2
        python transcriptome_prediction.py $1 ${sample} ${types} $2
    #done
done
