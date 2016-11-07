neph=(10 50 100)
wndw=(10 15 20 25 30 35 40 45 50)

for n in "${neph[@]}"
do
    for w in "${wndw[@]}"
    do
        printf '######################## python train.py %3d %3d ######################\n' "$w" "$n"
    printf '#######################################################################\n\n\nccc'
        python train.py $w $n
    printf '\n\n\n#######################################################################\n\n\n'

        printf '######################## python test.py %3d %3d #######################\n' "$w" "$n"
    printf '#######################################################################\n\n\nccc'
    python test.py $w $n
    printf '\n\n\n#######################################################################\n\n\n'
    done
done

######################## VALIDATION ACCURACY ########################
python scorer/ev.py 2016/gold/dev-train-subtaskA.2016.relevancy 2016/pred/validation-subtaskA.2016.pred

######################### TESTING ACCURACY #########################
python scorer/ev.py 2016/gold/SemEval2016-Task3-CQA-QL-test-subtaskA.xml.subtaskA.relevancy 2016/pred/test-subtaskA.2016.pred