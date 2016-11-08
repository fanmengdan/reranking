# neph=(10 50 100)
# wndw=(10 15 20 25 30 35 40 45 50)

# for n in "${neph[@]}"
# do
#     for w in "${wndw[@]}"
#     do
#         printf '######################## python train.py %3d %3d ######################\n' "$w" "$n"
#     printf '#######################################################################\n\n\nccc'
#         python train.py $w $n
#     printf '\n\n\n#######################################################################\n\n\n'

#         printf '######################## python test.py %3d %3d #######################\n' "$w" "$n"
#     printf '#######################################################################\n\n\nccc'
#     python test.py $w $n
#     printf '\n\n\n#######################################################################\n\n\n'
#     done
# done

solver=("lbfgs" "sgd" "adam")
activation=("relu" "tanh" "logistic")

rm -f out/train-nn/validation.accuracy
touch out/train-nn/validation.accuracy

rm -f out/train-nn/testing.accuracy
touch out/train-nn/testing.accuracy

for s in ${solver[@]}
do
    for a in ${activation[@]}
    do
        i=3
        while [ $i -lt 4 ];
        do
            j=$i
            while [ $j -le 4 ];
            do
                k=$j
                while [ $k -le 4 ];
                do
                    l=$k
                    while [ $l -le 4 ];
                    do
                        output=$(/usr/bin/time -f "\ntime %e\n" python model.py $s $a $i $j $k $l 2>&1)
                        time=$(echo $output | grep -o 'time [0-9.]*' | grep -o [0-9.]*)
                        # printf 'bash %s %s %d %d %d %d %s\n' "$s" "$a" "$i" "$j" "$k" "$l" "$time"
                        python test/scorer/ev.py test/2016/gold/dev-train-subtaskA.2016.relevancy \
                                test/2016/pred/validation-subtaskA.2016.pred \
                                > out/train-nn/validation-$s-$a-$i-$j-$k-$l.accuracy
                        MAP=$(cat validation-$s-$a-$i-$j-$k-$l.accuracy | grep score | grep -o [0-9.]*)
                        Acc=$(cat validation-$s-$a-$i-$j-$k-$l.accuracy | grep Acc | grep -o [0-9.]*)
                        P=$(cat validation-$s-$a-$i-$j-$k-$l.accuracy | grep "P   =" | grep -o [0-9.]*)
                        R=$(cat validation-$s-$a-$i-$j-$k-$l.accuracy | grep "R   =" | grep -o [0-9.]*)
                        F1=$(cat validation-$s-$a-$i-$j-$k-$l.accuracy | grep F1 | grep -o [0-9.]* | tail -n 1)
                        AvgRec=$(cat validation-$s-$a-$i-$j-$k-$l.accuracy | grep AvgRec | grep -o [0-9.]* | tail -n 1)
                        MRR=$(cat validation-$s-$a-$i-$j-$k-$l.accuracy | grep MRR | grep -o [0-9.]* | tail -n 1)
                        printf '%s,%s,%d,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s' \
                            "$s" "$a" "$i" "$j" "$k" "$l" \
                            "$MAP" "$AvgRec" "$MRR" "$P" "$R" "$F1" "$Acc" "$time" \
                            >> out/train-nn/validation.accuracy
                        python test/scorer/ev.py test/2016/gold/SemEval2016-Task3-CQA-QL-test-subtaskA.xml.subtaskA.relevancy \
                                test/2016/pred/test-subtaskA.2016.pred \
                                > out/train-nn/testing-$s-$a-$i-$j-$k-$l.accuracy
                        MAP=$(cat testing-$s-$a-$i-$j-$k-$l.accuracy | grep score | grep -o [0-9.]*)
                        Acc=$(cat testing-$s-$a-$i-$j-$k-$l.accuracy | grep Acc | grep -o [0-9.]*)
                        P=$(cat testing-$s-$a-$i-$j-$k-$l.accuracy | grep "P   =" | grep -o [0-9.]*)
                        R=$(cat testing-$s-$a-$i-$j-$k-$l.accuracy | grep "R   =" | grep -o [0-9.]*)
                        F1=$(cat testing-$s-$a-$i-$j-$k-$l.accuracy | grep F1 | grep -o [0-9.]* | tail -n 1)
                        AvgRec=$(cat testing-$s-$a-$i-$j-$k-$l.accuracy | grep AvgRec | grep -o [0-9.]* | tail -n 1)
                        MRR=$(cat testing-$s-$a-$i-$j-$k-$l.accuracy | grep MRR | grep -o [0-9.]* | tail -n 1)
                        printf '%s,%s,%d,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s' \
                            "$s" "$a" "$i" "$j" "$k" "$l" \
                            "$MAP" "$AvgRec" "$MRR" "$P" "$R" "$F1" "$Acc" "$time" \
                            >> out/train-nn/testing.accuracy
                        echo $output
                        let l=l+1
                    done
                    let k=k+1
                done
                let j=j+1
            done
            let i=i+1
        done
    done
done