solver=("sgd" "adam")
activation=("relu" "tanh" "logistic")

rm -f out/train-nn/validation.accuracy
touch out/train-nn/validation.accuracy

rm -f out/train-nn/testing.accuracy
touch out/train-nn/testing.accuracy

for s in ${solver[@]}
do
    for a in ${activation[@]}
    do
        i=0
        while [ $i -lt 4 ];
        do
            j=$i
            while [ $j -le 4 ];
            do
                k=$j
                while [ $k -le 4 ];
                do
                    output=$(/usr/bin/time -f "\ntime %e\n" python model.py $s $a $i $j $k 2>&1)
                    time=$(echo $output | grep -o 'time [0-9.]*' | grep -o [0-9.]*)
                    printf 'bash %s %s %d %d %d %s\n' "$s" "$a" "$i" "$j" "$k" "$time"
                    python test/scorer/ev.py test/2016/gold/dev-train-subtaskA.2016.relevancy \
                            test/2016/pred/validation-subtaskA.2016.pred \
                            > out/train-nn/validation-$s-$a-$i-$j-$k.accuracy
                    MAP=$(cat out/train-nn/validation-$s-$a-$i-$j-$k.accuracy | grep score | grep -o [0-9.]*)
                    Acc=$(cat out/train-nn/validation-$s-$a-$i-$j-$k.accuracy | grep Acc | grep -o [0-9.]*)
                    P=$(cat out/train-nn/validation-$s-$a-$i-$j-$k.accuracy | grep "P   =" | grep -o [0-9.]*)
                    R=$(cat out/train-nn/validation-$s-$a-$i-$j-$k.accuracy | grep "R   =" | grep -o [0-9.]*)
                    F1=$(cat out/train-nn/validation-$s-$a-$i-$j-$k.accuracy | grep F1 | grep -o [0-9.]* | tail -n 1)
                    AvgRec=$(cat out/train-nn/validation-$s-$a-$i-$j-$k.accuracy | grep AvgRec | grep -o [0-9.]* | tail -n 1)
                    MRR=$(cat out/train-nn/validation-$s-$a-$i-$j-$k.accuracy | grep MRR | grep -o [0-9.]* | tail -n 1)
                    printf '%s,%s,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                        "$s" "$a" "$i" "$j" "$k" \
                        "$MAP" "$AvgRec" "$MRR" "$P" "$R" "$F1" "$Acc" "$time" \
                        >> out/train-nn/validation.accuracy
                    python test/scorer/ev.py test/2016/gold/SemEval2016-Task3-CQA-QL-test-subtaskA.xml.subtaskA.relevancy \
                            test/2016/pred/test-subtaskA.2016.pred \
                            > out/train-nn/testing-$s-$a-$i-$j-$k.accuracy
                    MAP=$(cat out/train-nn/testing-$s-$a-$i-$j-$k.accuracy | grep score | grep -o [0-9.]*)
                    Acc=$(cat out/train-nn/testing-$s-$a-$i-$j-$k.accuracy | grep Acc | grep -o [0-9.]*)
                    P=$(cat out/train-nn/testing-$s-$a-$i-$j-$k.accuracy | grep "P   =" | grep -o [0-9.]*)
                    R=$(cat out/train-nn/testing-$s-$a-$i-$j-$k.accuracy | grep "R   =" | grep -o [0-9.]*)
                    F1=$(cat out/train-nn/testing-$s-$a-$i-$j-$k.accuracy | grep F1 | grep -o [0-9.]* | tail -n 1)
                    AvgRec=$(cat out/train-nn/testing-$s-$a-$i-$j-$k.accuracy | grep AvgRec | grep -o [0-9.]* | tail -n 1)
                    MRR=$(cat out/train-nn/testing-$s-$a-$i-$j-$k.accuracy | grep MRR | grep -o [0-9.]* | tail -n 1)
                    printf '%s,%s,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                        "$s" "$a" "$i" "$j" "$k" \
                        "$MAP" "$AvgRec" "$MRR" "$P" "$R" "$F1" "$Acc" "$time" \
                        >> out/train-nn/testing.accuracy
                    printf '%s\n' "$output"
                    let k=k+1
                done
                let j=j+1
            done
            let i=i+1
        done
    done
done