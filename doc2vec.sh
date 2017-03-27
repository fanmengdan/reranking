mode="dbow"
meph=10
neph="5,10"
ephs=(5 10)
wndw=(10 15 20)
dimen=200

for w in "${wndw[@]}"
do
    printf '######################## python train.py %3d %3d %s %s %3d ######################\n' "$w" "$meph" "$neph" "$mode" "$dimen"
    python train.py $w $meph $neph $mode $dimen
    printf '\n\n\n#######################################################################\n\n\n'
done

for w in "${wndw[@]}"
do
    for n in "${ephs[@]}"
    do
        printf '######################## python test.py %3d %3d %s %3d #######################\n' "$w" "$n" "$mode" "$dimen"
        python test.py $w $n $mode $dimen
        printf '\n\n\n#######################################################################\n\n\n'
    done
done
