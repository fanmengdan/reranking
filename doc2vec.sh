mode="dbow"
meph=25
neph="1,5,10,25"
ephs=(1 5 10 25)
wndw=(10 15 20 25 30 35 40 45 50)

for w in "${wndw[@]}"
do
    printf '######################## python train.py %3d %3d %s %s ######################\n' "$w" "$meph" "$neph" "$mode"
    python train.py $w $meph $neph $mode
    printf '\n\n\n#######################################################################\n\n\n'
done

for w in "${wndw[@]}"
do
    for n in "${ephs[@]}"
    do
        printf '######################## python test.py %3d %3d %s %s #######################\n' "$w" "$n" "$mode"
        python test.py $w $n $mode
        printf '\n\n\n#######################################################################\n\n\n'
    done
done