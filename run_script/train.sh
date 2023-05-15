filenames=('AgNews' 'StackOverflow' 'SearchSnippets' 'Biomedical' 'GooglenewsTS_trans_subst_30' 'GooglenewsT' 'GooglenewsS' 'Tweet')
DataSize=(8000 20000 12340 20000 11109 11108 11108 2472)
classNum=(4 20 8 20 152 152 152 89)
maxLen=(32 25 32 45 40 16 32 20)
epoch=100

id=(0 1 2 3 4 5 6 7)

for i in ${id[*]}
do
echo "I am good at ${filenames[$i]}"
python ./train.py \
    --batch_size 800 \
    --epoch $epoch \
    --data_name ${filenames[$i]} \
    --class_num ${classNum[$i]} \
    --max_len ${maxLen[$i]} \
    --resume 0 \
    --use_noise 0

done

