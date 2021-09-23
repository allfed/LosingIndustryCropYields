  topsoil30snew.asc | sed 's/../& /g' > bigdump.txt
 # cat asfdb.txt | sed 's/../& /g' > bigdump.txt
 cat bigdump.txt | sed 's/10/16/g'| sed 's/0a/10/g'| sed ' s/0b/11/g'| sed 's/0c/12/g'| sed 's/0d/13/g'| sed 's/0e/14/g'| sed 's/0f/15/g' > dumpwithnewlines.txt
 
# xxd -b -p topsoil30snew.asc \
# | sed 's/../& /g'\
# | sed 's/10/16/g'\
# | sed 's/0a/10/g'\
# | sed ' s/0b/11/g'\
# | sed 's/0c/12/g'\
# | sed 's/0d/13/g'\
# | sed 's/0e/14/g'\
# | sed 's/0f/15/g'\
# > output.txt

sed -i '1s/^/ncols 43200\nnrows 21600\nxllcorner -180\nyllcorner -90\ncellsize 0.008333333333\nNODATA_value -9\n/' dumpwithnewlines.txt

# cat testdump.txt \
# | tr -d "\n\r" \

# cat bigdump.txt  \
# | sed 's/10/16/g'\
# | sed 's/0a/10/g'\
# | sed ' s/0b/11/g'\
# | sed 's/0c/12/g'\
# | sed 's/0d/13/g'\
# | sed 's/0e/14/g'\
# | sed 's/0f/15/g'\
# > output.txt

# sed -e "s/.\{20\}/&\n/g" < temp.txt

# 1: 2356
# 2: 54860
# 3: 585866
# 4: 357721
# 5: 0
# 6: 1099551
# 7: 23786
# 8: 0
# 9: 171561
# 10: 0
# 11: 0
# 12: 51743
# 13: 0
# 14: 1343447
# 15: 0
# 16: 90440