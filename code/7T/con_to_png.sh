dos2unix original/*.con

# rename con files to patient ids
grep Patient_id original/*.con | perl -pe 's/:Patient_id=/ /;chomp;$_="cp $_.con\n"' | bash -

# first convert con files to tsv
for i in *.con
do
	../../../code/7T/con_to_tsv.sh $i $(basename $i .con).tsv
done

# create a file with image resolutions
grep Image_resolution *.con | perl -pe 'BEGIN{print "id\tcolumns\trows\n"}s/.con:Image_resolution=/\t/;s/x/\t/' >resolution.tsv

# create masks in ../masks
python ../../../code/7T/con2png.py
