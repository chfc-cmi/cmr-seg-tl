for j in ../dicom/*
do
	ID=$(echo $j | rev | cut -c1,2 | rev)
	echo $ID
	# check InPlanePhaseEncodingDirection
	if [ $(dcmdump $(ls $j/*/*.dcm | head -n1) | grep InPlanePhaseEncodingDirection | grep -c COL) -ne 0 ]
	then
		SLICE=$(ls $j | cut -f1 -d"_" | sort -nr | head -n1)
		for i in $(ls $j/*/*.dcm | sort -V)
		do
			out=$(echo $i | perl -ne '/(\d+)(_c)?_gre/;printf "'$ID'_slice%03d",('$SLICE'-$1);/i(\d+).dcm/;printf "_frame%03d-image.png\n",($1-1)')
			echo dcm2pnm --write-png --use-window 1 $i images/$out
			cp $i dcms/$(basename $out .png).dcm
		done
	else
		SLICE=$(ls $j | cut -f1 -d"_" | sort -n | head -n1)
		for i in $(ls $j/*/*.dcm | sort -V)
		do
			out=$(echo $i | perl -ne '/(\d+)(_c)?_gre/;printf "'$ID'_slice%03d",($1-'$SLICE');/i(\d+).dcm/;printf "_frame%03d-image.png\n",($1-1)')
			echo dcm2pnm --write-png --use-window 1 $i images/$out
			cp $i dcms/$(basename $out .png).dcm
		done
	fi
done
