
python3 run_multigroup_sensitivity.py $(cat $1 | grep -v \#)  $(echo $@ | awk {'$1 = ""; print $0'})